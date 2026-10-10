"""Train Stage-2 LightGBM LambdaRanker on Stage-1 Candidates with Hard Negatives and No Leakage."""
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import torch

from ml.config import PipelineConfig
from ml.data.dataset import MovieLensData, NUM_GENRES
from ml.models.two_tower import TwoTowerModel
from ml.models.vector_index import VectorIndex
from ml.features.feature_engineer import FeatureEngineer, FEATURE_NAMES
from ml.models.ranker import LambdaRanker
from ml.evaluate import (
    Evaluator,
    hit_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
    catalog_coverage,
    intra_list_diversity,
    novelty_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_ranking_dataset(
    data: MovieLensData,
    model: TwoTowerModel,
    index: VectorIndex,
    feature_engineer: FeatureEngineer,
    config: PipelineConfig,
    num_users_to_sample: int = 4000,
    max_cutoffs_per_user: int = 5,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, Any]]:
    """
    Generate training pairs for LambdaRanker without temporal leakage:
    Uses several time cutoffs per user from their training history.
    At cutoff t, history strictly precedes t, and remaining items in train_df
    serve as positive labels with graded relevance (5 -> 2, 4 -> 1, unobserved -> 0).
    Only Stage-1 retrieved candidates (k=200) are ranked, exactly matching inference.
    """
    logger.info("Generating candidate pairs with multiple time cutoffs (no leakage)...")
    all_features = []
    all_labels = []
    group_sizes = []

    train_sorted = data.train_df.sort_values(by=["u_idx", "timestamp"])
    user_train_records = {}
    for u_idx, grp in train_sorted.groupby("u_idx"):
        user_train_records[u_idx] = grp[["i_idx", "rating", "timestamp"]].to_dict("records")

    train_users = sorted(list(user_train_records.keys()))[:num_users_to_sample]
    total_groups_attempted = 0
    groups_with_positive = 0

    for u_idx in train_users:
        records = user_train_records[u_idx]
        n_rec = len(records)
        if n_rec < 6:
            continue

        u_meta = data.user_meta[u_idx]
        num_cutoffs = min(max_cutoffs_per_user, n_rec - 3)
        cutoff_indices = range(n_rec - num_cutoffs, n_rec)

        for c_idx in cutoff_indices:
            total_groups_attempted += 1
            hist_records = records[:c_idx]
            hist_items = [r["i_idx"] for r in hist_records]
            hist_ratings = [r["rating"] for r in hist_records]

            # Future items in train_df starting from c_idx
            future_ratings = {r["i_idx"]: r["rating"] for r in records[c_idx:]}

            # Build user representation strictly from history before cutoff
            last_n_hist = hist_items[-config.data.last_n_watched:]
            pad_len = config.data.last_n_watched - len(last_n_hist)
            padded_hist = [data.num_items] * pad_len + last_n_hist

            pos_hist = [r["i_idx"] for r in hist_records if r["rating"] >= config.data.positive_rating_threshold]
            if pos_hist:
                g_sum = data.movie_genres_matrix[pos_hist].sum(axis=0)
                g_norm = np.linalg.norm(g_sum)
                u_genres = g_sum / g_norm if g_norm > 0 else np.zeros(NUM_GENRES, dtype=np.float32)
            else:
                u_genres = np.zeros(NUM_GENRES, dtype=np.float32)

            u_mean_rating = float(np.mean(hist_ratings)) if hist_ratings else 3.5

            with torch.no_grad():
                u_emb = model.user_tower(
                    user_ids=torch.tensor([u_idx], dtype=torch.long),
                    gender_idx=torch.tensor([1 if u_meta["gender"] == "F" else 0], dtype=torch.long),
                    age_idx=torch.tensor([min(6, max(0, u_meta["age"] // 10))], dtype=torch.long),
                    occ_idx=torch.tensor([min(21, max(0, u_meta["occupation"]))], dtype=torch.long),
                    mean_rating=torch.tensor([u_mean_rating], dtype=torch.float32),
                    top_genres=torch.from_numpy(u_genres[np.newaxis, :]),
                    history_items=torch.tensor([padded_hist], dtype=torch.long),
                ).cpu().numpy()[0]

            # Stage 1 Retrieval: top 200 candidates excluding history items
            ex = set(hist_items)
            query_k = min(data.num_items, config.retrieval.top_k_candidates + len(ex) + 10)
            _, cand_indices = index.search(u_emb, k=query_k)

            cands = [int(i) for i in cand_indices if int(i) >= 0 and int(i) not in ex][
                : config.retrieval.top_k_candidates
            ]
            if len(cands) < config.retrieval.top_k_candidates:
                continue

            # Assign graded relevance: rating >= 5 -> 2, rating >= 4 -> 1, unobserved -> 0
            labels = []
            has_pos = False
            for c in cands:
                if c in future_ratings:
                    r = future_ratings[c]
                    if r >= 5.0:
                        labels.append(2.0)
                        has_pos = True
                    elif r >= 4.0:
                        labels.append(1.0)
                        has_pos = True
                    else:
                        labels.append(0.0)
                else:
                    labels.append(0.0)

            if has_pos:
                groups_with_positive += 1
                feats = feature_engineer.extract_features_for_user_candidates(
                    u_idx=u_idx,
                    u_emb=u_emb,
                    candidate_items=cands,
                    history_items=hist_items,
                    history_ratings=hist_ratings,
                    user_genres=u_genres,
                    user_mean_rating=u_mean_rating,
                )
                assert feats.shape[0] == len(cands) == len(labels), "Row alignment error in training group!"
                all_features.append(feats)
                all_labels.extend(labels)
                group_sizes.append(len(cands))

    X = np.vstack(all_features)
    y = np.array(all_labels, dtype=np.float32)
    assert sum(group_sizes) == len(X) == len(y), "Row and group alignment assertion failed!"

    stats = {
        "total_groups_attempted": total_groups_attempted,
        "groups_with_positive": groups_with_positive,
        "share_groups_with_positive": float(round(groups_with_positive / max(1, total_groups_attempted), 4)),
        "num_rows": len(y),
        "num_groups": len(group_sizes),
    }
    logger.info(
        f"Training dataset: {len(X)} rows across {len(group_sizes)} groups. "
        f"Positive groups: {groups_with_positive}/{total_groups_attempted} ({stats['share_groups_with_positive']*100:.2f}%)."
    )
    return X, y, group_sizes, stats


def generate_validation_dataset(
    data: MovieLensData,
    model: TwoTowerModel,
    index: VectorIndex,
    feature_engineer: FeatureEngineer,
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[str, Any]]:
    """
    Generate validation pairs from val_df (clean leave-second-out per user).
    Used for early stopping on validation NDCG@10. Never uses test_df.
    """
    logger.info("Generating validation dataset from val_df...")
    val_ratings = data.val_df.set_index(["u_idx", "i_idx"])["rating"].to_dict()
    val_users = sorted(list(data.val_df["u_idx"].unique()))

    train_sorted = data.train_df.sort_values(by=["u_idx", "timestamp"])
    user_train_records = {}
    for u_idx, grp in train_sorted.groupby("u_idx"):
        user_train_records[u_idx] = grp[["i_idx", "rating"]].to_dict("records")

    all_features = []
    all_labels = []
    group_sizes = []
    total_val_groups = 0
    val_groups_with_pos = 0

    for u_idx in val_users:
        total_val_groups += 1
        u_meta = data.user_meta[u_idx]
        hist_items = data.user_history_i_idx.get(u_idx, [])
        hist_ratings = [r["rating"] for r in user_train_records.get(u_idx, [])]
        last_n_hist = hist_items[-config.data.last_n_watched:]
        pad_len = config.data.last_n_watched - len(last_n_hist)
        padded_hist = [data.num_items] * pad_len + last_n_hist

        u_genres = data.user_top_genres[u_idx]
        u_mean_rating = float(data.user_mean_rating[u_idx])

        with torch.no_grad():
            u_emb = model.user_tower(
                user_ids=torch.tensor([u_idx], dtype=torch.long),
                gender_idx=torch.tensor([1 if u_meta["gender"] == "F" else 0], dtype=torch.long),
                age_idx=torch.tensor([min(6, max(0, u_meta["age"] // 10))], dtype=torch.long),
                occ_idx=torch.tensor([min(21, max(0, u_meta["occupation"]))], dtype=torch.long),
                mean_rating=torch.tensor([u_mean_rating], dtype=torch.float32),
                top_genres=torch.from_numpy(u_genres[np.newaxis, :]),
                history_items=torch.tensor([padded_hist], dtype=torch.long),
            ).cpu().numpy()[0]

        ex = set(hist_items)
        query_k = min(data.num_items, config.retrieval.top_k_candidates + len(ex) + 10)
        _, cand_indices = index.search(u_emb, k=query_k)
        cands = [int(i) for i in cand_indices if int(i) >= 0 and int(i) not in ex][
            : config.retrieval.top_k_candidates
        ]
        if len(cands) < config.retrieval.top_k_candidates:
            continue

        labels = []
        has_pos = False
        for c in cands:
            if (u_idx, c) in val_ratings:
                r = val_ratings[(u_idx, c)]
                if r >= 5.0:
                    labels.append(2.0)
                    has_pos = True
                elif r >= 4.0:
                    labels.append(1.0)
                    has_pos = True
                else:
                    labels.append(0.0)
            else:
                labels.append(0.0)

        if has_pos:
            val_groups_with_pos += 1
            feats = feature_engineer.extract_features_for_user_candidates(
                u_idx=u_idx,
                u_emb=u_emb,
                candidate_items=cands,
                history_items=hist_items,
                history_ratings=hist_ratings,
                user_genres=u_genres,
                user_mean_rating=u_mean_rating,
            )
            assert feats.shape[0] == len(cands) == len(labels), "Row alignment error in validation group!"
            all_features.append(feats)
            all_labels.extend(labels)
            group_sizes.append(len(cands))

    X_val = np.vstack(all_features)
    y_val = np.array(all_labels, dtype=np.float32)
    assert sum(group_sizes) == len(X_val) == len(y_val), "Validation row alignment failed!"

    stats = {
        "total_val_groups": total_val_groups,
        "val_groups_with_pos": val_groups_with_pos,
        "share_val_groups_with_pos": float(round(val_groups_with_pos / max(1, total_val_groups), 4)),
    }
    logger.info(
        f"Validation dataset: {len(X_val)} rows across {len(group_sizes)} groups. "
        f"Positive groups: {val_groups_with_pos}/{total_val_groups} ({stats['share_val_groups_with_pos']*100:.2f}%)."
    )
    return X_val, y_val, group_sizes, stats


def train_and_evaluate_ranker(
    config: PipelineConfig,
) -> Tuple[LambdaRanker, Dict[str, Any]]:
    """Train ranker, run full two-stage evaluation, ablations, and latency benchmarking."""
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # 1. Load data
    data = MovieLensData(
        dataset_name=config.data.dataset_name,
        raw_dir=config.data.raw_dir,
        min_user_interactions=config.data.min_user_interactions,
        positive_rating_threshold=config.data.positive_rating_threshold,
    ).load_and_process()

    # 2. Load Two-Tower and FAISS
    artifacts_dir = Path("artifacts")
    model = TwoTowerModel(
        num_users=data.num_users,
        num_items=data.num_items,
        num_genres=NUM_GENRES,
        embedding_dim=config.retrieval.embedding_dim,
    )
    model.load_state_dict(torch.load(artifacts_dir / "two_tower.pt", weights_only=True))
    model.eval()

    index = VectorIndex(dim=config.retrieval.embedding_dim).load(
        str(artifacts_dir / "faiss_index.bin")
    )
    item_embeddings = np.load(str(artifacts_dir / "item_embeddings.npy"))

    feature_engineer = FeatureEngineer(
        data=data, item_embeddings=item_embeddings, last_n=config.data.last_n_watched
    )

    # 3. Generate Training Set and Validation Set for Ranker
    X_train, y_train, groups_train, train_stats = generate_ranking_dataset(
        data, model, index, feature_engineer, config, num_users_to_sample=4000
    )

    X_val, y_val, groups_val, val_stats = generate_validation_dataset(
        data, model, index, feature_engineer, config
    )

    # 4. Train LambdaRanker with early stopping on validation NDCG@10
    ranker = LambdaRanker(
        n_estimators=config.ranker.n_estimators,
        learning_rate=config.ranker.learning_rate,
        num_leaves=config.ranker.num_leaves,
        min_child_samples=config.ranker.min_child_samples,
        seed=config.seed,
    ).fit(
        X=X_train,
        y=y_train,
        group=groups_train,
        eval_set=(X_val, y_val),
        eval_group=groups_val,
        early_stopping_rounds=20,
    )

    ranker.save(str(artifacts_dir / "ranker.txt"))

    # 5. Evaluate on Validation Set
    val_evaluator = Evaluator(
        test_df=data.val_df,
        total_items=data.num_items,
        item_popularity=data.item_popularity,
        genre_matrix=data.movie_genres_matrix,
        ks=config.eval.ks,
        ranking_k=config.eval.ranking_k,
    )

    # 6. Evaluate Two-Stage Recommender on Test Set
    test_evaluator = Evaluator(
        test_df=data.test_df,
        total_items=data.num_items,
        item_popularity=data.item_popularity,
        genre_matrix=data.movie_genres_matrix,
        ks=config.eval.ks,
        ranking_k=config.eval.ranking_k,
    )

    retrieval_latencies = []
    ranking_latencies = []

    def two_stage_recommend(
        u_idx: int,
        k: int = 10,
        exclude: set = None,
        feature_subset: Optional[List[int]] = None,
        model_to_use: Optional[LambdaRanker] = None,
        return_cands_and_scores: bool = False,
    ):
        u_meta = data.user_meta[u_idx]
        hist = data.user_history_i_idx.get(u_idx, [])[-config.data.last_n_watched :]
        pad_len = config.data.last_n_watched - len(hist)
        padded_hist = [data.num_items] * pad_len + hist

        # Stage 1: Retrieval
        t0 = time.perf_counter()
        with torch.no_grad():
            u_emb = model.user_tower(
                user_ids=torch.tensor([u_idx], dtype=torch.long),
                gender_idx=torch.tensor([1 if u_meta["gender"] == "F" else 0], dtype=torch.long),
                age_idx=torch.tensor([min(6, max(0, u_meta["age"] // 10))], dtype=torch.long),
                occ_idx=torch.tensor([min(21, max(0, u_meta["occupation"]))], dtype=torch.long),
                mean_rating=torch.tensor([u_meta["mean_rating"]], dtype=torch.float32),
                top_genres=torch.from_numpy(data.user_top_genres[u_idx : u_idx + 1]),
                history_items=torch.tensor([padded_hist], dtype=torch.long),
            ).cpu().numpy()[0]

        top_cand_k = config.retrieval.top_k_candidates + len(exclude or set()) + 10
        _, cand_indices = index.search(u_emb, k=min(data.num_items, top_cand_k))
        t1 = time.perf_counter()
        retrieval_latencies.append((t1 - t0) * 1000.0)

        # Filter excluded history and FAISS -1 sentinels
        cands = [int(i) for i in cand_indices if int(i) >= 0 and int(i) not in (exclude or set())][
            : config.retrieval.top_k_candidates
        ]

        # Stage 2: Ranking
        t2 = time.perf_counter()
        cand_feats = feature_engineer.extract_features_for_user_candidates(u_idx, u_emb, cands)
        if feature_subset is not None:
            cand_feats = cand_feats[:, feature_subset]

        active_ranker = model_to_use or ranker
        scores = active_ranker.predict(cand_feats)
        t3 = time.perf_counter()
        ranking_latencies.append((t3 - t2) * 1000.0)

        ranked_indices = np.argsort(-scores)[:k]
        top_k_items = [cands[i] for i in ranked_indices]
        if return_cands_and_scores:
            return top_k_items, cands, scores
        return top_k_items

    # Evaluation on Validation Split
    logger.info("Evaluating Two-Stage on Validation Set...")
    val_metrics = val_evaluator.evaluate_model(
        model_name="Validation: Two-Stage",
        rec_fn=two_stage_recommend,
        user_histories=data.user_history_i_idx,
    )

    # Evaluation on Test Split
    logger.info("Evaluating Two-Stage Recommendation System on Test Set...")
    two_stage_metrics = test_evaluator.evaluate_model(
        model_name="Two-Stage (Two-Tower + Ranker)",
        rec_fn=two_stage_recommend,
        user_histories=data.user_history_i_idx,
    )

    two_stage_metrics["retrieval_latency_p50_ms"] = float(round(np.percentile(retrieval_latencies, 50), 3))
    two_stage_metrics["retrieval_latency_p95_ms"] = float(round(np.percentile(retrieval_latencies, 95), 3))
    two_stage_metrics["ranking_latency_p50_ms"] = float(round(np.percentile(ranking_latencies, 50), 3))
    two_stage_metrics["ranking_latency_p95_ms"] = float(round(np.percentile(ranking_latencies, 95), 3))
    logger.info(f"Two-Stage Test Results: {two_stage_metrics}")

    # 7. Ablation Studies
    logger.info("Running Ablation: Ranker without Two-Tower Similarity score...")
    sub_indices_no_sim = [i for i, name in enumerate(FEATURE_NAMES) if name != "two_tower_sim"]
    X_train_no_sim = X_train[:, sub_indices_no_sim]
    X_val_no_sim = X_val[:, sub_indices_no_sim]
    names_no_sim = [FEATURE_NAMES[i] for i in sub_indices_no_sim]
    ranker_no_sim = LambdaRanker(seed=config.seed).fit(
        X_train_no_sim,
        y_train,
        groups_train,
        feature_names=names_no_sim,
        eval_set=(X_val_no_sim, y_val),
        eval_group=groups_val,
        early_stopping_rounds=20,
    )

    metrics_no_sim = test_evaluator.evaluate_model(
        model_name="Ablation: Ranker w/o Similarity",
        rec_fn=lambda u, k=10, exclude=None: two_stage_recommend(
            u, k, exclude, feature_subset=sub_indices_no_sim, model_to_use=ranker_no_sim
        ),
        user_histories=data.user_history_i_idx,
    )
    logger.info(f"Ablation w/o Sim: {metrics_no_sim}")

    logger.info("Running Ablation: Ranker without User features...")
    user_feat_names = {"user_activity", "user_mean_rating", "genre_affinity"}
    sub_indices_no_user = [i for i, name in enumerate(FEATURE_NAMES) if name not in user_feat_names]
    X_train_no_user = X_train[:, sub_indices_no_user]
    X_val_no_user = X_val[:, sub_indices_no_user]
    names_no_user = [FEATURE_NAMES[i] for i in sub_indices_no_user]
    ranker_no_user = LambdaRanker(seed=config.seed).fit(
        X_train_no_user,
        y_train,
        groups_train,
        feature_names=names_no_user,
        eval_set=(X_val_no_user, y_val),
        eval_group=groups_val,
        early_stopping_rounds=20,
    )

    metrics_no_user = test_evaluator.evaluate_model(
        model_name="Ablation: Ranker w/o User Features",
        rec_fn=lambda u, k=10, exclude=None: two_stage_recommend(
            u, k, exclude, feature_subset=sub_indices_no_user, model_to_use=ranker_no_user
        ),
        user_histories=data.user_history_i_idx,
    )
    logger.info(f"Ablation w/o User Features: {metrics_no_user}")

    # 8. Error Analysis (Cold-Start Users & Feature Importances)
    logger.info("Performing Error Analysis on Subpopulations...")
    user_train_counts = np.array([data.user_activity[u] for u in test_evaluator.eval_users])
    cold_threshold = np.percentile(user_train_counts, 20)  # Bottom 20% active users
    cold_users = [
        test_evaluator.eval_users[i]
        for i in range(len(test_evaluator.eval_users))
        if user_train_counts[i] <= cold_threshold
    ]

    cold_recs = [two_stage_recommend(u, k=10, exclude=set(data.user_history_i_idx.get(u, []))) for u in cold_users]
    cold_hits = [
        hit_at_k(rec, test_evaluator.test_ground_truth[u], k=10)
        for rec, u in zip(cold_recs, cold_users)
    ]
    cold_ndcgs = [
        ndcg_at_k(rec, test_evaluator.test_ground_truth[u], k=10)
        for rec, u in zip(cold_recs, cold_users)
    ]

    error_analysis = {
        "cold_start_users": {
            "num_users": len(cold_users),
            "max_interactions": int(cold_threshold),
            "NDCG@10": float(round(np.mean(cold_ndcgs), 4)),
            "Hit@10": float(round(np.mean(cold_hits), 4)),
        },
        "feature_importances": ranker.get_feature_importances(),
        "training_data_stats": train_stats,
        "validation_data_stats": val_stats,
    }

    # 9. Precompute Winning Users for the Compare page
    logger.info("Computing winning users (Two-Stage ranks held-out movie higher than Retrieval)...")
    winning_users = []
    for u in test_evaluator.eval_users:
        gt_items = test_evaluator.test_ground_truth[u]
        # Held-out item
        held_out_item = list(gt_items)[0]

        _, cands, scores = two_stage_recommend(
            u, k=200, exclude=set(data.user_history_i_idx.get(u, [])), return_cands_and_scores=True
        )
        if held_out_item in cands:
            retrieval_rank = cands.index(held_out_item) + 1
            ranked_indices = np.argsort(-scores)
            ts_ordered = [cands[i] for i in ranked_indices]
            ts_rank = ts_ordered.index(held_out_item) + 1

            if ts_rank < retrieval_rank and ts_rank <= 10:
                user_id = data.idx_to_user[u]
                winning_users.append({
                    "user_id": int(user_id),
                    "ts_rank": int(ts_rank),
                    "retrieval_rank": int(retrieval_rank),
                })

    winning_users.sort(key=lambda x: (x["ts_rank"], -x["retrieval_rank"]))
    winning_users = winning_users[:50]  # Store top 50 winning showcase users
    winners_path = artifacts_dir / "winning_users.json"
    with open(winners_path, "w", encoding="utf-8") as f:
        json.dump(winning_users, f)
    logger.info(f"Saved {len(winning_users)} winning users to {winners_path}")

    # 10. Save updated metrics.json
    results_dir = Path(config.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"

    metrics_db = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_db = json.load(f)

    metrics_db["Two-Stage (Two-Tower + Ranker)"] = two_stage_metrics
    metrics_db["Validation: Two-Stage"] = val_metrics
    metrics_db["Ablation: Ranker w/o Similarity"] = metrics_no_sim
    metrics_db["Ablation: Ranker w/o User Features"] = metrics_no_user
    metrics_db["error_analysis"] = error_analysis

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_db, f, indent=2)

    logger.info(f"Full metrics and ablations successfully saved to {metrics_path}")
    return ranker, metrics_db


if __name__ == "__main__":
    cfg = PipelineConfig.load()
    train_and_evaluate_ranker(cfg)
