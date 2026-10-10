"""Train Stage-2 LightGBM LambdaRanker on Stage-1 Candidates with Hard Negatives."""
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
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
    num_users_to_sample: int = 3000,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Generate training pairs for LambdaRanker:
    Uses Stage 1 retrieval candidates + user's true interactions to construct
    lists of items with positive items and Stage 1 hard negatives.
    """
    logger.info("Generating candidate pairs with hard negatives for Ranker training...")
    all_features = []
    all_labels = []
    group_sizes = []

    # Map user interactions in validation set and training set
    # Target interactions come from validation split (clean leave-second-out)
    val_ratings = data.val_df.set_index(["u_idx", "i_idx"])["rating"].to_dict()
    train_ratings = data.train_df.set_index(["u_idx", "i_idx"])["rating"].to_dict()

    user_sample = sorted(list(data.val_df["u_idx"].unique()))[:num_users_to_sample]

    for u_idx in user_sample:
        u_meta = data.user_meta[u_idx]
        hist = data.user_history_i_idx.get(u_idx, [])[-config.data.last_n_watched :]
        pad_len = config.data.last_n_watched - len(hist)
        padded_hist = [data.num_items] * pad_len + hist

        with torch.no_grad():
            u_emb = model.user_tower(
                user_ids=torch.tensor([u_idx], dtype=torch.long),
                gender_idx=torch.tensor([1 if u_meta["gender"] == "F" else 0], dtype=torch.long),
                age_idx=torch.tensor([min(6, max(0, u_meta["age"] // 10))], dtype=torch.long),
                occ_idx=torch.tensor([min(21, max(0, u_meta["occupation"]))], dtype=torch.long),
                mean_rating=torch.tensor([u_meta["mean_rating"]], dtype=torch.float32),
                top_genres=torch.from_numpy(data.user_top_genres[u_idx: u_idx + 1]),
                history_items=torch.tensor([padded_hist], dtype=torch.long),
            ).cpu().numpy()[0]

        # Stage 1 retrieval: top 50 candidates (filter FAISS -1 sentinels)
        _, cand_indices = index.search(u_emb, k=50)
        cands = [int(c) for c in cand_indices if int(c) >= 0]

        # Include positive validation target if not present
        val_item_rows = data.val_df[data.val_df["u_idx"] == u_idx]
        for _, row in val_item_rows.iterrows():
            item = int(row["i_idx"])
            if item not in cands:
                cands.append(item)

        # Build labels and features
        user_cands = []
        user_y = []
        for item in cands:
            user_cands.append(item)
            # Label assignment: actual rating if known, else 0 (hard negative candidate)
            if (u_idx, item) in val_ratings:
                user_y.append(float(val_ratings[(u_idx, item)]))
            elif (u_idx, item) in train_ratings:
                user_y.append(float(train_ratings[(u_idx, item)]))
            else:
                user_y.append(0.0)  # Hard negative from retrieval!

        feats = feature_engineer.extract_features_for_user_candidates(u_idx, u_emb, user_cands)
        all_features.append(feats)
        all_labels.extend(user_y)
        group_sizes.append(len(user_cands))

    X = np.vstack(all_features)
    y = np.array(all_labels, dtype=np.float32)
    logger.info(f"Dataset generated: X shape {X.shape}, y shape {y.shape}, {len(group_sizes)} groups.")
    return X, y, group_sizes


def train_and_evaluate_ranker(
    config: PipelineConfig,
) -> Tuple[LambdaRanker, Dict[str, Any]]:
    """Train ranker, run full two-stage evaluation, ablations, and latency benchmarking."""
    np.random.seed(config.seed)
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

    # 3. Generate Training Set for Ranker
    X, y, groups = generate_ranking_dataset(
        data, model, index, feature_engineer, config, num_users_to_sample=4000
    )

    # 4. Train LambdaRanker
    ranker = LambdaRanker(
        n_estimators=config.ranker.n_estimators,
        learning_rate=config.ranker.learning_rate,
        num_leaves=config.ranker.num_leaves,
        min_child_samples=config.ranker.min_child_samples,
        seed=config.seed,
    ).fit(X, y, groups)

    ranker.save(str(artifacts_dir / "ranker.txt"))

    # 5. Evaluate Two-Stage Recommender on Test Set
    evaluator = Evaluator(
        test_df=data.test_df,
        total_items=data.num_items,
        item_popularity=data.item_popularity,
        genre_matrix=data.movie_genres_matrix,
        ks=config.eval.ks,
        ranking_k=config.eval.ranking_k,
    )

    # Recommender function: Two-Stage (Stage 1 Retrieve 200 -> Stage 2 Rank to top 10)
    retrieval_latencies = []
    ranking_latencies = []

    def two_stage_recommend(
        u_idx: int,
        k: int = 10,
        exclude: set = None,
        feature_subset: Optional[List[int]] = None,
        model_to_use: Optional[LambdaRanker] = None,
    ) -> List[int]:
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
                top_genres=torch.from_numpy(data.user_top_genres[u_idx: u_idx + 1]),
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
        return [cands[i] for i in ranked_indices]

    logger.info("Evaluating Two-Stage Recommendation System...")
    two_stage_metrics = evaluator.evaluate_model(
        model_name="Two-Stage (Two-Tower + Ranker)",
        rec_fn=two_stage_recommend,
        user_histories=data.user_history_i_idx,
    )

    # Calculate end-to-end latency statistics
    two_stage_metrics["retrieval_latency_p50_ms"] = float(round(np.percentile(retrieval_latencies, 50), 3))
    two_stage_metrics["retrieval_latency_p95_ms"] = float(round(np.percentile(retrieval_latencies, 95), 3))
    two_stage_metrics["ranking_latency_p50_ms"] = float(round(np.percentile(ranking_latencies, 50), 3))
    two_stage_metrics["ranking_latency_p95_ms"] = float(round(np.percentile(ranking_latencies, 95), 3))
    logger.info(f"Two-Stage System Results: {two_stage_metrics}")

    # 6. Ablation Studies
    logger.info("Running Ablation: Ranker without Two-Tower Similarity score...")
    # Feature 0 is two_tower_sim
    sub_indices_no_sim = [i for i, name in enumerate(FEATURE_NAMES) if name != "two_tower_sim"]
    X_no_sim = X[:, sub_indices_no_sim]
    names_no_sim = [FEATURE_NAMES[i] for i in sub_indices_no_sim]
    ranker_no_sim = LambdaRanker(seed=config.seed).fit(
        X_no_sim, y, groups, feature_names=names_no_sim
    )

    metrics_no_sim = evaluator.evaluate_model(
        model_name="Ablation: Ranker w/o Similarity",
        rec_fn=lambda u, k=10, exclude=None: two_stage_recommend(
            u, k, exclude, feature_subset=sub_indices_no_sim, model_to_use=ranker_no_sim
        ),
        user_histories=data.user_history_i_idx,
    )
    logger.info(f"Ablation w/o Sim: {metrics_no_sim}")

    logger.info("Running Ablation: Ranker without User features...")
    # User features: user_activity, user_mean_rating, genre_affinity
    user_feat_names = {"user_activity", "user_mean_rating", "genre_affinity"}
    sub_indices_no_user = [i for i, name in enumerate(FEATURE_NAMES) if name not in user_feat_names]
    X_no_user = X[:, sub_indices_no_user]
    names_no_user = [FEATURE_NAMES[i] for i in sub_indices_no_user]
    ranker_no_user = LambdaRanker(seed=config.seed).fit(
        X_no_user, y, groups, feature_names=names_no_user
    )

    metrics_no_user = evaluator.evaluate_model(
        model_name="Ablation: Ranker w/o User Features",
        rec_fn=lambda u, k=10, exclude=None: two_stage_recommend(
            u, k, exclude, feature_subset=sub_indices_no_user, model_to_use=ranker_no_user
        ),
        user_histories=data.user_history_i_idx,
    )
    logger.info(f"Ablation w/o User Features: {metrics_no_user}")

    # 7. Error Analysis (Cold-Start Users & Niche Genres)
    logger.info("Performing Error Analysis on Subpopulations...")
    user_train_counts = np.array([data.user_activity[u] for u in evaluator.eval_users])
    cold_threshold = np.percentile(user_train_counts, 20)  # Bottom 20% active users
    cold_users = [
        evaluator.eval_users[i]
        for i in range(len(evaluator.eval_users))
        if user_train_counts[i] <= cold_threshold
    ]

    cold_recs = [two_stage_recommend(u, k=10, exclude=set(data.user_history_i_idx.get(u, []))) for u in cold_users]
    cold_hits = [
        hit_at_k(rec, evaluator.test_ground_truth[u], k=10)
        for rec, u in zip(cold_recs, cold_users)
    ]
    cold_ndcgs = [
        ndcg_at_k(rec, evaluator.test_ground_truth[u], k=10)
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
    }
    logger.info(f"Error Analysis: {error_analysis}")

    # 8. Save updated metrics.json
    results_dir = Path(config.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"

    metrics_db = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_db = json.load(f)

    metrics_db["Two-Stage (Two-Tower + Ranker)"] = two_stage_metrics
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
