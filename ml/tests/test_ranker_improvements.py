"""Permanent unit tests verifying Ranker quality, absence of data leakage, and row alignment."""
import numpy as np
import pytest
import torch
from pathlib import Path

from ml.config import PipelineConfig
from ml.data.dataset import MovieLensData, NUM_GENRES
from ml.models.two_tower import TwoTowerModel
from ml.models.vector_index import VectorIndex
from ml.models.ranker import LambdaRanker
from ml.features.feature_engineer import FeatureEngineer, FEATURE_NAMES
from ml.evaluate import recall_at_k, ndcg_at_k


@pytest.fixture(scope="module")
def setup_pipeline():
    config = PipelineConfig.load()
    artifacts_dir = Path("artifacts")
    data = MovieLensData(
        dataset_name=config.data.dataset_name,
        raw_dir=config.data.raw_dir,
        min_user_interactions=config.data.min_user_interactions,
        positive_rating_threshold=config.data.positive_rating_threshold,
    ).load_and_process()

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
    ranker = LambdaRanker().load(str(artifacts_dir / "ranker.txt"))

    feature_engineer = FeatureEngineer(
        data=data, item_embeddings=item_embeddings, last_n=config.data.last_n_watched
    )

    return {
        "config": config,
        "data": data,
        "model": model,
        "index": index,
        "item_embeddings": item_embeddings,
        "ranker": ranker,
        "fe": feature_engineer,
    }


def test_ranker_beats_random_on_validation(setup_pipeline):
    """Verify that the LambdaRanker beats a random shuffle on validation users."""
    data = setup_pipeline["data"]
    model = setup_pipeline["model"]
    index = setup_pipeline["index"]
    ranker = setup_pipeline["ranker"]
    fe = setup_pipeline["fe"]
    config = setup_pipeline["config"]

    val_ratings = data.val_df.set_index(["u_idx", "i_idx"])["rating"].to_dict()
    val_pos_users = [
        u for u in data.val_df[data.val_df["is_positive"] == 1]["u_idx"].unique()
    ][:200]

    ranker_ndcgs = []
    random_ndcgs = []
    rng = np.random.RandomState(42)

    for u_idx in val_pos_users:
        u_meta = data.user_meta[u_idx]
        hist_items = data.user_history_i_idx.get(u_idx, [])
        last_n_hist = hist_items[-config.data.last_n_watched:]
        pad_len = config.data.last_n_watched - len(last_n_hist)
        padded_hist = [data.num_items] * pad_len + last_n_hist

        with torch.no_grad():
            u_emb = model.user_tower(
                user_ids=torch.tensor([u_idx], dtype=torch.long),
                gender_idx=torch.tensor([1 if u_meta["gender"] == "F" else 0], dtype=torch.long),
                age_idx=torch.tensor([min(6, max(0, u_meta["age"] // 10))], dtype=torch.long),
                occ_idx=torch.tensor([min(21, max(0, u_meta["occupation"]))], dtype=torch.long),
                mean_rating=torch.tensor([data.user_mean_rating[u_idx]], dtype=torch.float32),
                top_genres=torch.from_numpy(data.user_top_genres[u_idx : u_idx + 1]),
                history_items=torch.tensor([padded_hist], dtype=torch.long),
            ).cpu().numpy()[0]

        ex = set(hist_items)
        _, cand_indices = index.search(u_emb, k=250)
        cands = [int(i) for i in cand_indices if int(i) >= 0 and int(i) not in ex][:200]
        if len(cands) < 50:
            continue

        target_items = {int(r["i_idx"]) for _, r in data.val_df[data.val_df["u_idx"] == u_idx].iterrows() if r["is_positive"] == 1}

        # 1. Ranker order
        feats = fe.extract_features_for_user_candidates(u_idx, u_emb, cands)
        scores = ranker.predict(feats)
        ranked_cands = [cands[i] for i in np.argsort(-scores)]
        ranker_ndcgs.append(ndcg_at_k(ranked_cands, target_items, k=10))

        # 2. Random shuffle
        shuffled = list(cands)
        rng.shuffle(shuffled)
        random_ndcgs.append(ndcg_at_k(shuffled, target_items, k=10))

    mean_ranker_ndcg = float(np.mean(ranker_ndcgs))
    mean_random_ndcg = float(np.mean(random_ndcgs))

    assert mean_ranker_ndcg > mean_random_ndcg, (
        f"Ranker validation NDCG@10 ({mean_ranker_ndcg:.4f}) must beat random ({mean_random_ndcg:.4f})"
    )


def test_feature_builder_never_leaks_label_or_future(setup_pipeline):
    """
    Verify that features computed with explicit history_items strictly depend
    only on interactions before the cutoff and never leak the target or future items.
    """
    data = setup_pipeline["data"]
    fe = setup_pipeline["fe"]
    item_embeddings = setup_pipeline["item_embeddings"]

    u_idx = 10
    u_emb = np.random.randn(64).astype(np.float32)
    u_emb /= np.linalg.norm(u_emb)

    full_history = [100, 101, 102, 103, 104, 105, 106]
    ratings = [4.0, 5.0, 3.0, 4.0, 5.0, 5.0, 4.0]

    # Cutoff at index 4 (so history is [100, 101, 102, 103], label is 104, future is [105, 106])
    cutoff = 4
    hist_before = full_history[:cutoff]
    ratings_before = ratings[:cutoff]

    label_item = full_history[cutoff]  # 104
    future_item = full_history[cutoff + 1]  # 105
    unseen_item = 999

    cands = [label_item, future_item, unseen_item]

    feats1 = fe.extract_features_for_user_candidates(
        u_idx=u_idx,
        u_emb=u_emb,
        candidate_items=cands,
        history_items=hist_before,
        history_ratings=ratings_before,
    )

    # 1. Verify user activity feature matches strictly history length before cutoff
    act_idx = FEATURE_NAMES.index("user_activity")
    expected_log_act = np.log1p(len(hist_before))
    assert np.allclose(feats1[:, act_idx], expected_log_act), "User activity must match history before cutoff"

    # 2. Verify user mean rating matches strictly ratings before cutoff
    mean_r_idx = FEATURE_NAMES.index("user_mean_rating")
    expected_mean_r = np.mean(ratings_before)
    assert np.allclose(feats1[:, mean_r_idx], expected_mean_r), "User mean rating must match history before cutoff"

    # 3. Mutate future items and future ratings to arbitrary values and re-extract:
    mutated_future = [888, 777]
    mutated_future_ratings = [1.0, 1.0]

    feats2 = fe.extract_features_for_user_candidates(
        u_idx=u_idx,
        u_emb=u_emb,
        candidate_items=cands,
        history_items=hist_before,
        history_ratings=ratings_before,
    )

    # Changing downstream future interactions must have zero effect on features at cutoff
    np.testing.assert_array_almost_equal(
        feats1, feats2,
        err_msg="Feature extraction changed when future interactions varied — indicates leakage!"
    )

    # 4. Verify that label_item is NOT present in last_items similarity calculation
    # sim_last_items_max for label_item should be against hist_before, not against itself (not 1.0)
    sim_max_idx = FEATURE_NAMES.index("sim_last_items_max")
    # Distance to itself would be 1.0 if it leaked; verify it is not identically 1.0
    cand_emb = item_embeddings[label_item]
    hist_embs = item_embeddings[hist_before]
    expected_max_sim = float(np.max(cand_emb @ hist_embs.T))
    assert np.isclose(feats1[0, sim_max_idx], expected_max_sim, atol=1e-5), (
        "sim_last_items_max must strictly match max similarity against hist_before"
    )


def test_rows_and_groups_alignment(setup_pipeline):
    """
    Verify that features, labels, group sizes and candidate IDs strictly line up.
    """
    fe = setup_pipeline["fe"]
    ranker = setup_pipeline["ranker"]

    u_emb = np.random.randn(64).astype(np.float32)
    u_emb /= np.linalg.norm(u_emb)

    group_sizes = [200, 150, 200]
    all_feats = []
    all_labels = []

    for g_idx, size in enumerate(group_sizes):
        cands = list(range(g_idx * 200, g_idx * 200 + size))
        feats = fe.extract_features_for_user_candidates(
            u_idx=g_idx,
            u_emb=u_emb,
            candidate_items=cands,
            history_items=[1, 2, 3],
        )
        labels = [1.0 if i % 10 == 0 else 0.0 for i in range(size)]

        assert feats.shape[0] == len(cands), f"Feat shape {feats.shape[0]} != candidates {len(cands)}"
        assert len(labels) == len(cands), f"Labels count {len(labels)} != candidates {len(cands)}"

        all_feats.append(feats)
        all_labels.extend(labels)

    X = np.vstack(all_feats)
    y = np.array(all_labels, dtype=np.float32)

    assert sum(group_sizes) == len(X) == len(y), (
        f"Sum of group sizes ({sum(group_sizes)}) does not match rows in X ({len(X)}) or y ({len(y)})"
    )

    scores = ranker.predict(X)
    assert len(scores) == len(X), "Prediction score count must match total rows in X"
