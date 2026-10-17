"""
Parity test comparing PyTorch Two-Tower inference and precomputed NumPy serving artifacts.

Verifies that for at least 200 users (including cold-start and active cohorts):
1. User embeddings match within 1e-5 absolute difference.
2. Top-10 retrieved candidate IDs (/api/recommend mode=retrieval_only) match 100% identically.
3. Top-10 two-stage ranked IDs (/api/recommend mode=two_stage) match 100% identically.

Automatically skipped if PyTorch is not installed in the environment.
"""
from pathlib import Path
import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from ml.config import PipelineConfig
from ml.data.dataset import MovieLensData, NUM_GENRES
if TORCH_AVAILABLE:
    from ml.models.two_tower import TwoTowerModel
else:
    TwoTowerModel = None
from ml.models.vector_index import VectorIndex
from ml.models.ranker import LambdaRanker
from ml.features.feature_engineer import FeatureEngineer

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


@pytest.fixture(scope="module")
def parity_fixtures():
    cfg = PipelineConfig.load()
    data = MovieLensData.load_cache(ARTIFACTS_DIR / "dataset_cache.pkl")

    model = TwoTowerModel(
        num_users=data.num_users,
        num_items=data.num_items,
        num_genres=NUM_GENRES,
        embedding_dim=cfg.retrieval.embedding_dim,
    )
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "two_tower.pt", weights_only=True))
    model.eval()

    numpy_user_embs = np.load(str(ARTIFACTS_DIR / "user_embeddings.npy"))
    index = VectorIndex(dim=cfg.retrieval.embedding_dim).load(str(ARTIFACTS_DIR / "faiss_index.bin"))
    item_embeddings = np.load(str(ARTIFACTS_DIR / "item_embeddings.npy"))
    ranker = LambdaRanker().load(str(ARTIFACTS_DIR / "ranker.txt"))
    feature_eng = FeatureEngineer(data=data, item_embeddings=item_embeddings, last_n=cfg.data.last_n_watched)

    return {
        "cfg": cfg,
        "data": data,
        "model": model,
        "numpy_user_embs": numpy_user_embs,
        "index": index,
        "ranker": ranker,
        "feature_eng": feature_eng,
    }


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed; skipping torch-vs-numpy parity test")
def test_user_embeddings_and_recommendations_parity(parity_fixtures):
    cfg = parity_fixtures["cfg"]
    data = parity_fixtures["data"]
    model = parity_fixtures["model"]
    numpy_user_embs = parity_fixtures["numpy_user_embs"]
    index = parity_fixtures["index"]
    ranker = parity_fixtures["ranker"]
    feature_eng = parity_fixtures["feature_eng"]

    # Identify cold-start users (bottom 20% active users, <= 34 interactions) and warm users
    user_counts = [data.user_activity[u] for u in range(data.num_users)]
    cold_thresh = np.percentile(user_counts, 20)
    cold_users = [u for u in range(data.num_users) if user_counts[u] <= cold_thresh]
    warm_users = [u for u in range(data.num_users) if user_counts[u] > cold_thresh]

    # Sample 120 cold and 120 warm users (total 240 users >= 200 required)
    test_users = cold_users[:120] + warm_users[:120]
    assert len(test_users) >= 200, f"Expected at least 200 test users, got {len(test_users)}"

    max_emb_diff = 0.0
    retrieval_mismatches = []
    two_stage_mismatches = []

    for u_idx in test_users:
        u_meta = data.user_meta[u_idx]
        hist = data.user_history_i_idx.get(u_idx, [])[-cfg.data.last_n_watched :]
        pad_len = cfg.data.last_n_watched - len(hist)
        padded_hist = [data.num_items] * pad_len + hist

        with torch.no_grad():
            torch_emb = model.user_tower(
                user_ids=torch.tensor([u_idx], dtype=torch.long),
                gender_idx=torch.tensor([1 if u_meta.get("gender") == "F" else 0], dtype=torch.long),
                age_idx=torch.tensor([min(6, max(0, int(u_meta.get("age", 25)) // 10))], dtype=torch.long),
                occ_idx=torch.tensor([min(21, max(0, int(u_meta.get("occupation", 0))))], dtype=torch.long),
                mean_rating=torch.tensor([float(u_meta.get("mean_rating", 3.0))], dtype=torch.float32),
                top_genres=torch.from_numpy(data.user_top_genres[u_idx : u_idx + 1]),
                history_items=torch.tensor([padded_hist], dtype=torch.long),
            ).cpu().numpy()[0]

        np_emb = numpy_user_embs[u_idx]
        diff = float(np.max(np.abs(torch_emb - np_emb)))
        if diff > max_emb_diff:
            max_emb_diff = diff

        # Check embedding tolerance
        assert diff < 1e-5, f"User {u_idx} embedding diff {diff:.2e} exceeded 1e-5 threshold"

        exclude = set(data.user_history_i_idx.get(u_idx, []))
        query_k = min(data.num_items, cfg.retrieval.top_k_candidates + len(exclude) + 10)

        # 1. Retrieval with PyTorch embedding
        _, t_cand_indices = index.search(torch_emb, k=query_k)
        t_cands = [int(i) for i in t_cand_indices if i not in exclude][: cfg.retrieval.top_k_candidates]
        t_top10_ret = t_cands[:10]

        # 2. Retrieval with NumPy precomputed embedding
        _, n_cand_indices = index.search(np_emb, k=query_k)
        n_cands = [int(i) for i in n_cand_indices if i not in exclude][: cfg.retrieval.top_k_candidates]
        n_top10_ret = n_cands[:10]

        if t_top10_ret != n_top10_ret:
            retrieval_mismatches.append((u_idx, t_top10_ret, n_top10_ret))

        # 3. Two-Stage with PyTorch embedding
        t_feats = feature_eng.extract_features_for_user_candidates(u_idx, torch_emb, t_cands)
        t_scores = ranker.predict(t_feats)
        t_top10_ts = [t_cands[i] for i in np.argsort(-t_scores)[:10]]

        # 4. Two-Stage with NumPy precomputed embedding
        n_feats = feature_eng.extract_features_for_user_candidates(u_idx, np_emb, n_cands)
        n_scores = ranker.predict(n_feats)
        n_top10_ts = [n_cands[i] for i in np.argsort(-n_scores)[:10]]

        if t_top10_ts != n_top10_ts:
            two_stage_mismatches.append((u_idx, t_top10_ts, n_top10_ts))

    assert max_emb_diff < 1e-5, f"Maximum embedding difference {max_emb_diff:.2e} >= 1e-5"
    assert len(retrieval_mismatches) == 0, f"Retrieval top-10 mismatches on {len(retrieval_mismatches)} users"
    assert len(two_stage_mismatches) == 0, f"Two-stage top-10 mismatches on {len(two_stage_mismatches)} users"
