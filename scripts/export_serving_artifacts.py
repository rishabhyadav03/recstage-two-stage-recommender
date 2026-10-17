"""
Export Torch-free Serving Artifacts for RecStage API.

Precomputes 64-dimensional user embeddings for all MovieLens users
using the trained Two-Tower model, saving them as artifacts/user_embeddings.npy.
This allows the serving backend to perform fast embedding lookups without
importing PyTorch or loading two_tower.pt at runtime.
"""
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import torch

from ml.config import PipelineConfig
from ml.data.dataset import MovieLensData, NUM_GENRES
from ml.models.two_tower import TwoTowerModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATASET_CACHE_FILE = ARTIFACTS_DIR / "dataset_cache.pkl"
TWO_TOWER_FILE = ARTIFACTS_DIR / "two_tower.pt"
OUTPUT_USER_EMB_FILE = ARTIFACTS_DIR / "user_embeddings.npy"


def export_user_embeddings():
    logger.info("Loading pipeline config...")
    cfg = PipelineConfig.load()

    logger.info(f"Loading dataset cache from {DATASET_CACHE_FILE}...")
    if not DATASET_CACHE_FILE.exists():
        raise FileNotFoundError(f"{DATASET_CACHE_FILE} does not exist.")
    data = MovieLensData.load_cache(DATASET_CACHE_FILE)

    logger.info(f"Loading Two-Tower model from {TWO_TOWER_FILE}...")
    if not TWO_TOWER_FILE.exists():
        raise FileNotFoundError(f"{TWO_TOWER_FILE} does not exist.")

    model = TwoTowerModel(
        num_users=data.num_users,
        num_items=data.num_items,
        num_genres=NUM_GENRES,
        embedding_dim=cfg.retrieval.embedding_dim,
    )
    model.load_state_dict(torch.load(TWO_TOWER_FILE, weights_only=True))
    model.eval()

    num_users = data.num_users
    logger.info(f"Computing user embeddings for all {num_users} users...")

    user_embeddings = np.zeros((num_users, cfg.retrieval.embedding_dim), dtype=np.float32)

    # Process in batches for speed and low memory
    batch_size = 512
    for start_idx in range(0, num_users, batch_size):
        end_idx = min(start_idx + batch_size, num_users)
        b_users = list(range(start_idx, end_idx))
        b_size = len(b_users)

        gender_list = []
        age_list = []
        occ_list = []
        mean_r_list = []
        padded_hist_list = []

        for u in b_users:
            u_meta = data.user_meta[u]
            gender_list.append(1 if u_meta.get("gender") == "F" else 0)
            age_list.append(min(6, max(0, int(u_meta.get("age", 25)) // 10)))
            occ_list.append(min(21, max(0, int(u_meta.get("occupation", 0)))))
            mean_r_list.append(float(u_meta.get("mean_rating", 3.0)))

            hist = data.user_history_i_idx.get(u, [])[-cfg.data.last_n_watched :]
            pad_len = cfg.data.last_n_watched - len(hist)
            padded_hist = [data.num_items] * pad_len + hist
            padded_hist_list.append(padded_hist)

        with torch.no_grad():
            b_emb = model.user_tower(
                user_ids=torch.tensor(b_users, dtype=torch.long),
                gender_idx=torch.tensor(gender_list, dtype=torch.long),
                age_idx=torch.tensor(age_list, dtype=torch.long),
                occ_idx=torch.tensor(occ_list, dtype=torch.long),
                mean_rating=torch.tensor(mean_r_list, dtype=torch.float32),
                top_genres=torch.from_numpy(data.user_top_genres[start_idx:end_idx]),
                history_items=torch.tensor(padded_hist_list, dtype=torch.long),
            ).cpu().numpy()

        user_embeddings[start_idx:end_idx] = b_emb

    # Sanity checks
    assert not np.isnan(user_embeddings).any(), "NaN values found in user embeddings"
    norms = np.linalg.norm(user_embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "User embeddings are not properly L2-normalized"

    np.save(str(OUTPUT_USER_EMB_FILE), user_embeddings)
    file_size_mb = OUTPUT_USER_EMB_FILE.stat().st_size / (1024 * 1024)
    logger.info(
        f"Successfully exported {user_embeddings.shape} user embeddings to "
        f"{OUTPUT_USER_EMB_FILE} ({file_size_mb:.2f} MB)"
    )


if __name__ == "__main__":
    export_user_embeddings()
