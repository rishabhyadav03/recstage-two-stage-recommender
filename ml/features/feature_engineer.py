"""Feature Engineering for Stage 2 Ranker."""
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import logging
from ml.data.dataset import MovieLensData

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "two_tower_sim",
    "genre_affinity",
    "item_popularity",
    "item_mean_rating",
    "recency",
    "user_activity",
    "user_mean_rating",
    "sim_last_items_max",
    "sim_last_items_mean",
    "stage1_rank_inv",
]


class FeatureEngineer:
    """Extracts ranking features for (user, item) pairs."""

    def __init__(
        self,
        data: MovieLensData,
        item_embeddings: np.ndarray,
        last_n: int = 5,
    ):
        self.data = data
        self.item_embeddings = item_embeddings  # (num_items, 64)
        self.last_n = last_n

    def extract_features_for_user_candidates(
        self,
        u_idx: int,
        u_emb: np.ndarray,
        candidate_items: List[int],
        history_items: Optional[List[int]] = None,
        history_ratings: Optional[List[float]] = None,
        user_genres: Optional[np.ndarray] = None,
        user_mean_rating: Optional[float] = None,
    ) -> np.ndarray:
        """
        Extract feature matrix for a user and a list of candidate items.
        Returns array of shape (len(candidate_items), num_features).

        When history_items is provided (e.g., during training with cutoff time t),
        features are strictly computed from interactions prior to cutoff t to prevent leakage.
        """
        # Filter FAISS sentinel -1 indices before any lookup
        candidate_items = [int(i) for i in candidate_items if int(i) >= 0]
        num_cands = len(candidate_items)
        if num_cands == 0:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

        cand_array = np.array(candidate_items, dtype=np.int64)
        cand_embeds = self.item_embeddings[cand_array]  # (num_cands, 64)

        # 1. Two-tower similarity (dot product of L2 normalized vectors)
        two_tower_sim = (cand_embeds @ u_emb).astype(np.float32)

        # 2. Genre affinity
        if user_genres is not None:
            u_genres = user_genres
        elif history_items is not None:
            # Compute strictly from positive items in history_items
            pos_items = [
                h for idx, h in enumerate(history_items)
                if (history_ratings is None or history_ratings[idx] >= self.data.positive_rating_threshold)
            ]
            if pos_items:
                g_sum = self.data.movie_genres_matrix[pos_items].sum(axis=0)
                norm = np.linalg.norm(g_sum)
                u_genres = g_sum / norm if norm > 0 else np.zeros(self.data.movie_genres_matrix.shape[1], dtype=np.float32)
            else:
                u_genres = np.zeros(self.data.movie_genres_matrix.shape[1], dtype=np.float32)
        else:
            u_genres = self.data.user_top_genres[u_idx]  # (num_genres,)

        cand_genres = self.data.movie_genres_matrix[cand_array]  # (num_cands, num_genres)
        genre_affinity = (cand_genres @ u_genres).astype(np.float32)

        # 3. Item popularity & 4. Item mean rating & 5. Recency
        # These are computed strictly from the training split in dataset.py
        item_pop = np.log1p(self.data.item_popularity[cand_array]).astype(np.float32)
        item_mean_r = self.data.item_mean_rating[cand_array].astype(np.float32)
        recency = np.array(
            [self.data.item_meta[i]["year_norm"] for i in cand_array], dtype=np.float32
        )

        # 6. User activity & 7. User mean rating
        if history_items is not None:
            act_count = len(history_items)
        else:
            act_count = self.data.user_activity[u_idx]
        u_activity = np.full(num_cands, np.log1p(act_count), dtype=np.float32)

        if user_mean_rating is not None:
            mean_r = user_mean_rating
        elif history_ratings is not None and len(history_ratings) > 0:
            mean_r = float(np.mean(history_ratings))
        else:
            mean_r = float(self.data.user_mean_rating[u_idx])
        u_mean_r = np.full(num_cands, mean_r, dtype=np.float32)

        # 8 & 9: Similarity to user's last N watched items
        if history_items is not None:
            u_history = history_items
        else:
            u_history = self.data.user_history_i_idx.get(u_idx, [])
        last_items = u_history[-self.last_n :] if u_history else []

        if last_items:
            last_embeds = self.item_embeddings[last_items]  # (len_hist, 64)
            hist_sims = cand_embeds @ last_embeds.T  # (num_cands, len_hist)
            sim_last_max = np.max(hist_sims, axis=1).astype(np.float32)
            sim_last_mean = np.mean(hist_sims, axis=1).astype(np.float32)
        else:
            sim_last_max = np.zeros(num_cands, dtype=np.float32)
            sim_last_mean = np.zeros(num_cands, dtype=np.float32)

        # 10. Stage-1 rank reciprocal
        ranks = np.arange(1, num_cands + 1, dtype=np.float32)
        stage1_rank_inv = 1.0 / np.sqrt(ranks)

        features = np.column_stack([
            two_tower_sim,
            genre_affinity,
            item_pop,
            item_mean_r,
            recency,
            u_activity,
            u_mean_r,
            sim_last_max,
            sim_last_mean,
            stage1_rank_inv,
        ])
        assert features.shape[0] == num_cands, f"Feature row count {features.shape[0]} != candidate count {num_cands}"
        return features.astype(np.float32)
