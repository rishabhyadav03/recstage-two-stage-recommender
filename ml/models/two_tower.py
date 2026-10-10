"""Two-Tower DNN Retrieval Model with In-Batch Sampled Softmax and LogQ Correction."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class UserTower(nn.Module):
    """User tower neural network producing L2-normalized 64-dim embeddings."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_genres: int = 19,
        id_dim: int = 32,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.user_emb = nn.Embedding(num_users + 1, id_dim, padding_idx=num_users)
        self.gender_emb = nn.Embedding(3, 4)
        self.age_emb = nn.Embedding(8, 8)
        self.occ_emb = nn.Embedding(22, 8)
        self.item_hist_emb = nn.Embedding(num_items + 1, id_dim, padding_idx=num_items)
        self.genre_proj = nn.Linear(num_genres, 16)

        # Total input dim: id_dim(32) + 4 + 8 + 8 + 1(mean_rating) + 16(genre) + id_dim(32) = 101
        in_dim = id_dim + 4 + 8 + 8 + 1 + 16 + id_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        user_ids: torch.Tensor,
        gender_idx: torch.Tensor,
        age_idx: torch.Tensor,
        occ_idx: torch.Tensor,
        mean_rating: torch.Tensor,
        top_genres: torch.Tensor,
        history_items: torch.Tensor,  # (batch_size, last_n)
    ) -> torch.Tensor:
        u_e = self.user_emb(user_ids)
        g_e = self.gender_emb(gender_idx)
        a_e = self.age_emb(age_idx)
        o_e = self.occ_emb(occ_idx)
        r_f = mean_rating.unsqueeze(-1) / 5.0
        genre_f = F.relu(self.genre_proj(top_genres))

        # Average embedding of last N items
        hist_e = self.item_hist_emb(history_items).mean(dim=1)

        x = torch.cat([u_e, g_e, a_e, o_e, r_f, genre_f, hist_e], dim=-1)
        out = self.mlp(x)
        return F.normalize(out, p=2, dim=-1)


class ItemTower(nn.Module):
    """Item tower neural network producing L2-normalized 64-dim embeddings."""

    def __init__(
        self,
        num_items: int,
        num_genres: int = 19,
        id_dim: int = 32,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.item_emb = nn.Embedding(num_items + 1, id_dim, padding_idx=num_items)
        self.genre_proj = nn.Linear(num_genres, 16)
        self.pop_emb = nn.Embedding(11, 8)

        # Total input dim: id_dim(32) + 16 + 1(year_norm) + 8 = 57
        in_dim = id_dim + 16 + 1 + 8

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        item_ids: torch.Tensor,
        genres: torch.Tensor,
        year_norm: torch.Tensor,
        pop_bucket: torch.Tensor,
    ) -> torch.Tensor:
        i_e = self.item_emb(item_ids)
        g_f = F.relu(self.genre_proj(genres))
        y_f = year_norm.unsqueeze(-1)
        p_e = self.pop_emb(pop_bucket)

        x = torch.cat([i_e, g_f, y_f, p_e], dim=-1)
        out = self.mlp(x)
        return F.normalize(out, p=2, dim=-1)


class TwoTowerModel(nn.Module):
    """
    Two-Tower recommender with sampled softmax and logQ popularity correction.
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_genres: int = 19,
        embedding_dim: int = 64,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.user_tower = UserTower(
            num_users=num_users,
            num_items=num_items,
            num_genres=num_genres,
            embedding_dim=embedding_dim,
        )
        self.item_tower = ItemTower(
            num_items=num_items,
            num_genres=num_genres,
            embedding_dim=embedding_dim,
        )
        self.temperature = temperature

    def forward(
        self,
        user_inputs: Dict[str, torch.Tensor],
        item_inputs: Dict[str, torch.Tensor],
        item_logq: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute in-batch cosine similarity logits with logQ correction.
        user_inputs: dict of tensors for user tower
        item_inputs: dict of tensors for item tower
        item_logq: log(P(i)) of items in the batch
        """
        u_emb = self.user_tower(**user_inputs)  # (B, D)
        i_emb = self.item_tower(**item_inputs)  # (B, D)

        # Dot product of normalized vectors = Cosine similarity
        logits = torch.matmul(u_emb, i_emb.T) / self.temperature  # (B, B)

        if item_logq is not None:
            # Broadcast logQ correction across batch rows: s(u, i) - log(P(i))
            logits = logits - item_logq.unsqueeze(0)

        targets = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(logits, targets)

        return loss, u_emb, i_emb
