"""Baseline recommendation models: Popularity and Matrix Factorization (BPR-MF)."""
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple, Set
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PopularityModel:
    """Popularity baseline recommender."""

    def __init__(self, item_popularity: np.ndarray):
        """item_popularity: array of shape (num_items,) with interaction counts."""
        self.item_popularity = item_popularity
        self.ranked_items = np.argsort(-item_popularity)

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_items: Optional[Set[int]] = None,
    ) -> List[int]:
        """Return top-k popular items, excluding already consumed items if provided."""
        exclude = exclude_items or set()
        recommendations = []
        for item in self.ranked_items:
            if item not in exclude:
                recommendations.append(int(item))
            if len(recommendations) >= k:
                break
        return recommendations

    def batch_recommend(
        self,
        user_indices: List[int],
        user_histories: Dict[int, List[int]],
        k: int = 10,
    ) -> Dict[int, List[int]]:
        """Batch recommendation for evaluation."""
        results = {}
        for u in user_indices:
            exclude = set(user_histories.get(u, []))
            results[u] = self.recommend(u, k=k, exclude_items=exclude)
        return results


class BPRMatrixFactorization(nn.Module):
    """Bayesian Personalized Ranking Matrix Factorization implemented in PyTorch."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        self.item_bias = nn.Embedding(num_items, 1)

        # Initialization
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        u_emb = self.user_embeddings(users)
        pos_emb = self.item_embeddings(pos_items)
        neg_emb = self.item_embeddings(neg_items)

        pos_bias = self.item_bias(pos_items).squeeze(-1)
        neg_bias = self.item_bias(neg_items).squeeze(-1)

        pos_scores = (u_emb * pos_emb).sum(dim=-1) + pos_bias
        neg_scores = (u_emb * neg_emb).sum(dim=-1) + neg_bias

        return pos_scores, neg_scores

    def get_user_embedding(self, user_idx: int) -> np.ndarray:
        with torch.no_grad():
            u = torch.tensor([user_idx], dtype=torch.long)
            return self.user_embeddings(u).cpu().numpy()[0]

    def get_all_item_embeddings(self) -> np.ndarray:
        with torch.no_grad():
            return self.item_embeddings.weight.cpu().numpy()

    def recommend(
        self,
        user_idx: int,
        k: int = 10,
        exclude_items: Optional[Set[int]] = None,
    ) -> List[int]:
        with torch.no_grad():
            u = torch.tensor([user_idx], dtype=torch.long)
            u_emb = self.user_embeddings(u)  # (1, d)
            scores = (u_emb @ self.item_embeddings.weight.T).squeeze(0) + self.item_bias.weight.squeeze(-1)
            scores = scores.cpu().numpy()

        if exclude_items:
            for item in exclude_items:
                if item < len(scores):
                    scores[item] = -1e9

        top_indices = np.argpartition(-scores, k)[:k]
        top_sorted = top_indices[np.argsort(-scores[top_indices])]
        return [int(x) for x in top_sorted]


def train_bpr_model(
    num_users: int,
    num_items: int,
    train_user_pos_items: Dict[int, List[int]],
    embedding_dim: int = 64,
    epochs: int = 10,
    lr: float = 0.01,
    weight_decay: float = 1e-5,
    batch_size: int = 1024,
    seed: int = 42,
) -> BPRMatrixFactorization:
    """Train BPR Matrix Factorization with negative sampling."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BPRMatrixFactorization(num_users, num_items, embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Flatten positive interactions for training
    user_item_pairs = []
    for u, items in train_user_pos_items.items():
        for i in items:
            user_item_pairs.append((u, i))

    user_item_pairs = np.array(user_item_pairs, dtype=np.int64)
    num_samples = len(user_item_pairs)
    logger.info(f"Training BPR-MF on {num_samples} positive pairs for {epochs} epochs...")

    model.train()
    for epoch in range(epochs):
        np.random.shuffle(user_item_pairs)
        epoch_loss = 0.0
        num_batches = int(np.ceil(num_samples / batch_size))

        for b in range(num_batches):
            batch_pairs = user_item_pairs[b * batch_size : (b + 1) * batch_size]
            b_users = torch.from_numpy(batch_pairs[:, 0])
            b_pos_items = torch.from_numpy(batch_pairs[:, 1])

            # Sample random negatives
            b_neg_items = torch.from_numpy(np.random.randint(0, num_items, size=len(batch_pairs)))

            optimizer.zero_grad()
            pos_scores, neg_scores = model(b_users, b_pos_items, b_neg_items)
            # BPR loss: -ln(sigmoid(pos - neg))
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, num_batches)
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            logger.info(f"BPR-MF Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    model.eval()
    return model
