"""Comprehensive Evaluation Harness for Recommendation Systems."""
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def hit_at_k(recommended: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """Hit Rate @ K."""
    top_k = set(recommended[:k])
    return 1.0 if len(top_k.intersection(ground_truth)) > 0 else 0.0


def recall_at_k(recommended: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """Recall @ K."""
    if not ground_truth:
        return 0.0
    top_k = set(recommended[:k])
    hits = len(top_k.intersection(ground_truth))
    return float(hits) / float(len(ground_truth))


def ndcg_at_k(recommended: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain @ K (Binary relevance)."""
    if not ground_truth:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(recommended[:k]):
        if item in ground_truth:
            dcg += 1.0 / np.log2(rank + 2)  # rank + 1 + 1 (1-based index)

    # Ideal DCG: best possible order for ground truth items
    idcg = 0.0
    for rank in range(min(len(ground_truth), k)):
        idcg += 1.0 / np.log2(rank + 2)

    return float(dcg / idcg) if idcg > 0 else 0.0


def map_at_k(recommended: List[int], ground_truth: Set[int], k: int = 10) -> float:
    """Mean Average Precision @ K."""
    if not ground_truth:
        return 0.0
    hits = 0
    score = 0.0
    for rank, item in enumerate(recommended[:k]):
        if item in ground_truth:
            hits += 1
            score += float(hits) / float(rank + 1)
    return float(score / min(len(ground_truth), k)) if len(ground_truth) > 0 else 0.0


def catalog_coverage(all_recommendations: List[List[int]], total_catalog_items: int) -> float:
    """Fraction of catalog items recommended at least once."""
    if total_catalog_items == 0:
        return 0.0
    unique_rec = set()
    for rec_list in all_recommendations:
        unique_rec.update(rec_list)
    return float(len(unique_rec)) / float(total_catalog_items)


def intra_list_diversity(
    recommended: List[int],
    genre_matrix: np.ndarray,
) -> float:
    """Average pairwise cosine distance (1 - cosine_similarity) between recommended items."""
    k = len(recommended)
    if k <= 1:
        return 0.0
    vectors = genre_matrix[recommended]  # (k, num_genres)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    normed_vecs = vectors / norms
    sim_matrix = normed_vecs @ normed_vecs.T  # (k, k)
    
    # Extract upper triangle without diagonal
    triu_indices = np.triu_indices(k, k=1)
    if len(triu_indices[0]) == 0:
        return 0.0
    distances = 1.0 - sim_matrix[triu_indices]
    return float(np.mean(distances))


def novelty_score(
    recommended: List[int],
    item_popularity: np.ndarray,
    total_interactions: int,
) -> float:
    """Average self-information: -log2(P(item)) where P(item) = pop / total_interactions."""
    if not recommended or total_interactions <= 0:
        return 0.0
    pop = item_popularity[recommended]
    probs = np.maximum(pop, 1.0) / float(total_interactions)
    info = -np.log2(probs)
    return float(np.mean(info))


def paired_bootstrap_difference(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    num_bootstrap: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, Any]:
    """
    Compute paired bootstrap confidence interval for mean difference (scores_a - scores_b).
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    assert len(scores_a) == len(scores_b), "Input score arrays must have the same length"
    n = len(scores_a)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "includes_zero": True}

    diff = scores_a - scores_b
    mean_diff = float(np.mean(diff))

    rng = np.random.RandomState(seed)
    boot_means = np.empty(num_bootstrap, dtype=np.float64)
    for i in range(num_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(diff[idx])

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    includes_zero = bool(ci_lower <= 0.0 <= ci_upper)

    return {
        "mean_diff": float(round(mean_diff, 6)),
        "ci_lower": float(round(ci_lower, 6)),
        "ci_upper": float(round(ci_upper, 6)),
        "includes_zero": includes_zero,
    }


class Evaluator:
    """Evaluates recommendation models on a test set."""

    def __init__(
        self,
        test_df: pd.DataFrame,
        total_items: int,
        item_popularity: np.ndarray,
        genre_matrix: np.ndarray,
        ks: List[int] = [10, 50, 100],
        ranking_k: int = 10,
    ):
        self.test_df = test_df
        self.total_items = total_items
        self.item_popularity = item_popularity
        self.genre_matrix = genre_matrix
        self.ks = ks
        self.ranking_k = ranking_k
        self.total_interactions = int(np.sum(item_popularity))

        # Ground truth mapping: user_idx -> set of positive test item indices
        self.test_ground_truth: Dict[int, Set[int]] = {}
        # Also store all test items (including rating < 4) for error analysis
        self.test_all_items: Dict[int, Set[int]] = {}

        pos_test = test_df[test_df["is_positive"] == 1]
        for u, group in pos_test.groupby("u_idx"):
            self.test_ground_truth[int(u)] = set(group["i_idx"].tolist())

        for u, group in test_df.groupby("u_idx"):
            self.test_all_items[int(u)] = set(group["i_idx"].tolist())

        self.eval_users = sorted(list(self.test_ground_truth.keys()))
        logger.info(f"Evaluator initialized with {len(self.eval_users)} test users having positive targets.")

    def evaluate_model(
        self,
        model_name: str,
        rec_fn,
        user_histories: Dict[int, List[int]],
        max_users: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a model's recommendation function.
        rec_fn(user_idx, k=100, exclude=history) -> List[int]
        """
        users = self.eval_users[:max_users] if max_users else self.eval_users
        max_k = max(self.ks)

        hits_k = {k: [] for k in self.ks}
        recalls_k = {k: [] for k in self.ks}
        ndcgs = []
        maps = []
        diversities = []
        novelties = []
        all_recs_at_10 = []
        latencies = []

        for u in users:
            history = set(user_histories.get(u, []))
            target = self.test_ground_truth[u]

            t0 = time.perf_counter()
            recs = rec_fn(u, k=max_k, exclude=history)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

            for k in self.ks:
                hits_k[k].append(hit_at_k(recs, target, k=k))
                recalls_k[k].append(recall_at_k(recs, target, k=k))

            ndcgs.append(ndcg_at_k(recs, target, k=self.ranking_k))
            maps.append(map_at_k(recs, target, k=self.ranking_k))

            recs_10 = recs[:self.ranking_k]
            all_recs_at_10.append(recs_10)
            diversities.append(intra_list_diversity(recs_10, self.genre_matrix))
            novelties.append(novelty_score(recs_10, self.item_popularity, self.total_interactions))

        coverage = catalog_coverage(all_recs_at_10, self.total_items)

        metrics = {
            "model": model_name,
            "num_evaluated_users": len(users),
            "NDCG@10": float(round(np.mean(ndcgs), 4)),
            "MAP@10": float(round(np.mean(maps), 4)),
            "Hit@10": float(round(np.mean(hits_k[10]), 4)),
            "Coverage@10": float(round(coverage, 4)),
            "Diversity@10": float(round(np.mean(diversities), 4)),
            "Novelty@10": float(round(np.mean(novelties), 4)),
            "latency_p50_ms": float(round(np.percentile(latencies, 50), 3)),
            "latency_p95_ms": float(round(np.percentile(latencies, 95), 3)),
        }
        for k in self.ks:
            metrics[f"Recall@{k}"] = float(round(np.mean(recalls_k[k]), 4))

        return metrics
