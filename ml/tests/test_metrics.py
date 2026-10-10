"""Unit tests for recommendation metrics."""
import pytest
import numpy as np
from ml.evaluate import (
    hit_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
    catalog_coverage,
    intra_list_diversity,
    novelty_score,
)


def test_hit_and_recall_at_k():
    ground_truth = {5, 10}
    recommended = [1, 2, 5, 4, 8]

    assert hit_at_k(recommended, ground_truth, k=2) == 0.0
    assert hit_at_k(recommended, ground_truth, k=3) == 1.0

    # Recall: 1 hit out of 2 ground truth items
    assert recall_at_k(recommended, ground_truth, k=3) == 0.5
    assert recall_at_k(recommended, ground_truth, k=2) == 0.0


def test_ndcg_at_k():
    ground_truth = {5}
    # Item 5 at position 0 (1-based rank 1): DCG = 1 / log2(2) = 1.0, IDCG = 1.0 -> NDCG = 1.0
    assert ndcg_at_k([5, 1, 2], ground_truth, k=3) == 1.0

    # Item 5 at position 1 (1-based rank 2): DCG = 1 / log2(3) = 0.6309, IDCG = 1.0 -> NDCG = 1 / log2(3)
    expected_ndcg = 1.0 / np.log2(3)
    assert pytest.approx(ndcg_at_k([1, 5, 2], ground_truth, k=3), 1e-4) == expected_ndcg

    # Empty ground truth or no hit
    assert ndcg_at_k([1, 2, 3], ground_truth, k=3) == 0.0
    assert ndcg_at_k([1, 2, 3], set(), k=3) == 0.0


def test_map_at_k():
    ground_truth = {5, 10}
    # Items at rank 1 (idx 0) and rank 3 (idx 2)
    # Precision@1 = 1/1, Precision@2 = 1/2, Precision@3 = 2/3
    # AP = (1/1 + 2/3) / 2 = (1 + 0.6667)/2 = 0.8333
    rec = [5, 1, 10, 2]
    expected_map = (1.0 + (2.0 / 3.0)) / 2.0
    assert pytest.approx(map_at_k(rec, ground_truth, k=4), 1e-4) == expected_map


def test_catalog_coverage():
    all_recs = [[1, 2], [2, 3], [1, 4]]
    # Unique = {1, 2, 3, 4} (4 items) out of total 10
    assert catalog_coverage(all_recs, total_catalog_items=10) == 0.4


def test_intra_list_diversity():
    # Genre matrix for 3 items with 2 genres
    genre_matrix = np.array([
        [1.0, 0.0],  # Item 0: Action
        [0.0, 1.0],  # Item 1: Comedy (orthogonal to item 0 -> distance = 1.0)
        [1.0, 0.0],  # Item 2: Action (identical to item 0 -> distance = 0.0)
    ], dtype=np.float32)

    # Recommending item 0 and item 1 -> distance = 1.0
    assert pytest.approx(intra_list_diversity([0, 1], genre_matrix), 1e-4) == 1.0
    # Recommending item 0 and item 2 -> distance = 0.0
    assert pytest.approx(intra_list_diversity([0, 2], genre_matrix), 1e-4) == 0.0


def test_novelty_score():
    # Item pop = [10, 100], total = 1000
    pop = np.array([10, 100], dtype=np.float32)
    # Item 0: -log2(10/1000) = -log2(0.01) = 6.6438
    # Item 1: -log2(100/1000) = -log2(0.1) = 3.3219
    score = novelty_score([0, 1], pop, total_interactions=1000)
    expected = (-np.log2(0.01) - np.log2(0.1)) / 2.0
    assert pytest.approx(score, 1e-4) == expected
