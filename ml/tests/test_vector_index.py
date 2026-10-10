"""Unit tests for VectorIndex (FAISS wrapper)."""
import pytest
import numpy as np
import tempfile
import os
from ml.models.vector_index import VectorIndex


DIM = 64
N = 500


@pytest.fixture
def random_vectors():
    """N L2-normalized random vectors of dimension DIM."""
    vecs = np.random.randn(N, DIM).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def flat_index(random_vectors):
    return VectorIndex(dim=DIM, index_type="FLAT").build(random_vectors)


@pytest.fixture
def hnsw_index(random_vectors):
    return VectorIndex(dim=DIM, index_type="HNSW").build(random_vectors)


def test_flat_index_top1_self_retrieval(flat_index, random_vectors):
    """Querying with a vector from the index should return itself at rank 0."""
    query = random_vectors[0]
    scores, indices = flat_index.search(query, k=5)
    assert indices[0] == 0, "Exact index should return query vector as top result"


def test_flat_index_output_lengths(flat_index, random_vectors):
    """Search must return exactly k results."""
    k = 10
    scores, indices = flat_index.search(random_vectors[7], k=k)
    assert len(scores) == k
    assert len(indices) == k


def test_flat_index_scores_descending(flat_index, random_vectors):
    """Cosine similarity scores must be in descending order."""
    scores, _ = flat_index.search(random_vectors[3], k=10)
    assert np.all(np.diff(scores) <= 1e-5), "Scores must be non-increasing"


def test_hnsw_index_approximate_top1(hnsw_index, random_vectors):
    """HNSW approximate search should still find the exact top-1 for well-separated vectors."""
    query = random_vectors[42]
    _, indices = hnsw_index.search(query, k=5)
    # HNSW might not always get exact top-1, but query itself should be within top-5
    assert 42 in indices[:5], "HNSW should find the query item in top-5"


def test_ivf_flat_index(random_vectors):
    """IVF-Flat index should build and return k results."""
    index = VectorIndex(dim=DIM, index_type="IVF_FLAT", nlist=16).build(random_vectors)
    scores, indices = index.search(random_vectors[10], k=5)
    assert len(indices) == 5


def test_save_load_roundtrip(flat_index, random_vectors):
    """Saved and loaded index should return same results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fpath = os.path.join(tmpdir, "test.bin")
        flat_index.save(fpath)

        loaded = VectorIndex(dim=DIM, index_type="FLAT").load(fpath)
        s1, i1 = flat_index.search(random_vectors[5], k=5)
        s2, i2 = loaded.search(random_vectors[5], k=5)

        np.testing.assert_array_equal(i1, i2)


def test_dimension_mismatch_raises():
    """Building with wrong-dim vectors should raise AssertionError."""
    vecs = np.random.randn(10, 32).astype(np.float32)
    with pytest.raises(AssertionError):
        VectorIndex(dim=64).build(vecs)
