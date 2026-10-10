"""Vector Index abstraction using FAISS for top-K candidate retrieval."""
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class VectorIndex:
    """FAISS-powered vector index with support for HNSW, IVF-Flat, and Flat index types."""

    def __init__(
        self,
        dim: int = 64,
        index_type: str = "HNSW",
        nlist: int = 64,
        nprobe: int = 16,
        hnsw_m: int = 32,
    ):
        self.dim = dim
        self.index_type = index_type.upper()
        self.nlist = nlist
        self.nprobe = nprobe
        self.hnsw_m = hnsw_m
        self.index = None
        self.item_vectors: Optional[np.ndarray] = None

    def build(self, vectors: np.ndarray) -> "VectorIndex":
        """
        Build index from embeddings (must be float32 and L2-normalized for cosine similarity).
        """
        num_items, d = vectors.shape
        assert d == self.dim, f"Vector dimension {d} does not match index dim {self.dim}"
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.item_vectors = vectors

        if not FAISS_AVAILABLE:
            logger.warning("FAISS not installed; using exact numpy cosine retrieval fallback.")
            return self

        if self.index_type == "HNSW":
            logger.info(f"Building FAISS HNSW index (M={self.hnsw_m}, dim={self.dim})...")
            self.index = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = 64
            self.index.add(vectors)
        elif self.index_type == "IVF_FLAT":
            n_clusters = min(self.nlist, max(4, num_items // 30))
            logger.info(f"Building FAISS IVF-Flat index (nlist={n_clusters}, dim={self.dim})...")
            quantizer = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            self.index.train(vectors)
            self.index.nprobe = min(self.nprobe, n_clusters)
            self.index.add(vectors)
        else:
            logger.info(f"Building FAISS FlatIP exact index (dim={self.dim})...")
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(vectors)

        logger.info(f"FAISS index built with {self.index.ntotal} vectors.")
        return self

    def search(self, query_vector: np.ndarray, k: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query the index for top-K neighbors.
        Returns: (scores, indices) of shapes (k,)
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)

        if self.index is not None and FAISS_AVAILABLE:
            scores, indices = self.index.search(query_vector, k)
            return scores[0], indices[0]
        else:
            # Fallback exact dot product
            scores = (self.item_vectors @ query_vector.T).squeeze(-1)
            top_k_idx = np.argpartition(-scores, k)[:k]
            top_k_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
            return scores[top_k_sorted], top_k_sorted

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.index is not None and FAISS_AVAILABLE:
            faiss.write_index(self.index, str(path))
        if self.item_vectors is not None:
            np.save(str(path.with_suffix(".npy")), self.item_vectors)
        logger.info(f"Saved vector index to {filepath}")

    def load(self, filepath: str) -> "VectorIndex":
        path = Path(filepath)
        if path.exists() and FAISS_AVAILABLE:
            self.index = faiss.read_index(str(path))
            logger.info(f"Loaded FAISS index from {filepath} ({self.index.ntotal} items)")
        npy_path = path.with_suffix(".npy")
        if npy_path.exists():
            self.item_vectors = np.load(str(npy_path))
        return self
