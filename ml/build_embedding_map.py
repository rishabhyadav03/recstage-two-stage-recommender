"""
Build 2D UMAP projection of item embeddings for the Embedding Map.

Reproducible offline script:
- Loads trained two-tower ITEM embeddings (64-dim L2-normalized).
- Projects to 2D with UMAP (cosine metric, fixed seed).
- Normalizes coordinates to [0, 1].
- Maps primary genres to 6 semantic groups.
- Saves data/embedding_map.json.
"""
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import umap

from ml.config import PipelineConfig
from ml.data.dataset import MovieLensData

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Config mapping primary genre to one of 6 semantic groups
GENRE_TO_GROUP = {
    "Animation": "Animation/Children's",
    "Children's": "Animation/Children's",
    "Children": "Animation/Children's",
    "Drama": "Drama",
    "Comedy": "Comedy",
    "Action": "Action/Adventure/Thriller",
    "Adventure": "Action/Adventure/Thriller",
    "Thriller": "Action/Adventure/Thriller",
    "Romance": "Romance/Musical",
    "Musical": "Romance/Musical",
}
DEFAULT_GROUP = "Other"


def get_group(first_genre: str) -> str:
    clean = first_genre.strip()
    return GENRE_TO_GROUP.get(clean, DEFAULT_GROUP)


def main():
    start_time = time.perf_counter()
    artifacts_dir = Path("artifacts")
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    out_file = data_dir / "embedding_map.json"

    # 1. Load pipeline config and data
    logger.info("Loading PipelineConfig and MovieLensData...")
    cfg = PipelineConfig.load()
    data = MovieLensData(
        dataset_name=cfg.data.dataset_name,
        raw_dir=cfg.data.raw_dir,
        min_user_interactions=cfg.data.min_user_interactions,
        positive_rating_threshold=cfg.data.positive_rating_threshold,
    ).load_and_process()

    # 2. Load trained two-tower item embeddings
    embeddings_file = artifacts_dir / "item_embeddings.npy"
    if not embeddings_file.exists():
        raise FileNotFoundError(f"Missing {embeddings_file}. Train retrieval model first.")

    item_embeddings = np.load(str(embeddings_file))
    num_items, dim = item_embeddings.shape
    logger.info(f"Loaded item embeddings: {num_items} items, {dim} dimensions.")

    # Popularity ranks (1 = most popular)
    rank_order = np.argsort(-data.item_popularity)
    popularity_ranks = np.empty_like(rank_order)
    popularity_ranks[rank_order] = np.arange(1, len(rank_order) + 1)

    # 3. Project to 2D using UMAP
    # Strict parameters: n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    logger.info("Projecting to 2D with UMAP (cosine metric, n_neighbors=15, min_dist=0.1, random_state=42)...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords_2d = reducer.fit_transform(item_embeddings)

    # 4. Normalize x and y to 0..1 range
    x_min, x_max = float(coords_2d[:, 0].min()), float(coords_2d[:, 0].max())
    y_min, y_max = float(coords_2d[:, 1].min()), float(coords_2d[:, 1].max())

    norm_x = (coords_2d[:, 0] - x_min) / (x_max - x_min if x_max > x_min else 1.0)
    norm_y = (coords_2d[:, 1] - y_min) / (y_max - y_min if y_max > y_min else 1.0)

    # Load posters if present
    posters = {}
    posters_file = data_dir / "posters.json"
    if posters_file.exists():
        try:
            with open(posters_file, "r", encoding="utf-8") as pf:
                posters = json.load(pf)
        except Exception:
            posters = {}

    # 5. Build output points
    points = []
    group_counts = {}
    for i in range(num_items):
        meta = data.item_meta.get(i, {})
        movie_id = int(meta.get("movie_id", data.idx_to_item.get(i, i)))
        title = str(meta.get("title", ""))
        genres = meta.get("genres", ["Unknown"])
        first_genre = genres[0] if genres else "Unknown"
        group = get_group(first_genre)
        group_counts[group] = group_counts.get(group, 0) + 1
        year = int(meta.get("release_year", 0))
        pop_rank = int(popularity_ranks[i])
        p_info = posters.get(str(movie_id), {})
        poster_path = p_info.get("poster_path")

        point_dict = {
            "movie_id": movie_id,
            "x": round(float(norm_x[i]), 5),
            "y": round(float(norm_y[i]), 5),
            "group": group,
            "title": title,
            "year": year,
            "popularity_rank": pop_rank,
        }
        if poster_path:
            point_dict["poster_path"] = poster_path
        points.append(point_dict)

    # Save data/embedding_map.json
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(points, f, indent=None)

    elapsed_time = time.perf_counter() - start_time
    print(f"Embedding map saved successfully to {out_file}")
    print(f"Points count: {len(points)}")
    print(f"Time taken: {elapsed_time:.2f} seconds")
    print(f"Group breakdown: {group_counts}")

    # Cluster validation check
    # Check average pairwise distance within groups vs overall
    print("\n--- Cluster validation check ---")
    groups_coords = {}
    for p in points:
        groups_coords.setdefault(p["group"], []).append([p["x"], p["y"]])

    for grp, coords in sorted(groups_coords.items()):
        c_arr = np.array(coords)
        centroid = c_arr.mean(axis=0)
        dispersion = np.mean(np.linalg.norm(c_arr - centroid, axis=1))
        print(f"Group '{grp}' (count={len(coords)}): centroid=({centroid[0]:.3f}, {centroid[1]:.3f}), mean dispersion={dispersion:.3f}")


if __name__ == "__main__":
    main()
