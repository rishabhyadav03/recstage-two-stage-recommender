"""Dataset loader and feature preprocessor for MovieLens datasets."""
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import logging
from .download import download_dataset
from .splits import create_temporal_splits

logger = logging.getLogger(__name__)

ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children's", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
# Canonicalize "Children" vs "Children's"
GENRE_TO_IDX = {g: i for i, g in enumerate(sorted(list(set(
    [g.replace("Children's", "Children") for g in ALL_GENRES]
))))}
NUM_GENRES = len(GENRE_TO_IDX)

AGE_MAP = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}


class MovieLensData:
    """Container for loaded and preprocessed MovieLens data."""

    def __init__(
        self,
        dataset_name: str = "ml-1m",
        raw_dir: str = "data/raw",
        min_user_interactions: int = 5,
        positive_rating_threshold: float = 4.0,
    ):
        self.dataset_name = dataset_name
        self.raw_dir = raw_dir
        self.min_user_interactions = min_user_interactions
        self.positive_rating_threshold = positive_rating_threshold

        self.users_df: pd.DataFrame = pd.DataFrame()
        self.movies_df: pd.DataFrame = pd.DataFrame()
        self.ratings_df: pd.DataFrame = pd.DataFrame()

        self.train_df: pd.DataFrame = pd.DataFrame()
        self.val_df: pd.DataFrame = pd.DataFrame()
        self.test_df: pd.DataFrame = pd.DataFrame()

        # Mappings
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.item_to_idx: Dict[int, int] = {}
        self.idx_to_item: Dict[int, int] = {}

        # Meta caches
        self.item_meta: Dict[int, Dict[str, Any]] = {}
        self.user_meta: Dict[int, Dict[str, Any]] = {}

    def load_and_process(self) -> "MovieLensData":
        """Main entrypoint to download, load, encode and split."""
        dataset_path = download_dataset(self.dataset_name, self.raw_dir)
        logger.info(f"Loading {self.dataset_name} files...")

        if self.dataset_name == "ml-1m":
            self._load_ml_1m(dataset_path)
        elif self.dataset_name == "ml-25m":
            self._load_ml_25m(dataset_path)
        else:
            raise ValueError(f"Unknown dataset {self.dataset_name}")

        self._preprocess_movies()
        self._preprocess_users()
        self._encode_ids_and_split()
        self._compute_train_aggregates()
        return self

    def _load_ml_1m(self, path: Path):
        movies_file = path / "movies.dat"
        users_file = path / "users.dat"
        ratings_file = path / "ratings.dat"

        # movies: MovieID::Title::Genres
        self.movies_df = pd.read_csv(
            movies_file,
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            encoding="latin-1",
        )

        # users: UserID::Gender::Age::Occupation::Zip-code
        self.users_df = pd.read_csv(
            users_file,
            sep="::",
            engine="python",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            encoding="latin-1",
        )

        # ratings: UserID::MovieID::Rating::Timestamp
        self.ratings_df = pd.read_csv(
            ratings_file,
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
            encoding="latin-1",
        )

    def _load_ml_25m(self, path: Path):
        movies_file = path / "movies.csv"
        ratings_file = path / "ratings.csv"

        self.movies_df = pd.read_csv(movies_file)
        self.movies_df.rename(columns={"movieId": "movie_id"}, inplace=True)

        self.ratings_df = pd.read_csv(ratings_file)
        self.ratings_df.rename(
            columns={"userId": "user_id", "movieId": "movie_id"}, inplace=True
        )

        # Create dummy users for 25m
        unique_users = self.ratings_df["user_id"].unique()
        self.users_df = pd.DataFrame({
            "user_id": unique_users,
            "gender": ["U"] * len(unique_users),
            "age": [25] * len(unique_users),
            "occupation": [0] * len(unique_users),
            "zip_code": ["00000"] * len(unique_users),
        })

    def _preprocess_movies(self):
        """Extract year and canonical genres."""
        def extract_year(title: str) -> int:
            match = re.search(r"\((\d{4})\)", title)
            if match:
                try:
                    return int(match.group(1))
                except ValueError:
                    return 1990
            return 1990

        self.movies_df["release_year"] = self.movies_df["title"].apply(extract_year)
        # Year normalized roughly between 1920 and 2000
        min_year, max_year = 1919, 2001
        self.movies_df["year_norm"] = np.clip(
            (self.movies_df["release_year"] - min_year) / (max_year - min_year), 0.0, 1.0
        )

        # Multi-hot genres
        def get_genre_vector(genre_str: str) -> np.ndarray:
            vec = np.zeros(NUM_GENRES, dtype=np.float32)
            if not isinstance(genre_str, str):
                return vec
            for g in genre_str.split("|"):
                clean_g = g.replace("Children's", "Children")
                if clean_g in GENRE_TO_IDX:
                    vec[GENRE_TO_IDX[clean_g]] = 1.0
            return vec

        self.movies_df["genre_vector"] = self.movies_df["genres"].apply(get_genre_vector)

    def _preprocess_users(self):
        """Encode gender, age, occupation."""
        gender_map = {"M": 0, "F": 1, "U": 0}
        self.users_df["gender_idx"] = self.users_df["gender"].map(lambda x: gender_map.get(x, 0))
        self.users_df["age_idx"] = self.users_df["age"].map(lambda x: AGE_MAP.get(x, 2))
        self.users_df["occupation_idx"] = self.users_df["occupation"].clip(0, 20)

    def _encode_ids_and_split(self):
        """Create contiguous indices and create train/val/test splits."""
        # Filter ratings for valid items only
        valid_items = set(self.movies_df["movie_id"].unique())
        self.ratings_df = self.ratings_df[self.ratings_df["movie_id"].isin(valid_items)].copy()

        # Split before defining user mappings to filter users with insufficient interactions
        self.train_df, self.val_df, self.test_df = create_temporal_splits(
            self.ratings_df, min_user_interactions=self.min_user_interactions
        )

        # Contiguous mappings based on filtered users and items
        active_users = sorted(self.train_df["user_id"].unique())
        self.user_to_idx = {u: i for i, u in enumerate(active_users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}

        active_items = sorted(self.movies_df["movie_id"].unique())
        self.item_to_idx = {m: i for i, m in enumerate(active_items)}
        self.idx_to_item = {i: m for m, i in self.item_to_idx.items()}

        # Filter val and test to keep users in user_to_idx
        self.train_df["u_idx"] = self.train_df["user_id"].map(self.user_to_idx)
        self.train_df["i_idx"] = self.train_df["movie_id"].map(self.item_to_idx)

        self.val_df = self.val_df[self.val_df["user_id"].isin(self.user_to_idx)].copy()
        self.val_df["u_idx"] = self.val_df["user_id"].map(self.user_to_idx)
        self.val_df["i_idx"] = self.val_df["movie_id"].map(self.item_to_idx)

        self.test_df = self.test_df[self.test_df["user_id"].isin(self.user_to_idx)].copy()
        self.test_df["u_idx"] = self.test_df["user_id"].map(self.user_to_idx)
        self.test_df["i_idx"] = self.test_df["movie_id"].map(self.item_to_idx)

        # Drop any item outside dictionary if any
        self.train_df = self.train_df.dropna(subset=["u_idx", "i_idx"]).astype({"u_idx": int, "i_idx": int})
        self.val_df = self.val_df.dropna(subset=["u_idx", "i_idx"]).astype({"u_idx": int, "i_idx": int})
        self.test_df = self.test_df.dropna(subset=["u_idx", "i_idx"]).astype({"u_idx": int, "i_idx": int})

        # Mark positive interactions for retrieval (rating >= threshold)
        for df in [self.train_df, self.val_df, self.test_df]:
            df["is_positive"] = (df["rating"] >= self.positive_rating_threshold).astype(int)

    def _compute_train_aggregates(self):
        """Compute user and item training statistics (NO LEAKAGE)."""
        logger.info("Computing training aggregates...")
        # Item popularity & average rating in train
        item_stats = self.train_df.groupby("i_idx").agg(
            train_count=("rating", "count"),
            train_pos_count=("is_positive", "sum"),
            train_mean_rating=("rating", "mean"),
        ).reset_index()

        # Map to all items
        num_items = len(self.item_to_idx)
        self.item_popularity = np.zeros(num_items, dtype=np.float32)
        self.item_pos_count = np.zeros(num_items, dtype=np.float32)
        self.item_mean_rating = np.full(num_items, 3.0, dtype=np.float32)

        for _, row in item_stats.iterrows():
            idx = int(row["i_idx"])
            self.item_popularity[idx] = row["train_count"]
            self.item_pos_count[idx] = row["train_pos_count"]
            self.item_mean_rating[idx] = row["train_mean_rating"]

        # Log popularity buckets (0-9)
        max_pop = max(1.0, float(np.max(self.item_popularity)))
        log_pop = np.log1p(self.item_popularity)
        self.item_pop_bucket = np.clip((log_pop / np.log1p(max_pop) * 9.99).astype(int), 0, 9)

        # User mean rating & activity level in train
        num_users = len(self.user_to_idx)
        self.user_mean_rating = np.full(num_users, 3.5, dtype=np.float32)
        self.user_activity = np.zeros(num_users, dtype=np.float32)
        self.user_top_genres = np.zeros((num_users, NUM_GENRES), dtype=np.float32)
        self.user_history_i_idx: Dict[int, List[int]] = {u: [] for u in range(num_users)}

        # Movie genre array
        self.movie_genres_matrix = np.zeros((num_items, NUM_GENRES), dtype=np.float32)
        for _, row in self.movies_df.iterrows():
            m_id = row["movie_id"]
            if m_id in self.item_to_idx:
                idx = self.item_to_idx[m_id]
                self.movie_genres_matrix[idx] = row["genre_vector"]

        # Group train by user
        train_sorted = self.train_df.sort_values(by=["u_idx", "timestamp"])
        for u_idx, group in train_sorted.groupby("u_idx"):
            self.user_mean_rating[u_idx] = group["rating"].mean()
            self.user_activity[u_idx] = len(group)
            self.user_history_i_idx[u_idx] = group["i_idx"].tolist()

            # Genre affinity from positive interactions
            pos_items = group[group["is_positive"] == 1]["i_idx"].tolist()
            if pos_items:
                genre_sum = self.movie_genres_matrix[pos_items].sum(axis=0)
                norm = np.linalg.norm(genre_sum)
                if norm > 0:
                    self.user_top_genres[u_idx] = genre_sum / norm

        # Build movie and user metadata dictionaries for API / frontend
        self._build_meta_dicts()

    def _build_meta_dicts(self):
        for _, row in self.movies_df.iterrows():
            m_id = int(row["movie_id"])
            if m_id in self.item_to_idx:
                idx = self.item_to_idx[m_id]
                self.item_meta[idx] = {
                    "movie_id": m_id,
                    "title": str(row["title"]),
                    "genres": str(row["genres"]).split("|"),
                    "release_year": int(row["release_year"]),
                    "year_norm": float(row["year_norm"]),
                    "genre_vector": row["genre_vector"].tolist(),
                    "popularity": int(self.item_popularity[idx]),
                    "mean_rating": float(round(self.item_mean_rating[idx], 2)),
                }

        user_indexed = self.users_df.set_index("user_id")
        for u_id, u_idx in self.user_to_idx.items():
            if u_id in user_indexed.index:
                row = user_indexed.loc[u_id]
                self.user_meta[u_idx] = {
                    "user_id": int(u_id),
                    "gender": str(row["gender"]),
                    "age": int(row["age"]),
                    "occupation": int(row["occupation"]),
                    "activity": int(self.user_activity[u_idx]),
                    "mean_rating": float(round(self.user_mean_rating[u_idx], 2)),
                }
            else:
                self.user_meta[u_idx] = {
                    "user_id": int(u_id),
                    "gender": "M",
                    "age": 25,
                    "occupation": 0,
                    "activity": int(self.user_activity[u_idx]),
                    "mean_rating": float(round(self.user_mean_rating[u_idx], 2)),
                }

    @property
    def num_users(self) -> int:
        return len(self.user_to_idx)

    @property
    def num_items(self) -> int:
        return len(self.item_to_idx)
