"""
Two-Stage Recommendation System: FastAPI Backend

Endpoints:
  GET /api/health                      — liveness check
  GET /api/metrics                     — all benchmark results from metrics.json
  GET /api/users/{user_id}             — user profile + watch history
  GET /api/recommend/{user_id}         — top-N recommendations with explanations
  GET /api/similar/{item_id}           — FAISS nearest-neighbour similar items
  GET /api/search?q=<query>            — title-substring search over catalog
"""
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.config import PipelineConfig
from ml.data.dataset import MovieLensData, NUM_GENRES
from ml.models.two_tower import TwoTowerModel
from ml.models.vector_index import VectorIndex
from ml.models.ranker import LambdaRanker
from ml.features.feature_engineer import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str
    artifacts_loaded: bool
    num_users: int
    num_items: int


class ItemCard(BaseModel):
    item_id: int
    title: str
    genres: List[str]
    release_year: int
    avg_rating: float
    popularity_rank: int
    why: str  # Human-readable explanation
    poster_url: Optional[str] = None
    poster_url_large: Optional[str] = None
    backdrop_url: Optional[str] = None
    overview: Optional[str] = None


class UserProfile(BaseModel):
    user_id: int
    gender: str
    age: int
    occupation: int
    mean_rating: float
    total_interactions: int
    top_genres: List[str]
    genre_affinities: Optional[List[Dict[str, Any]]] = None
    watch_history: List[Dict[str, Any]]  # Last 10 watched items


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[ItemCard]
    retrieval_ms: float
    ranking_ms: float
    total_ms: float


class SimilarResponse(BaseModel):
    item_id: int
    title: str
    poster_url: Optional[str] = None
    poster_url_large: Optional[str] = None
    backdrop_url: Optional[str] = None
    overview: Optional[str] = None
    similar_items: List[ItemCard]
    query_ms: float


class CompareItem(BaseModel):
    movie_id: int
    title: str
    year: int
    genres: List[str]
    poster_url: Optional[str] = None
    poster_url_large: Optional[str] = None
    backdrop_url: Optional[str] = None
    overview: Optional[str] = None
    score: float
    rank: int
    # Ranks in other lists
    popularity_rank: Optional[int] = None
    retrieval_rank: Optional[int] = None
    two_stage_rank: Optional[int] = None
    rank_in_popularity: Optional[int] = None
    rank_in_retrieval: Optional[int] = None
    rank_in_two_stage: Optional[int] = None
    retrieval_candidate_rank: Optional[int] = None
    watch_count: Optional[int] = None


class ListStats(BaseModel):
    avg_popularity_rank: float
    genre_diversity: float
    long_tail_share: float
    latency_ms: float


class HeldOutInfo(BaseModel):
    movie_id: Optional[int] = None
    title: Optional[str] = None
    year: Optional[int] = None
    genres: Optional[List[str]] = None
    poster_url: Optional[str] = None
    rank_in_popularity: Optional[int] = None
    rank_in_retrieval: Optional[int] = None
    rank_in_two_stage: Optional[int] = None
    retrieval_candidate_rank: Optional[int] = None


class CompareResponse(BaseModel):
    user_id: int
    k: int
    held_out_movie_id: Optional[int] = None
    held_out_ranks: Dict[str, Optional[int]] = {}
    held_out: Optional[HeldOutInfo] = None
    popularity: List[CompareItem]
    retrieval_only: List[CompareItem]
    two_stage: List[CompareItem]
    stats: Dict[str, ListStats]


# ---------------------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------------------

_cfg: Optional[PipelineConfig] = None
_data: Optional[MovieLensData] = None
_model: Optional[TwoTowerModel] = None
_index: Optional[VectorIndex] = None
_ranker: Optional[LambdaRanker] = None
_feature_eng: Optional[FeatureEngineer] = None
_item_embeddings: Optional[np.ndarray] = None
_popularity_rank: Optional[np.ndarray] = None  # item -> rank (1=most popular)
_posters: Dict[str, Any] = {}
_winning_users: List[Dict[str, Any]] = []
_embedding_map_raw: Optional[str] = None
_embedding_map: Optional[List[Dict[str, Any]]] = None
_movie_id_to_xy: Dict[int, Tuple[float, float]] = {}
_movie_id_to_group: Dict[int, str] = {}

ARTIFACTS_DIR = Path("artifacts")
RESULTS_DIR = Path("results")
POSTERS_FILE = Path("data/posters.json")
WINNERS_FILE = ARTIFACTS_DIR / "winning_users.json"
EMBEDDING_MAP_FILE = Path("data/embedding_map.json")

GENRE_NAMES = [
    "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "Unknown",
]

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cfg, _data, _model, _index, _ranker, _feature_eng, _item_embeddings, _popularity_rank, _posters, _winning_users, _embedding_map_raw, _embedding_map, _movie_id_to_xy, _movie_id_to_group
    logger.info("Loading pipeline config and data...")
    _cfg = PipelineConfig.load()

    _data = MovieLensData(
        dataset_name=_cfg.data.dataset_name,
        raw_dir=_cfg.data.raw_dir,
        min_user_interactions=_cfg.data.min_user_interactions,
        positive_rating_threshold=_cfg.data.positive_rating_threshold,
    ).load_and_process()

    # Load posters if present
    if POSTERS_FILE.exists():
        try:
            with open(POSTERS_FILE, "r", encoding="utf-8") as f:
                _posters = json.load(f)
            logger.info(f"Loaded {len(_posters)} movie posters from {POSTERS_FILE}")
        except Exception as e:
            logger.warning(f"Could not load {POSTERS_FILE}: {e}")
            _posters = {}
    else:
        logger.info(f"{POSTERS_FILE} not found yet; posters will be loaded once generated.")
        _posters = {}

    # Load precomputed winning users if present
    if WINNERS_FILE.exists():
        try:
            with open(WINNERS_FILE, "r", encoding="utf-8") as f:
                _winning_users = json.load(f)
            logger.info(f"Loaded {len(_winning_users)} precomputed winning users from {WINNERS_FILE}")
        except Exception as e:
            logger.warning(f"Could not load {WINNERS_FILE}: {e}")
            _winning_users = []
    else:
        _winning_users = []

    # Load embedding map if present
    if EMBEDDING_MAP_FILE.exists():
        try:
            with open(EMBEDDING_MAP_FILE, "r", encoding="utf-8") as f:
                _embedding_map_raw = f.read()
                _embedding_map = json.loads(_embedding_map_raw)
            _movie_id_to_xy = {int(p["movie_id"]): (float(p["x"]), float(p["y"])) for p in _embedding_map}
            _movie_id_to_group = {int(p["movie_id"]): str(p.get("group", "Other")) for p in _embedding_map}
            logger.info(f"Loaded {len(_embedding_map)} embedding map points from {EMBEDDING_MAP_FILE}")
        except Exception as e:
            logger.warning(f"Could not load {EMBEDDING_MAP_FILE}: {e}")
            _embedding_map_raw = None
            _embedding_map = None
            _movie_id_to_xy = {}
            _movie_id_to_group = {}
    else:
        logger.info(f"{EMBEDDING_MAP_FILE} not found. Run ml/build_embedding_map.py to generate it.")
        _embedding_map_raw = None
        _embedding_map = None
        _movie_id_to_xy = {}
        _movie_id_to_group = {}

    logger.info("Loading Two-Tower model...")
    _model = TwoTowerModel(
        num_users=_data.num_users,
        num_items=_data.num_items,
        num_genres=NUM_GENRES,
        embedding_dim=_cfg.retrieval.embedding_dim,
    )
    _model.load_state_dict(
        torch.load(ARTIFACTS_DIR / "two_tower.pt", weights_only=True)
    )
    _model.eval()

    logger.info("Loading FAISS index...")
    _index = VectorIndex(dim=_cfg.retrieval.embedding_dim).load(
        str(ARTIFACTS_DIR / "faiss_index.bin")
    )

    _item_embeddings = np.load(str(ARTIFACTS_DIR / "item_embeddings.npy"))

    logger.info("Loading LambdaRanker...")
    _ranker = LambdaRanker().load(str(ARTIFACTS_DIR / "ranker.txt"))

    _feature_eng = FeatureEngineer(
        data=_data,
        item_embeddings=_item_embeddings,
        last_n=_cfg.data.last_n_watched,
    )

    # Pre-compute popularity rank array (1=most popular)
    rank_order = np.argsort(-_data.item_popularity)
    _popularity_rank = np.empty_like(rank_order)
    _popularity_rank[rank_order] = np.arange(1, len(rank_order) + 1)

    logger.info(
        f"All models loaded. {_data.num_users} users, {_data.num_items} items."
    )
    yield


app = FastAPI(
    title="Two-Stage Recommendation System API",
    description=(
        "Portfolio project demonstrating a production-grade two-stage recommendation "
        "system: Two-Tower DNN retrieval (FAISS) + LightGBM LambdaRank re-ranking."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_user_embedding(u_idx: int) -> np.ndarray:
    u_meta = _data.user_meta[u_idx]
    hist = _data.user_history_i_idx.get(u_idx, [])[-_cfg.data.last_n_watched :]
    pad_len = _cfg.data.last_n_watched - len(hist)
    padded_hist = [_data.num_items] * pad_len + hist

    with torch.no_grad():
        u_emb = _model.user_tower(
            user_ids=torch.tensor([u_idx], dtype=torch.long),
            gender_idx=torch.tensor([1 if u_meta["gender"] == "F" else 0], dtype=torch.long),
            age_idx=torch.tensor([min(6, max(0, u_meta["age"] // 10))], dtype=torch.long),
            occ_idx=torch.tensor([min(21, max(0, u_meta["occupation"]))], dtype=torch.long),
            mean_rating=torch.tensor([u_meta["mean_rating"]], dtype=torch.float32),
            top_genres=torch.from_numpy(_data.user_top_genres[u_idx : u_idx + 1]),
            history_items=torch.tensor([padded_hist], dtype=torch.long),
        ).cpu().numpy()[0]
    return u_emb


def _get_poster_info(movie_id: int) -> Dict[str, Optional[str]]:
    entry = _posters.get(str(movie_id))
    if not entry:
        return {"poster_url": None, "poster_url_large": None, "backdrop_url": None, "overview": None}
    p_path = entry.get("poster_path") or entry.get("poster_url")
    b_path = entry.get("backdrop_path")

    if p_path:
        if p_path.startswith("http"):
            poster_url = p_path
            poster_url_large = p_path
        else:
            clean_p = p_path if p_path.startswith("/") else f"/{p_path}"
            poster_url = f"https://image.tmdb.org/t/p/w342{clean_p}"
            poster_url_large = f"https://image.tmdb.org/t/p/w500{clean_p}"
    else:
        poster_url = None
        poster_url_large = None

    if b_path:
        if b_path.startswith("http"):
            backdrop_url = b_path
        else:
            clean_b = b_path if b_path.startswith("/") else f"/{b_path}"
            backdrop_url = f"https://image.tmdb.org/t/p/w780{clean_b}"
    else:
        backdrop_url = None

    return {
        "poster_url": poster_url,
        "poster_url_large": poster_url_large,
        "backdrop_url": backdrop_url,
        "overview": entry.get("overview") or None,
    }


def _clean_display_title(raw_title: str) -> str:
    clean = re.sub(r"\s*\(\d{4}\)$", "", raw_title).strip()
    m = re.match(r"^(.*?),\s*(The|A|An|La|Le|Les|El|Il)$", clean, re.IGNORECASE)
    if m:
        return f"{m.group(2)} {m.group(1)}"
    return clean


def _generate_editorial_why(i_idx: int, u_idx: Optional[int] = None) -> str:
    meta = _data.item_meta[i_idx]
    item_genres_vec = _data.movie_genres_matrix[i_idx]
    genre_names = [GENRE_NAMES[g] for g in range(len(item_genres_vec)) if item_genres_vec[g] > 0]
    primary_genre = genre_names[0].lower() if genre_names else "film"

    watched_titles = []
    if u_idx is not None:
        hist_idxs = _data.user_history_i_idx.get(u_idx, [])
        for h_idx in reversed(hist_idxs[-20:]):
            h_genres = _data.movie_genres_matrix[h_idx]
            if np.any((item_genres_vec > 0) & (h_genres > 0)):
                h_title = _clean_display_title(str(_data.item_meta[h_idx]["title"]))
                if h_title not in watched_titles:
                    watched_titles.append(h_title)
            if len(watched_titles) >= 2:
                break

    if len(watched_titles) >= 2:
        return f"Because you watched {watched_titles[0]} and {watched_titles[1]}, and you rate {primary_genre}s highly."
    elif len(watched_titles) == 1:
        return f"Because you watched {watched_titles[0]} and gravitate toward {primary_genre} cinema."
    elif genre_names:
        genres_str = " & ".join(genre_names[:2]).lower()
        avg_r = float(_data.item_mean_rating[i_idx])
        if avg_r >= 4.2:
            return f"An acclaimed {genres_str} classic that strongly matches your profile."
        return f"Because you enjoy {genres_str} films with compelling storytelling."
    return "A high-affinity recommendation based on your overall viewing taste."


def _item_card(i_idx: int, u_emb: Optional[np.ndarray] = None, rank: int = 0, u_idx: Optional[int] = None) -> ItemCard:
    meta = _data.item_meta[i_idx]
    movie_id = int(meta["movie_id"])
    avg_r = float(_data.item_mean_rating[i_idx])
    pop_r = int(_popularity_rank[i_idx])

    item_genres_vec = _data.movie_genres_matrix[i_idx]
    genre_names = [GENRE_NAMES[g] for g in range(len(item_genres_vec)) if item_genres_vec[g] > 0]
    why = _generate_editorial_why(i_idx, u_idx=u_idx)
    posters = _get_poster_info(movie_id)

    return ItemCard(
        item_id=movie_id,
        title=str(meta["title"]),
        genres=genre_names,
        release_year=int(meta["release_year"]),
        avg_rating=round(avg_r, 2),
        popularity_rank=pop_r,
        why=why,
        poster_url=posters["poster_url"],
        poster_url_large=posters["poster_url_large"],
        backdrop_url=posters["backdrop_url"],
        overview=posters["overview"],
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check — returns OK when all models are loaded."""
    loaded = all(x is not None for x in [_model, _index, _ranker])
    return HealthResponse(
        status="ok" if loaded else "loading",
        artifacts_loaded=loaded,
        num_users=_data.num_users if _data else 0,
        num_items=_data.num_items if _data else 0,
    )


@app.get("/api/metrics", tags=["System"])
def get_metrics():
    """Return all benchmark metrics from results/metrics.json."""
    path = RESULTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/users/{user_id}", response_model=UserProfile, tags=["Users"])
def get_user(user_id: int):
    """Return user profile and last-10 watch history."""
    if user_id not in _data.user_to_idx:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    u_idx = _data.user_to_idx[user_id]
    u_meta = _data.user_meta[u_idx]

    # Top genres (name-mapped)
    genre_vec = _data.user_top_genres[u_idx]
    genre_pairs = sorted(enumerate(genre_vec), key=lambda x: -x[1])
    top_genre_names = [GENRE_NAMES[g] for g, v in genre_pairs if v > 0][:5]

    top_pairs = [p for p in genre_pairs if p[1] > 0][:5]
    top_sum = sum(p[1] for p in top_pairs) or 1.0
    genre_affinities = [
        {"genre": GENRE_NAMES[g], "percentage": max(1, int(round((v / top_sum) * 100)))}
        for g, v in top_pairs
    ]

    # Last-10 watched items
    history_idxs = _data.user_history_i_idx.get(u_idx, [])[-10:]
    watch_history = []
    for i_idx in history_idxs:
        meta = _data.item_meta.get(i_idx, {})
        m_id = int(meta.get("movie_id", 0))
        p_info = _get_poster_info(m_id)
        watch_history.append({
            "item_id": m_id,
            "title": str(meta.get("title", "Unknown")),
            "genres": [GENRE_NAMES[g] for g in range(NUM_GENRES) if _data.movie_genres_matrix[i_idx][g] > 0],
            "release_year": int(meta.get("release_year", 0)),
            "poster_url": p_info["poster_url"],
            "poster_url_small": p_info["poster_url"],
        })

    return UserProfile(
        user_id=user_id,
        gender=str(u_meta.get("gender", "U")),
        age=int(u_meta.get("age", 0)),
        occupation=int(u_meta.get("occupation", 0)),
        mean_rating=float(round(_data.user_mean_rating[u_idx], 2)),
        total_interactions=int(_data.user_activity[u_idx]),
        top_genres=top_genre_names,
        genre_affinities=genre_affinities,
        watch_history=watch_history,
    )


@app.get("/api/recommend/{user_id}", response_model=RecommendResponse, tags=["Recommendations"])
def recommend(
    user_id: int,
    n: int = Query(default=10, ge=1, le=50, description="Number of recommendations"),
    mode: str = Query(default="two_stage", enum=["two_stage", "retrieval_only"], description="Pipeline mode"),
):
    """
    Full two-stage recommendations for a user.
    Stage 1: Two-Tower FAISS retrieval (top-200 candidates).
    Stage 2: LightGBM LambdaRank re-ranking to top-N.
    Includes editorial explanation ('why') for each result.
    """
    if user_id not in _data.user_to_idx:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    u_idx = _data.user_to_idx[user_id]
    exclude = set(_data.user_history_i_idx.get(u_idx, []))

    # Stage 1: Retrieval
    t0 = time.perf_counter()
    u_emb = _get_user_embedding(u_idx)
    query_k = min(_data.num_items, _cfg.retrieval.top_k_candidates + len(exclude) + 10)
    _, cand_indices = _index.search(u_emb, k=query_k)
    cands = [int(i) for i in cand_indices if i not in exclude][: _cfg.retrieval.top_k_candidates]
    t1 = time.perf_counter()
    retrieval_ms = round((t1 - t0) * 1000, 2)

    if mode == "retrieval_only" or _ranker is None or _ranker.model is None:
        top_items = cands[:n]
        ranking_ms = 0.0
    else:
        # Stage 2: Ranking
        t2 = time.perf_counter()
        feats = _feature_eng.extract_features_for_user_candidates(u_idx, u_emb, cands)
        scores = _ranker.predict(feats)
        top_indices = np.argsort(-scores)[:n]
        top_items = [cands[i] for i in top_indices]
        t3 = time.perf_counter()
        ranking_ms = round((t3 - t2) * 1000, 2)

    total_ms = round((time.perf_counter() - t0) * 1000, 2)

    recs = [_item_card(i_idx, u_emb=u_emb, rank=pos, u_idx=u_idx) for pos, i_idx in enumerate(top_items)]
    return RecommendResponse(
        user_id=user_id,
        recommendations=recs,
        retrieval_ms=retrieval_ms,
        ranking_ms=ranking_ms,
        total_ms=total_ms,
    )


@app.get("/api/similar/{item_id}", response_model=SimilarResponse, tags=["Items"])
def similar_items(
    item_id: int,
    n: int = Query(default=10, ge=1, le=50),
):
    """Return top-N FAISS nearest-neighbour items for a given item."""
    if item_id not in _data.item_to_idx:
        raise HTTPException(status_code=404, detail=f"Item {item_id} not found")
    i_idx = _data.item_to_idx[item_id]

    t0 = time.perf_counter()
    q_emb = _item_embeddings[i_idx]
    _, indices = _index.search(q_emb, k=n + 1)
    # Exclude the query item itself
    sim_idxs = [int(j) for j in indices if int(j) != i_idx][:n]
    query_ms = round((time.perf_counter() - t0) * 1000, 2)

    similar = [_item_card(j) for j in sim_idxs]
    meta = _data.item_meta[i_idx]
    m_id = int(meta["movie_id"])
    p_info = _get_poster_info(m_id)
    return SimilarResponse(
        item_id=item_id,
        title=str(meta["title"]),
        poster_url=p_info["poster_url"],
        poster_url_large=p_info["poster_url_large"],
        backdrop_url=p_info["backdrop_url"],
        overview=p_info["overview"],
        similar_items=similar,
        query_ms=query_ms,
    )


@app.get("/api/search", tags=["Items"])
def search(
    q: str = Query(..., min_length=2, description="Title substring to search"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Full-text title substring search over the movie catalog."""
    q_lower = q.lower()
    results = []
    for i_idx, meta in _data.item_meta.items():
        if q_lower in str(meta.get("title", "")).lower():
            m_id = int(meta["movie_id"])
            p_info = _get_poster_info(m_id)
            results.append({
                "item_id": m_id,
                "title": str(meta["title"]),
                "genres": [GENRE_NAMES[g] for g in range(NUM_GENRES) if _data.movie_genres_matrix[i_idx][g] > 0],
                "release_year": int(meta.get("release_year", 0)),
                "avg_rating": float(round(_data.item_mean_rating[i_idx], 2)),
                "popularity_rank": int(_popularity_rank[i_idx]),
                "poster_url": p_info["poster_url"],
                "poster_url_large": p_info["poster_url_large"],
                "backdrop_url": p_info["backdrop_url"],
                "overview": p_info["overview"],
            })
        if len(results) >= limit:
            break

    results.sort(key=lambda x: x["popularity_rank"])
    return {"query": q, "total": len(results), "results": results[:limit]}


def _compute_list_stats(item_idxs: List[int], latency_ms: float) -> ListStats:
    if not item_idxs:
        return ListStats(avg_popularity_rank=0.0, genre_diversity=0.0, long_tail_share=0.0, latency_ms=round(latency_ms, 1))

    # 1. Average item popularity rank (from global training set frequency)
    pop_ranks = [_popularity_rank[i] for i in item_idxs]
    avg_pop = float(np.mean(pop_ranks))

    # 2. Genre diversity (intra-list distance: 1 - mean cosine similarity of genre vectors)
    if len(item_idxs) > 1:
        G = _data.movie_genres_matrix[item_idxs].astype(np.float32)
        norms = np.linalg.norm(G, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        G_norm = G / norms
        sim_mat = G_norm @ G_norm.T
        triu_idx = np.triu_indices(len(item_idxs), k=1)
        mean_sim = float(np.mean(sim_mat[triu_idx]))
        diversity = float(np.clip(1.0 - mean_sim, 0.0, 1.0))
    else:
        diversity = 0.0

    # 3. Share of long-tail items (outside top 20% most popular items in the catalog)
    cutoff_20 = int(0.20 * _data.num_items)
    long_tail_count = sum(1 for r in pop_ranks if r > cutoff_20)
    long_tail_share = float(long_tail_count / len(item_idxs))

    return ListStats(
        avg_popularity_rank=round(avg_pop, 1),
        genre_diversity=round(diversity, 3),
        long_tail_share=round(long_tail_share, 3),
        latency_ms=round(latency_ms, 1),
    )


def _build_compare_item(
    i_idx: int,
    rank: int,
    score: float,
    pop_items: List[int],
    ret_items: List[int],
    ts_items: List[int],
    cands: List[int],
) -> CompareItem:
    meta = _data.item_meta[i_idx]
    m_id = int(meta["movie_id"])
    p_info = _get_poster_info(m_id)
    genres = [GENRE_NAMES[g] for g in range(NUM_GENRES) if _data.movie_genres_matrix[i_idx][g] > 0]

    r_pop = (pop_items.index(i_idx) + 1) if i_idx in pop_items else None
    r_ret = (ret_items.index(i_idx) + 1) if i_idx in ret_items else None
    r_ts = (ts_items.index(i_idx) + 1) if i_idx in ts_items else None
    cand_r = (cands.index(i_idx) + 1) if i_idx in cands else None

    return CompareItem(
        movie_id=m_id,
        title=str(meta.get("title", "")),
        year=int(meta.get("release_year", 0)),
        genres=genres,
        poster_url=p_info["poster_url"],
        poster_url_large=p_info["poster_url_large"],
        backdrop_url=p_info["backdrop_url"],
        overview=p_info["overview"],
        score=round(float(score), 4),
        rank=rank,
        popularity_rank=r_pop,
        retrieval_rank=r_ret,
        two_stage_rank=r_ts,
        rank_in_popularity=r_pop,
        rank_in_retrieval=r_ret,
        rank_in_two_stage=r_ts,
        retrieval_candidate_rank=cand_r,
        watch_count=int(_data.item_popularity[i_idx]) if hasattr(_data, "item_popularity") else None,
    )


@app.get("/api/compare/winning-users", tags=["Compare"])
def get_winning_users():
    """Return precomputed list of users where Two-stage ranks the held-out movie higher than Retrieval."""
    return _winning_users


@app.get("/api/compare/{user_id}", response_model=CompareResponse, tags=["Compare"])
def compare(
    user_id: int,
    k: int = Query(default=10, ge=1, le=50, description="Number of items per list"),
):
    """
    Compare recommendation lists across three paradigms:
    1. Popularity baseline (most watched in training set, excluding user history)
    2. Retrieval-only (Two-Tower FAISS top-k)
    3. Two-Stage (Two-Tower FAISS retrieval -> LambdaRank re-ranking)
    Also returns held-out movie evaluation from the leave-last-out test split (strictly for display).
    """
    if user_id not in _data.user_to_idx:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    u_idx = _data.user_to_idx[user_id]
    exclude = set(_data.user_history_i_idx.get(u_idx, []))

    # --- 1. Popularity list ---
    t0_pop = time.perf_counter()
    pop_sorted = np.argsort(-_data.item_popularity)
    pop_items = [int(i) for i in pop_sorted if i not in exclude][:k]
    max_pop = float(np.max(_data.item_popularity)) or 1.0
    pop_scores = {i: float(_data.item_popularity[i]) / max_pop for i in pop_items}
    pop_latency = (time.perf_counter() - t0_pop) * 1000

    # --- 2. Retrieval-only list ---
    t0_ret = time.perf_counter()
    u_emb = _get_user_embedding(u_idx)
    query_k = min(_data.num_items, _cfg.retrieval.top_k_candidates + len(exclude) + 10)
    cand_distances, cand_indices = _index.search(u_emb, k=query_k)
    cands = []
    ret_scores = {}
    for idx, score in zip(cand_indices, cand_distances):
        i = int(idx)
        if i not in exclude:
            cands.append(i)
            ret_scores[i] = float(score)
        if len(cands) >= _cfg.retrieval.top_k_candidates:
            break
    ret_items = cands[:k]
    ret_latency = (time.perf_counter() - t0_ret) * 1000

    # --- 3. Two-stage list ---
    t0_rank = time.perf_counter()
    feats = _feature_eng.extract_features_for_user_candidates(u_idx, u_emb, cands)
    rank_scores = _ranker.predict(feats)
    ranked_indices = np.argsort(-rank_scores)
    ts_items = [cands[i] for i in ranked_indices[:k]]
    ts_scores = {cands[i]: float(rank_scores[i]) for i in ranked_indices}
    rank_latency = (time.perf_counter() - t0_rank) * 1000
    ts_total_latency = ret_latency + rank_latency

    # Build item cards with cross-list ranks
    pop_list = [
        _build_compare_item(i, rank=pos + 1, score=pop_scores[i], pop_items=pop_items, ret_items=ret_items, ts_items=ts_items, cands=cands)
        for pos, i in enumerate(pop_items)
    ]
    ret_list = [
        _build_compare_item(i, rank=pos + 1, score=ret_scores.get(i, 0.0), pop_items=pop_items, ret_items=ret_items, ts_items=ts_items, cands=cands)
        for pos, i in enumerate(ret_items)
    ]
    ts_list = [
        _build_compare_item(i, rank=pos + 1, score=ts_scores.get(i, 0.0), pop_items=pop_items, ret_items=ret_items, ts_items=ts_items, cands=cands)
        for pos, i in enumerate(ts_items)
    ]

    # --- Held-out movie check (strictly for display, never used in ranking) ---
    held_out_info = None
    held_out_movie_id = None
    held_out_ranks = {"popularity": None, "retrieval_only": None, "two_stage": None}
    test_rows = _data.test_df[_data.test_df["u_idx"] == u_idx]
    if len(test_rows) > 0:
        held_out_i = int(test_rows.iloc[0]["i_idx"])
        held_out_movie_id = int(test_rows.iloc[0]["movie_id"])
        h_meta = _data.item_meta[held_out_i]
        h_posters = _get_poster_info(held_out_movie_id)

        r_pop = (pop_items.index(held_out_i) + 1) if held_out_i in pop_items else None
        r_ret = (ret_items.index(held_out_i) + 1) if held_out_i in ret_items else None
        r_ts = (ts_items.index(held_out_i) + 1) if held_out_i in ts_items else None
        r_cand = (cands.index(held_out_i) + 1) if held_out_i in cands else None

        held_out_ranks = {
            "popularity": r_pop,
            "retrieval_only": r_ret,
            "two_stage": r_ts,
        }

        held_out_info = HeldOutInfo(
            movie_id=held_out_movie_id,
            title=str(h_meta.get("title", "")),
            year=int(h_meta.get("release_year", 0)),
            genres=[GENRE_NAMES[g] for g in range(NUM_GENRES) if _data.movie_genres_matrix[held_out_i][g] > 0],
            poster_url=h_posters["poster_url"],
            rank_in_popularity=r_pop,
            rank_in_retrieval=r_ret,
            rank_in_two_stage=r_ts,
            retrieval_candidate_rank=r_cand,
        )

    stats = {
        "popularity": _compute_list_stats(pop_items, pop_latency),
        "retrieval_only": _compute_list_stats(ret_items, ret_latency),
        "two_stage": _compute_list_stats(ts_items, ts_total_latency),
    }

    return CompareResponse(
        user_id=user_id,
        k=k,
        held_out_movie_id=held_out_movie_id,
        held_out_ranks=held_out_ranks,
        held_out=held_out_info,
        popularity=pop_list,
        retrieval_only=ret_list,
        two_stage=ts_list,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Embedding Map Endpoints
# ---------------------------------------------------------------------------

class RetrievedItem(BaseModel):
    movie_id: int
    title: str
    year: int
    group: str
    poster_url: Optional[str] = None
    similarity: float
    match_pct: int


class MapUserResponse(BaseModel):
    user_id: int
    user_xy: List[float]
    history_ids: List[int]
    recommended_ids: List[int]
    top_retrieved: List[RetrievedItem] = []


@app.get("/api/map", tags=["Embedding Map"])
def get_embedding_map():
    """Return precomputed 2D UMAP projection points for all catalog movies."""
    if _embedding_map_raw is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding map not built yet. Please run `python ml/build_embedding_map.py` to generate data/embedding_map.json.",
        )
    return Response(
        content=_embedding_map_raw,
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=86400, stale-while-revalidate=3600"},
    )


@app.get("/api/map/user/{user_id}", response_model=MapUserResponse, tags=["Embedding Map"])
def get_user_map_state(user_id: int, k: int = Query(default=10, ge=1, le=50)):
    """
    Return user-specific embedding map projection:
    - user_xy: [x, y] coordinates computed as the average of the 2D positions
      of the user's top 20 retrieved items, weighted by retrieval score.
    - history_ids: movies the user watched in training.
    - recommended_ids: top k movies recommended by the Two-Stage model.
    - top_retrieved: top 5 nearest movies in embedding space by retrieval score.
    """
    if _embedding_map is None or not _movie_id_to_xy:
        raise HTTPException(
            status_code=503,
            detail="Embedding map not built yet. Please run `python ml/build_embedding_map.py` to generate data/embedding_map.json.",
        )
    if user_id not in _data.user_to_idx:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    u_idx = _data.user_to_idx[user_id]
    exclude = set(_data.user_history_i_idx.get(u_idx, []))
    history_ids = [_data.idx_to_item[i] for i in _data.user_history_i_idx.get(u_idx, []) if i in _data.idx_to_item]

    # Two-Tower candidate retrieval
    u_emb = _get_user_embedding(u_idx)
    query_k = min(_data.num_items, _cfg.retrieval.top_k_candidates + len(exclude) + 10)
    cand_distances, cand_indices = _index.search(u_emb, k=query_k)

    cands = []
    ret_scores = {}
    for idx, score in zip(cand_indices, cand_distances):
        i = int(idx)
        if i not in exclude:
            cands.append(i)
            ret_scores[i] = float(score)
        if len(cands) >= _cfg.retrieval.top_k_candidates:
            break

    # Top 5 items by retrieval score for this user (nearest items in embedding space)
    top_retrieved = []
    for i in cands[:5]:
        m_id = _data.idx_to_item.get(i)
        if not m_id:
            continue
        meta = _data.item_meta.get(i, {})
        title = str(meta.get("title", ""))
        year = int(meta.get("release_year", 0))
        p_info = _get_poster_info(m_id)
        grp = _movie_id_to_group.get(m_id, "Other")
        sim = float(ret_scores.get(i, 0.0))
        match_pct = max(0, min(100, int(round(sim * 100))))
        top_retrieved.append(
            RetrievedItem(
                movie_id=m_id,
                title=title,
                year=year,
                group=grp,
                poster_url=p_info.get("poster_url"),
                similarity=round(sim, 4),
                match_pct=match_pct,
            )
        )

    # Compute user_xy: average of the 2D positions of top 20 retrieved items, weighted by retrieval score
    top_20_cands = cands[:20]
    sum_x, sum_y, total_w = 0.0, 0.0, 0.0
    for i in top_20_cands:
        m_id = _data.idx_to_item.get(i)
        if m_id and m_id in _movie_id_to_xy:
            pt = _movie_id_to_xy[m_id]
            w = max(float(ret_scores.get(i, 1.0)), 0.001)
            sum_x += pt[0] * w
            sum_y += pt[1] * w
            total_w += w

    if total_w > 0:
        user_xy = [round(sum_x / total_w, 5), round(sum_y / total_w, 5)]
    else:
        user_xy = [0.5, 0.5]

    # Two-stage re-ranking for recommended_ids
    feats = _feature_eng.extract_features_for_user_candidates(u_idx, u_emb, cands)
    rank_scores = _ranker.predict(feats)
    ranked_indices = np.argsort(-rank_scores)
    ts_items = [cands[i] for i in ranked_indices[:k]]
    recommended_ids = [_data.idx_to_item[i] for i in ts_items if i in _data.idx_to_item]

    return MapUserResponse(
        user_id=user_id,
        user_xy=user_xy,
        history_ids=history_ids,
        recommended_ids=recommended_ids,
        top_retrieved=top_retrieved,
    )



# Mount built frontend if available
frontend_dist = Path("frontend/dist")
if frontend_dist.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
