"""Train Two-Tower Retrieval Model and build FAISS vector index."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import time
from torch.utils.data import DataLoader, TensorDataset

from ml.config import PipelineConfig
from ml.data.dataset import MovieLensData, NUM_GENRES
from ml.models.two_tower import TwoTowerModel
from ml.models.vector_index import VectorIndex
from ml.evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_train_tensors(data: MovieLensData, last_n: int = 5) -> TensorDataset:
    """Pre-vectorize training samples into contiguous PyTorch TensorDataset for lightning-fast training."""
    pos_df = data.train_df[data.train_df["is_positive"] == 1].sort_values(
        by=["u_idx", "timestamp"]
    )

    user_grouped = pos_df.groupby("u_idx")
    u_list, i_list, hist_list = [], [], []

    for u_idx, group in user_grouped:
        items = group["i_idx"].tolist()
        for idx, item in enumerate(items):
            hist = items[max(0, idx - last_n) : idx]
            pad_len = last_n - len(hist)
            padded_hist = [data.num_items] * pad_len + hist
            u_list.append(int(u_idx))
            i_list.append(int(item))
            hist_list.append(padded_hist)

    N = len(u_list)
    logger.info(f"Created {N} training samples. Vectorizing into TensorDataset...")

    u_arr = np.array(u_list, dtype=np.int64)
    i_arr = np.array(i_list, dtype=np.int64)
    hist_arr = np.array(hist_list, dtype=np.int64)

    user_gender = np.array(
        [1 if data.user_meta[u]["gender"] == "F" else 0 for u in range(data.num_users)], dtype=np.int64
    )
    user_age = np.array(
        [min(6, max(0, data.user_meta[u]["age"] // 10)) for u in range(data.num_users)], dtype=np.int64
    )
    user_occ = np.array(
        [min(21, max(0, data.user_meta[u]["occupation"])) for u in range(data.num_users)], dtype=np.int64
    )
    user_mean_r = np.array(
        [data.user_meta[u]["mean_rating"] for u in range(data.num_users)], dtype=np.float32
    )

    item_year_norm = np.array(
        [data.item_meta[i]["year_norm"] for i in range(data.num_items)], dtype=np.float32
    )

    dataset = TensorDataset(
        torch.from_numpy(u_arr),
        torch.from_numpy(user_gender[u_arr]),
        torch.from_numpy(user_age[u_arr]),
        torch.from_numpy(user_occ[u_arr]),
        torch.from_numpy(user_mean_r[u_arr]),
        torch.from_numpy(data.user_top_genres[u_arr]),
        torch.from_numpy(hist_arr),
        torch.from_numpy(i_arr),
        torch.from_numpy(data.movie_genres_matrix[i_arr]),
        torch.from_numpy(item_year_norm[i_arr]),
        torch.from_numpy(data.item_pop_bucket[i_arr]),
    )
    return dataset


def train_two_tower(config: PipelineConfig) -> Tuple[TwoTowerModel, VectorIndex, MovieLensData]:
    """Train the two-tower model and build FAISS index."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    data = MovieLensData(
        dataset_name=config.data.dataset_name,
        raw_dir=config.data.raw_dir,
        min_user_interactions=config.data.min_user_interactions,
        positive_rating_threshold=config.data.positive_rating_threshold,
    ).load_and_process()

    train_dataset = prepare_train_tensors(data, last_n=config.data.last_n_watched)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.retrieval.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = TwoTowerModel(
        num_users=data.num_users,
        num_items=data.num_items,
        num_genres=NUM_GENRES,
        embedding_dim=config.retrieval.embedding_dim,
        temperature=config.retrieval.temperature,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.retrieval.learning_rate,
        weight_decay=config.retrieval.weight_decay,
    )

    # Compute item logQ popularity priors: log(P(i))
    item_probs = np.maximum(data.item_pos_count, 1.0) / np.sum(data.item_pos_count)
    logq_tensor = torch.from_numpy(np.log(item_probs)).float()

    logger.info(f"Training Two-Tower retrieval model for {config.retrieval.epochs} epochs...")
    model.train()
    for epoch in range(config.retrieval.epochs):
        t0 = time.time()
        total_loss = 0.0
        num_batches = 0

        for (
            b_u, b_gender, b_age, b_occ, b_mean_r, b_top_g, b_hist,
            b_i, b_genres, b_year, b_pop
        ) in train_loader:
            optimizer.zero_grad()
            user_inputs = {
                "user_ids": b_u,
                "gender_idx": b_gender,
                "age_idx": b_age,
                "occ_idx": b_occ,
                "mean_rating": b_mean_r,
                "top_genres": b_top_g,
                "history_items": b_hist,
            }
            item_inputs = {
                "item_ids": b_i,
                "genres": b_genres,
                "year_norm": b_year,
                "pop_bucket": b_pop,
            }

            batch_logq = logq_tensor[b_i] if config.retrieval.logq_correction else None
            loss, _, _ = model(user_inputs, item_inputs, item_logq=batch_logq)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        elapsed = time.time() - t0
        logger.info(f"Retrieval Epoch {epoch+1}/{config.retrieval.epochs} - Loss: {avg_loss:.4f} ({elapsed:.1f}s)")

    model.eval()

    # Precompute all catalog item embeddings
    logger.info("Computing catalog item embeddings...")
    all_item_ids = torch.arange(data.num_items, dtype=torch.long)
    all_genres = torch.from_numpy(data.movie_genres_matrix)
    all_year_norm = torch.tensor(
        [data.item_meta[i]["year_norm"] for i in range(data.num_items)], dtype=torch.float32
    )
    all_pop_bucket = torch.from_numpy(data.item_pop_bucket)

    with torch.no_grad():
        all_item_embeddings = model.item_tower(
            item_ids=all_item_ids,
            genres=all_genres,
            year_norm=all_year_norm,
            pop_bucket=all_pop_bucket,
        ).cpu().numpy()

    # Build FAISS index
    index = VectorIndex(
        dim=config.retrieval.embedding_dim,
        index_type=config.retrieval.index_type,
    ).build(all_item_embeddings)

    # Save artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), artifacts_dir / "two_tower.pt")
    index.save(str(artifacts_dir / "faiss_index.bin"))
    np.save(str(artifacts_dir / "item_embeddings.npy"), all_item_embeddings)
    logger.info("Saved two_tower.pt, faiss_index.bin, and item_embeddings.npy to artifacts/")

    return model, index, data


def evaluate_retrieval(model: TwoTowerModel, index: VectorIndex, data: MovieLensData, config: PipelineConfig):
    """Evaluate two-tower candidate generation."""
    evaluator = Evaluator(
        test_df=data.test_df,
        total_items=data.num_items,
        item_popularity=data.item_popularity,
        genre_matrix=data.movie_genres_matrix,
        ks=config.eval.ks,
        ranking_k=config.eval.ranking_k,
    )

    # Function to get user embedding and retrieve top-K via FAISS
    def retrieve_for_user(u_idx: int, k: int = 100, exclude: set = None) -> List[int]:
        u_meta = data.user_meta[u_idx]
        hist = data.user_history_i_idx[u_idx][-config.data.last_n_watched :]
        pad_len = config.data.last_n_watched - len(hist)
        padded_hist = [data.num_items] * pad_len + hist

        with torch.no_grad():
            u_emb = model.user_tower(
                user_ids=torch.tensor([u_idx], dtype=torch.long),
                gender_idx=torch.tensor([1 if u_meta["gender"] == "F" else 0], dtype=torch.long),
                age_idx=torch.tensor([min(6, max(0, u_meta["age"] // 10))], dtype=torch.long),
                occ_idx=torch.tensor([min(21, max(0, u_meta["occupation"]))], dtype=torch.long),
                mean_rating=torch.tensor([u_meta["mean_rating"]], dtype=torch.float32),
                top_genres=torch.from_numpy(data.user_top_genres[u_idx: u_idx + 1]),
                history_items=torch.tensor([padded_hist], dtype=torch.long),
            ).cpu().numpy()[0]

        # Query top-K from FAISS with candidate cushion to handle exclusions
        query_k = min(data.num_items, k + len(exclude or set()) + 10)
        _, indices = index.search(u_emb, k=query_k)

        recs = []
        ex = exclude or set()
        for idx in indices:
            if int(idx) >= 0 and int(idx) not in ex:
                recs.append(int(idx))
            if len(recs) >= k:
                break
        return recs

    logger.info("Evaluating Two-Tower + FAISS Retrieval...")
    retrieval_metrics = evaluator.evaluate_model(
        model_name="Two-Tower Retrieval",
        rec_fn=retrieve_for_user,
        user_histories=data.user_history_i_idx,
    )
    logger.info(f"Two-Tower Retrieval Results: {retrieval_metrics}")

    # Update metrics.json
    results_dir = Path(config.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"

    metrics_db = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_db = json.load(f)

    metrics_db["Two-Tower Retrieval"] = retrieval_metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_db, f, indent=2)

    logger.info(f"Updated metrics saved to {metrics_path}")
    return retrieval_metrics


if __name__ == "__main__":
    cfg = PipelineConfig.load()
    model, index, data = train_two_tower(cfg)
    evaluate_retrieval(model, index, data, cfg)
