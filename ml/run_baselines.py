"""Run baseline models (Popularity and BPR-MF) and record metrics."""
import json
import logging
from pathlib import Path
import numpy as np

from ml.config import PipelineConfig
from ml.data.dataset import MovieLensData
from ml.models.baselines import PopularityModel, train_bpr_model
from ml.evaluate import Evaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_baselines(config: PipelineConfig):
    np.random.seed(config.seed)

    # 1. Load data
    logger.info("Loading MovieLens dataset...")
    data = MovieLensData(
        dataset_name=config.data.dataset_name,
        raw_dir=config.data.raw_dir,
        min_user_interactions=config.data.min_user_interactions,
        positive_rating_threshold=config.data.positive_rating_threshold,
    ).load_and_process()

    logger.info(
        f"Dataset loaded: {data.num_users} users, {data.num_items} items. "
        f"Train={len(data.train_df)}, Val={len(data.val_df)}, Test={len(data.test_df)}"
    )

    # 2. Setup Evaluator
    evaluator = Evaluator(
        test_df=data.test_df,
        total_items=data.num_items,
        item_popularity=data.item_popularity,
        genre_matrix=data.movie_genres_matrix,
        ks=config.eval.ks,
        ranking_k=config.eval.ranking_k,
    )

    results_dir = Path(config.eval.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / "metrics.json"

    metrics_db = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_db = json.load(f)
        except Exception:
            metrics_db = {}

    # 3. Evaluate Popularity Baseline
    logger.info("Evaluating Popularity Baseline...")
    pop_model = PopularityModel(data.item_popularity)
    pop_metrics = evaluator.evaluate_model(
        model_name="Popularity",
        rec_fn=lambda u, k, exclude: pop_model.recommend(u, k=k, exclude_items=exclude),
        user_histories=data.user_history_i_idx,
    )
    metrics_db["Popularity"] = pop_metrics
    logger.info(f"Popularity Results: {pop_metrics}")

    # 4. Train & Evaluate BPR Matrix Factorization
    logger.info("Training BPR Matrix Factorization Baseline...")
    # Extract positive interactions in train
    train_pos_df = data.train_df[data.train_df["is_positive"] == 1]
    train_user_pos = {u: [] for u in range(data.num_users)}
    for _, row in train_pos_df.iterrows():
        train_user_pos[int(row["u_idx"])].append(int(row["i_idx"]))

    mf_model = train_bpr_model(
        num_users=data.num_users,
        num_items=data.num_items,
        train_user_pos_items=train_user_pos,
        embedding_dim=config.retrieval.embedding_dim,
        epochs=8,
        lr=0.01,
        weight_decay=1e-5,
        batch_size=2048,
        seed=config.seed,
    )

    logger.info("Evaluating BPR-MF Baseline...")
    mf_metrics = evaluator.evaluate_model(
        model_name="Matrix Factorization (BPR)",
        rec_fn=lambda u, k, exclude: mf_model.recommend(u, k=k, exclude_items=exclude),
        user_histories=data.user_history_i_idx,
    )
    metrics_db["Matrix Factorization (BPR)"] = mf_metrics
    logger.info(f"BPR-MF Results: {mf_metrics}")

    # Save metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_db, f, indent=2)

    logger.info(f"Metrics saved to {metrics_path}")
    return metrics_db


if __name__ == "__main__":
    config = PipelineConfig.load()
    run_baselines(config)
