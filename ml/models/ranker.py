"""Stage 2 Ranker using LightGBM LambdaRank."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import lightgbm as lgb
import logging
from ml.features.feature_engineer import FEATURE_NAMES

logger = logging.getLogger(__name__)


class LambdaRanker:
    """LightGBM LambdaRank model for re-ranking Stage-1 candidates."""

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.seed = seed
        self.model = None
        self.feature_names = list(FEATURE_NAMES)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group: List[int],
        feature_names: Optional[List[str]] = None,
    ) -> "LambdaRanker":
        """
        Train LambdaRank model.
        X: (num_pairs, num_features)
        y: (num_pairs,) graded relevance labels (e.g. 0 to 5)
        group: list of query group sizes (sum(group) == num_pairs)
        """
        if feature_names is not None:
            self.feature_names = feature_names

        logger.info(
            f"Fitting LambdaRanker on {len(X)} pairs across {len(group)} queries with {len(self.feature_names)} features..."
        )

        train_data = lgb.Dataset(
            X,
            label=y,
            group=group,
            feature_name=self.feature_names,
            free_raw_data=False,
        )

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10],
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "seed": self.seed,
            "verbosity": -1,
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
        )
        logger.info("LambdaRanker training complete.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict ranking scores for candidate items."""
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.predict(X)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importance dict."""
        if self.model is None:
            return {}
        raw_imp = self.model.feature_importance(importance_type="gain")
        total = max(1.0, float(np.sum(raw_imp)))
        return {
            name: float(round(raw_imp[i] / total, 4))
            for i, name in enumerate(self.feature_names)
        }

    def save(self, filepath: str) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            self.model.save_model(str(path))
            logger.info(f"Saved LambdaRanker model to {filepath}")

    def load(self, filepath: str) -> "LambdaRanker":
        path = Path(filepath)
        if path.exists():
            self.model = lgb.Booster(model_file=str(path))
            self.feature_names = self.model.feature_name()
            logger.info(f"Loaded LambdaRanker model from {filepath}")
        return self
