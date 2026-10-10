"""Configuration module for recommendation system pipeline."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import yaml


@dataclass
class DataConfig:
    dataset_name: str = "ml-1m"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    positive_rating_threshold: float = 4.0
    min_user_interactions: int = 5
    last_n_watched: int = 5


@dataclass
class RetrievalConfig:
    embedding_dim: int = 64
    user_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    item_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    learning_rate: float = 0.001
    batch_size: int = 512
    epochs: int = 15
    temperature: float = 0.07
    weight_decay: float = 1e-5
    logq_correction: bool = True
    top_k_candidates: int = 200
    index_type: str = "IVF_FLAT"


@dataclass
class RankerConfig:
    model_type: str = "lightgbm"
    n_estimators: int = 150
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 20
    negative_ratio: int = 10


@dataclass
class EvalConfig:
    ks: List[int] = field(default_factory=lambda: [10, 50, 100])
    ranking_k: int = 10
    results_dir: str = "results"


@dataclass
class PipelineConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "PipelineConfig":
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "default.yaml")
        
        path = Path(config_path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}

        data_cfg = DataConfig(**raw_cfg.get("data", {}))
        retrieval_cfg = RetrievalConfig(**raw_cfg.get("retrieval", {}))
        ranker_cfg = RankerConfig(**raw_cfg.get("ranker", {}))
        eval_cfg = EvalConfig(**raw_cfg.get("eval", {}))
        seed = raw_cfg.get("seed", 42)

        return cls(
            seed=seed,
            data=data_cfg,
            retrieval=retrieval_cfg,
            ranker=ranker_cfg,
            eval=eval_cfg,
        )
