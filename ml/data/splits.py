"""Temporal train/val/test splitting per user to prevent data leakage."""
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_temporal_splits(
    ratings_df: pd.DataFrame,
    min_user_interactions: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform a strict temporal leave-last-out split per user:
    - Last interaction -> test
    - Second-to-last interaction -> validation
    - All earlier interactions -> train

    Validates that:
    1. Users have at least `min_user_interactions`
    2. No temporal leakage: train < val <= test per user
    """
    logger.info("Starting temporal split...")
    df = ratings_df.sort_values(by=["user_id", "timestamp"]).copy()

    # Filter users with sufficient interactions
    user_counts = df.groupby("user_id")["timestamp"].count()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    df = df[df["user_id"].isin(valid_users)].copy()

    # Rank interactions per user in ascending order (0 = oldest, N-1 = newest)
    df["rank_desc"] = df.groupby("user_id").cumcount(ascending=False)

    test_df = df[df["rank_desc"] == 0].drop(columns=["rank_desc"]).reset_index(drop=True)
    val_df = df[df["rank_desc"] == 1].drop(columns=["rank_desc"]).reset_index(drop=True)
    train_df = df[df["rank_desc"] >= 2].drop(columns=["rank_desc"]).reset_index(drop=True)

    # Sanity checks
    verify_no_leakage(train_df, val_df, test_df)

    logger.info(
        f"Splits created: Train={len(train_df)} ({len(train_df['user_id'].unique())} users), "
        f"Val={len(val_df)} ({len(val_df['user_id'].unique())} users), "
        f"Test={len(test_df)} ({len(test_df['user_id'].unique())} users)"
    )
    return train_df, val_df, test_df


def verify_no_leakage(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Verify that there is zero temporal leakage between train, val, and test."""
    train_max_time = train_df.groupby("user_id")["timestamp"].max().rename("train_max")
    val_time = val_df.set_index("user_id")["timestamp"].rename("val_time")
    test_time = test_df.set_index("user_id")["timestamp"].rename("test_time")

    check_df = pd.concat([train_max_time, val_time, test_time], axis=1).dropna()

    leakage_train_val = (check_df["train_max"] > check_df["val_time"]).sum()
    leakage_val_test = (check_df["val_time"] > check_df["test_time"]).sum()

    if leakage_train_val > 0:
        raise ValueError(f"Found {leakage_train_val} train-val temporal leakages!")
    if leakage_val_test > 0:
        raise ValueError(f"Found {leakage_val_test} val-test temporal leakages!")

    logger.info("Verification passed: Zero temporal leakage detected across all splits.")
