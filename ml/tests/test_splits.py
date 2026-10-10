"""Unit tests for temporal splits and data leakage prevention."""
import pytest
import pandas as pd
import numpy as np
from ml.data.splits import create_temporal_splits, verify_no_leakage


def test_create_temporal_splits_ordering():
    # User 1: 5 interactions at times 10, 20, 30, 40, 50
    # User 2: 3 interactions at times 1, 2, 3 (below min 5 interactions -> filtered)
    # User 3: 5 interactions at times 100, 200, 300, 400, 500
    df = pd.DataFrame({
        "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3],
        "item_id": [10, 11, 12, 13, 14, 20, 21, 22, 30, 31, 32, 33, 34],
        "rating": [5, 4, 3, 5, 4, 5, 4, 3, 4, 4, 5, 5, 4],
        "timestamp": [10, 20, 30, 40, 50, 1, 2, 3, 100, 200, 300, 400, 500],
    })

    train, val, test = create_temporal_splits(df, min_user_interactions=5)

    assert set(train["user_id"].unique()) == {1, 3}
    assert 2 not in train["user_id"].unique()

    # User 1 test is timestamp 50 (last), val is 40 (2nd last), train is 10, 20, 30
    u1_test = test[test["user_id"] == 1].iloc[0]
    assert u1_test["timestamp"] == 50
    assert u1_test["item_id"] == 14

    u1_val = val[val["user_id"] == 1].iloc[0]
    assert u1_val["timestamp"] == 40
    assert u1_val["item_id"] == 13

    u1_train_times = train[train["user_id"] == 1]["timestamp"].tolist()
    assert u1_train_times == [10, 20, 30]


def test_verify_no_leakage_exception():
    train = pd.DataFrame({"user_id": [1], "timestamp": [50]})
    val = pd.DataFrame({"user_id": [1], "timestamp": [40]})  # Leakage! val < train
    test = pd.DataFrame({"user_id": [1], "timestamp": [60]})

    with pytest.raises(ValueError, match="train-val temporal leakages"):
        verify_no_leakage(train, val, test)
