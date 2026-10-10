"""Unit tests for Two-Tower model architecture."""
import pytest
import numpy as np
import torch
from ml.models.two_tower import UserTower, ItemTower, TwoTowerModel

NUM_USERS = 100
NUM_ITEMS = 200
NUM_GENRES = 19
EMB_DIM = 64
BATCH = 8


@pytest.fixture
def user_tower():
    return UserTower(num_users=NUM_USERS, num_items=NUM_ITEMS, num_genres=NUM_GENRES, embedding_dim=EMB_DIM)


@pytest.fixture
def item_tower():
    return ItemTower(num_items=NUM_ITEMS, num_genres=NUM_GENRES, embedding_dim=EMB_DIM)


@pytest.fixture
def two_tower():
    return TwoTowerModel(num_users=NUM_USERS, num_items=NUM_ITEMS, num_genres=NUM_GENRES, embedding_dim=EMB_DIM)


def _user_inputs(batch: int = BATCH) -> dict:
    return {
        "user_ids": torch.randint(0, NUM_USERS, (batch,)),
        "gender_idx": torch.randint(0, 2, (batch,)),
        "age_idx": torch.randint(0, 7, (batch,)),
        "occ_idx": torch.randint(0, 21, (batch,)),
        "mean_rating": torch.rand(batch) * 5.0,
        "top_genres": torch.rand(batch, NUM_GENRES),
        "history_items": torch.randint(0, NUM_ITEMS, (batch, 5)),
    }


def _item_inputs(batch: int = BATCH, num_items: int = NUM_ITEMS) -> dict:
    return {
        "item_ids": torch.randint(0, num_items, (batch,)),
        "genres": torch.rand(batch, NUM_GENRES),
        "year_norm": torch.rand(batch),
        "pop_bucket": torch.randint(0, 10, (batch,)),
    }


def test_user_tower_output_shape(user_tower):
    """UserTower must produce L2-normalized (B, EMB_DIM) embeddings."""
    user_tower.eval()
    with torch.no_grad():
        out = user_tower(**_user_inputs())
    assert out.shape == (BATCH, EMB_DIM)


def test_user_tower_l2_normalized(user_tower):
    """Embeddings from UserTower must be L2-normalized (norm ≈ 1)."""
    user_tower.eval()
    with torch.no_grad():
        out = user_tower(**_user_inputs())
    norms = torch.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)


def test_item_tower_output_shape(item_tower):
    """ItemTower must produce L2-normalized (B, EMB_DIM) embeddings."""
    item_tower.eval()
    with torch.no_grad():
        out = item_tower(**_item_inputs())
    assert out.shape == (BATCH, EMB_DIM)


def test_item_tower_l2_normalized(item_tower):
    """Embeddings from ItemTower must be L2-normalized."""
    item_tower.eval()
    with torch.no_grad():
        out = item_tower(**_item_inputs())
    norms = torch.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones(BATCH), atol=1e-5)


def test_two_tower_loss_shape(two_tower):
    """Forward pass must return a scalar loss."""
    loss, u_emb, i_emb = two_tower(_user_inputs(), _item_inputs())
    assert loss.shape == ()
    assert loss.item() > 0


def test_two_tower_logq_correction(two_tower):
    """LogQ correction changes loss value but not shape."""
    u_in = _user_inputs()
    i_in = _item_inputs()
    logq = torch.rand(BATCH)

    loss_plain, _, _ = two_tower(u_in, i_in)
    loss_logq, _, _ = two_tower(u_in, i_in, item_logq=logq)

    assert loss_plain.shape == ()
    assert loss_logq.shape == ()
    # They should differ (correction should change logits)
    assert abs(loss_plain.item() - loss_logq.item()) > 1e-6


def test_two_tower_embedding_dims(two_tower):
    """u_emb and i_emb from forward must be (B, EMB_DIM)."""
    _, u_emb, i_emb = two_tower(_user_inputs(), _item_inputs())
    assert u_emb.shape == (BATCH, EMB_DIM)
    assert i_emb.shape == (BATCH, EMB_DIM)


def test_two_tower_padding_index():
    """Padding index (num_items) must not cause index errors in UserTower history."""
    model = TwoTowerModel(num_users=10, num_items=50, embedding_dim=16)
    u_in = _user_inputs(batch=2)
    u_in["user_ids"] = torch.tensor([0, 1])
    u_in["history_items"] = torch.tensor([[50, 50, 0, 1, 2], [50, 3, 4, 5, 6]])  # 50 = padding idx

    i_in = _item_inputs(batch=2, num_items=50)
    loss, _, _ = model(u_in, i_in)
    assert loss.item() > 0
