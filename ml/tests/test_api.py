"""Unit tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    # Use TestClient with context manager to trigger startup events
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["artifacts_loaded"] is True
    assert data["num_users"] > 0
    assert data["num_items"] > 0


def test_metrics_endpoint(client):
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "Two-Tower Retrieval" in data
    assert "Two-Stage (Two-Tower + Ranker)" in data


def test_get_user_valid(client):
    res = client.get("/api/users/1")
    assert res.status_code == 200
    data = res.json()
    assert data["user_id"] == 1
    assert "top_genres" in data
    assert "watch_history" in data
    assert len(data["watch_history"]) > 0


def test_get_user_not_found(client):
    res = client.get("/api/users/99999999")
    assert res.status_code == 404


def test_recommend_retrieval_only(client):
    res = client.get("/api/recommend/1?n=5&mode=retrieval_only")
    assert res.status_code == 200
    data = res.json()
    assert data["user_id"] == 1
    assert len(data["recommendations"]) == 5
    assert data["retrieval_ms"] > 0
    # ranking_ms should be 0.0 for retrieval_only
    assert data["ranking_ms"] == 0.0
    for rec in data["recommendations"]:
        assert "item_id" in rec
        assert "title" in rec
        assert "why" in rec


def test_recommend_two_stage(client):
    res = client.get("/api/recommend/1?n=5&mode=two_stage")
    assert res.status_code == 200
    data = res.json()
    assert data["user_id"] == 1
    assert len(data["recommendations"]) == 5
    assert data["retrieval_ms"] > 0
    assert data["ranking_ms"] > 0
    for rec in data["recommendations"]:
        assert "item_id" in rec
        assert "title" in rec
        assert "why" in rec


def test_similar_items(client):
    res = client.get("/api/similar/1?n=5")
    assert res.status_code == 200
    data = res.json()
    assert data["item_id"] == 1
    assert len(data["similar_items"]) == 5
    assert all(item["item_id"] != 1 for item in data["similar_items"])


def test_similar_items_not_found(client):
    res = client.get("/api/similar/99999999?n=5")
    assert res.status_code == 404


def test_search(client):
    res = client.get("/api/search?q=Toy")
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "Toy"
    assert data["total"] > 0
    assert any("Toy Story" in item["title"] for item in data["results"])
