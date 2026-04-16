

from __future__ import annotations

import json

import pytest

from app.embeddings import Embedder
from app.exceptions import EmbeddingError


# ---------------------------------------------------------------------------
# Validation -> 422
# ---------------------------------------------------------------------------


def test_empty_query_rejected(app_client):
    resp = app_client.post("/search", json={"query": "", "top_k": 5})
    assert resp.status_code == 422
    assert resp.json()["error_code"] == "request_validation_failed"


def test_missing_query_rejected(app_client):
    resp = app_client.post("/search", json={"top_k": 5})
    assert resp.status_code == 422


def test_unknown_filter_field_rejected(app_client):
    resp = app_client.post("/search", json={"query": "q", "filters": {"banana": "yellow"}})
    assert resp.status_code == 422
    blob = json.dumps(resp.json()["details"])
    assert "banana" in blob


def test_top_k_too_large_rejected(app_client):
    resp = app_client.post("/search", json={"query": "q", "top_k": 101})
    assert resp.status_code == 422


def test_top_k_zero_rejected(app_client):
    resp = app_client.post("/search", json={"query": "q", "top_k": 0})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Index not found -> 404
# ---------------------------------------------------------------------------


def test_search_against_missing_index_returns_404(app_client):
    # Note: we deliberately do NOT create the index here.
    resp = app_client.post("/search", json={"query": "anything"})
    assert resp.status_code == 404
    body = resp.json()
    assert body["error_code"] == "index_not_found"
    assert body["correlation_id"]


# ---------------------------------------------------------------------------
# ES failure -> 502
# ---------------------------------------------------------------------------


def test_search_es_failure_returns_502(app_client, fake_es):
    app_client.post("/ingestion/index")
    fake_es.fail_on_search = True
    resp = app_client.post("/search", json={"query": "anything"})
    assert resp.status_code == 502
    body = resp.json()
    assert body["error_code"] == "search_failed"
    assert body["correlation_id"]


# ---------------------------------------------------------------------------
# Embedder failure -> 502
# ---------------------------------------------------------------------------


class _BrokenEmbedder:
    @property
    def dimension(self) -> int:
        return 8

    async def embed(self, texts):
        raise EmbeddingError("router unreachable", details={"endpoint": "http://x"})


def test_search_embedder_failure_returns_502(app_client, fake_es):
    """When the embedder errors, the global handler should map it to 502."""
    from main import app
    from app.routers.search_router import get_embedder

    app_client.post("/ingestion/index")
    app.dependency_overrides[get_embedder] = lambda: _BrokenEmbedder()
    try:
        resp = app_client.post("/search", json={"query": "anything"})
    finally:
        # Don't leak the override into other tests in the same session.
        app.dependency_overrides.pop(get_embedder, None)

    assert resp.status_code == 502
    body = resp.json()
    assert body["error_code"] == "embedding_service_error"
    assert "router unreachable" in body["message"]
    assert body["details"].get("endpoint") == "http://x"
    assert body["correlation_id"]


# ---------------------------------------------------------------------------
# Correlation ID propagation in errors
# ---------------------------------------------------------------------------


def test_search_validation_error_carries_correlation_id(app_client):
    cid = "search-trace-xyz"
    resp = app_client.post(
        "/search",
        json={"query": ""},
        headers={"X-Correlation-ID": cid},
    )
    assert resp.status_code == 422
    assert resp.json()["correlation_id"] == cid
    assert resp.headers["X-Correlation-ID"] == cid


def test_search_404_carries_correlation_id(app_client):
    cid = "search-404-trace"
    resp = app_client.post(
        "/search",
        json={"query": "x"},
        headers={"X-Correlation-ID": cid},
    )
    assert resp.status_code == 404
    assert resp.json()["correlation_id"] == cid
