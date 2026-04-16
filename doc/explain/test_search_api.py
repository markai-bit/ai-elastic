

from __future__ import annotations

import pytest

from app.search import build_hybrid_query


# ---------------------------------------------------------------------------
# Pure-function check on query construction
# ---------------------------------------------------------------------------


def test_build_hybrid_query_basic_shape():
    body = build_hybrid_query(
        query_text="capital requirements",
        query_vector=[0.1, 0.2, 0.3],
        top_k=5,
        filters={},
        bm25_boost=0.5,
        knn_boost=0.5,
        knn_num_candidates=100,
    )
    assert body["size"] == 5
    assert body["query"]["bool"]["must"][0]["match"]["content"]["query"] == "capital requirements"
    assert body["query"]["bool"]["must"][0]["match"]["content"]["boost"] == 0.5
    assert body["knn"]["field"] == "embedding"
    assert body["knn"]["k"] == 5
    assert body["knn"]["boost"] == 0.5
    # num_candidates should be at least 10x k.
    assert body["knn"]["num_candidates"] >= 50


def test_build_hybrid_query_with_filters():
    body = build_hybrid_query(
        query_text="x",
        query_vector=[0.0],
        top_k=3,
        filters={"jurisdiction": "UK", "regulation_name": ["CRR", "CRD"]},
        bm25_boost=0.5,
        knn_boost=0.5,
        knn_num_candidates=100,
    )
    bm25_filters = body["query"]["bool"]["filter"]
    knn_filters = body["knn"]["filter"]
    assert {"term": {"metadata.jurisdiction": "UK"}} in bm25_filters
    assert {"terms": {"metadata.regulation_name": ["CRR", "CRD"]}} in bm25_filters
    # Filters applied to BOTH legs so they actually narrow the result set.
    assert bm25_filters == knn_filters


def test_build_hybrid_query_num_candidates_floor():
    body = build_hybrid_query(
        query_text="x", query_vector=[0.0], top_k=50,
        filters={}, bm25_boost=0.5, knn_boost=0.5, knn_num_candidates=10,
    )
    # Even with a low configured floor, num_candidates must be at least 10*top_k.
    assert body["knn"]["num_candidates"] == 500


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


def _seed(app_client, contents):
    app_client.post("/ingestion/index")
    app_client.post("/ingestion", json={
        "chunks": [
            {"content": c, "metadata": {"doc_id": f"d-{i}", "doc_title": "t", "jurisdiction": "UK"}}
            for i, c in enumerate(contents)
        ]
    })


def test_search_returns_hits(app_client):
    _seed(app_client, ["alpha sentence", "beta sentence", "gamma sentence"])
    resp = app_client.post("/search", json={"query": "anything", "top_k": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "anything"
    assert body["total_hits"] >= 1
    # Each hit conforms to the schema.
    for hit in body["hits"]:
        assert "chunk_id" in hit
        assert "score" in hit
        assert "content" in hit
        assert "metadata" in hit


def test_search_respects_top_k(app_client):
    _seed(app_client, [f"sentence {i}" for i in range(20)])
    resp = app_client.post("/search", json={"query": "q", "top_k": 5})
    assert resp.status_code == 200
    assert len(resp.json()["hits"]) <= 5


def test_search_with_metadata_filter(app_client):
    _seed(app_client, ["uk content", "uk content 2"])
    resp = app_client.post(
        "/search",
        json={"query": "content", "top_k": 10, "filters": {"jurisdiction": "UK"}},
    )
    assert resp.status_code == 200
    # Fake ES doesn't actually apply filters, but the request shape was accepted
    # and the body validates -- we test filter translation in the pure-function tests.
    assert resp.json()["query"] == "content"


def test_search_default_top_k(app_client):
    _seed(app_client, ["one"])
    resp = app_client.post("/search", json={"query": "anything"})
    assert resp.status_code == 200
