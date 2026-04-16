

from __future__ import annotations

import pytest


def test_create_index_endpoint(app_client):
    resp = app_client.post("/ingestion/index")
    assert resp.status_code == 201
    body = resp.json()
    assert body["index"] == "poc_policy_index"  # default from settings
    assert body["created"] is True

    # Idempotent: second call doesn't error and reports created=False.
    resp2 = app_client.post("/ingestion/index")
    assert resp2.status_code == 201
    assert resp2.json()["created"] is False


def test_create_index_with_recreate(app_client):
    app_client.post("/ingestion/index")
    resp = app_client.post("/ingestion/index?recreate=true")
    assert resp.status_code == 201
    body = resp.json()
    assert body["created"] is True
    assert body["recreated"] is True


def test_get_schema_endpoint(app_client):
    resp = app_client.get("/ingestion/schema")
    assert resp.status_code == 200
    body = resp.json()
    names = {f["name"] for f in body["metadata_fields"]}
    # Spot-check a couple of fields we know are in the schema.
    assert "doc_id" in names
    assert "doc_title" in names
    assert "regulation_name" in names
    # Required-ness is exposed.
    doc_id = next(f for f in body["metadata_fields"] if f["name"] == "doc_id")
    assert doc_id["required"] is True


def test_ingest_basic_request(app_client):
    app_client.post("/ingestion/index")
    payload = {
        "chunks": [
            {
                "content": "The CRR sets capital requirements.",
                "metadata": {"doc_id": "crr-1", "doc_title": "CRR Overview", "page_number": 1},
            }
        ]
    }
    resp = app_client.post("/ingestion", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["ingested"] == 1
    assert body["skipped"] == 0
    assert body["outcomes"][0]["status"] == "ingested"
    assert body["outcomes"][0]["chunk_id"]


def test_ingest_returns_outcomes_per_chunk(app_client):
    app_client.post("/ingestion/index")
    payload = {
        "chunks": [
            {"content": "Para A", "metadata": {"doc_id": "d", "doc_title": "t"}},
            {"content": "Para B", "metadata": {"doc_id": "d", "doc_title": "t"}},
            {"content": "Para C", "metadata": {"doc_id": "d", "doc_title": "t"}},
        ]
    }
    resp = app_client.post("/ingestion", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert body["ingested"] == 3
    assert len(body["outcomes"]) == 3
    assert all(o["status"] == "ingested" for o in body["outcomes"])
    chunk_ids = {o["chunk_id"] for o in body["outcomes"]}
    assert len(chunk_ids) == 3, "each chunk should get a unique id"


def test_dedup_via_api(app_client):
    """End-to-end check that hash dedup works through the HTTP layer."""
    app_client.post("/ingestion/index")
    payload = {
        "chunks": [{"content": "One sentence.", "metadata": {"doc_id": "d", "doc_title": "t"}}]
    }
    first = app_client.post("/ingestion", json=payload).json()
    assert first["ingested"] == 1

    second = app_client.post("/ingestion", json=payload).json()
    assert second["ingested"] == 0
    assert second["skipped"] == 1
    assert second["outcomes"][0]["status"] == "skipped_duplicate"


def test_force_on_metadata_change_via_api(app_client):
    app_client.post("/ingestion/index")
    base = {"content": "Common sentence.", "metadata": {"doc_id": "d", "doc_title": "t", "page_number": 1}}
    app_client.post("/ingestion", json={"chunks": [base]})

    variant = {
        "content": "Common sentence.",
        "metadata": {"doc_id": "d", "doc_title": "t", "page_number": 99},
    }
    # Without flag: skipped.
    r1 = app_client.post("/ingestion", json={"chunks": [variant]}).json()
    assert r1["skipped"] == 1

    # With flag: ingested as new.
    r2 = app_client.post(
        "/ingestion",
        json={"chunks": [variant], "force_on_metadata_change": True},
    ).json()
    assert r2["ingested"] == 1
    assert r2["outcomes"][0]["status"] == "ingested_new_metadata"
