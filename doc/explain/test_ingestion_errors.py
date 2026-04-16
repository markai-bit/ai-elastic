

from __future__ import annotations

import json
import logging

import pytest


# ---------------------------------------------------------------------------
# Validation -> 422
# ---------------------------------------------------------------------------


def test_empty_chunks_rejected_with_422(app_client):
    resp = app_client.post("/ingestion", json={"chunks": []})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error_code"] == "request_validation_failed"
    assert "correlation_id" in body
    assert body["correlation_id"]


def test_missing_required_metadata_rejected_with_422(app_client):
    payload = {
        "chunks": [{"content": "x", "metadata": {"doc_id": "d"}}]  # no doc_title
    }
    resp = app_client.post("/ingestion", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["error_code"] == "request_validation_failed"
    # The Pydantic error detail should mention the missing field.
    blob = json.dumps(body["details"])
    assert "doc_title" in blob


def test_unknown_metadata_field_rejected_with_422(app_client):
    payload = {
        "chunks": [{
            "content": "x",
            "metadata": {"doc_id": "d", "doc_title": "t", "color": "red"},
        }]
    }
    resp = app_client.post("/ingestion", json=payload)
    assert resp.status_code == 422
    blob = json.dumps(resp.json()["details"])
    assert "color" in blob


def test_empty_content_rejected_with_422(app_client):
    payload = {
        "chunks": [{"content": "", "metadata": {"doc_id": "d", "doc_title": "t"}}]
    }
    resp = app_client.post("/ingestion", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# ES failures -> 502
# ---------------------------------------------------------------------------


def test_es_bulk_failure_returns_502(app_client, fake_es):
    app_client.post("/ingestion/index")
    fake_es.fail_on_bulk = True
    payload = {"chunks": [{"content": "x", "metadata": {"doc_id": "d", "doc_title": "t"}}]}
    resp = app_client.post("/ingestion", json=payload)
    assert resp.status_code == 502
    body = resp.json()
    assert body["error_code"] == "index_operation_failed"
    assert body["correlation_id"]
    assert "Bulk ingest failed" in body["message"]


# ---------------------------------------------------------------------------
# Correlation ID middleware
# ---------------------------------------------------------------------------


def test_correlation_id_echoed_when_supplied(app_client):
    app_client.post("/ingestion/index")
    cid = "test-correlation-12345"
    payload = {"chunks": [{"content": "x", "metadata": {"doc_id": "d", "doc_title": "t"}}]}
    resp = app_client.post("/ingestion", json=payload, headers={"X-Correlation-ID": cid})
    assert resp.status_code == 200
    assert resp.headers["X-Correlation-ID"] == cid


def test_correlation_id_minted_when_missing(app_client):
    app_client.post("/ingestion/index")
    payload = {"chunks": [{"content": "y", "metadata": {"doc_id": "d", "doc_title": "t"}}]}
    resp = app_client.post("/ingestion", json=payload)
    assert resp.status_code == 200
    cid = resp.headers.get("X-Correlation-ID")
    assert cid
    assert len(cid) >= 16  # uuid hex is 32 chars; allow some headroom


def test_correlation_id_present_in_error_response(app_client):
    """Errors carry the correlation ID so a 5xx is traceable to logs."""
    cid = "trace-this-one"
    resp = app_client.post(
        "/ingestion",
        json={"chunks": []},
        headers={"X-Correlation-ID": cid},
    )
    assert resp.status_code == 422
    assert resp.json()["correlation_id"] == cid
    assert resp.headers["X-Correlation-ID"] == cid


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------


def test_logs_include_correlation_id(app_client, caplog):
    """The JSON formatter is bypassed by caplog (it captures records, not output),
    but we can verify the correlation ID was in context when log records were emitted."""
    from app.logging_config import get_correlation_id

    captured: list[str | None] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            captured.append(get_correlation_id())

    handler = _Capture(level=logging.INFO)
    logging.getLogger().addHandler(handler)
    try:
        cid = "log-cid-abc"
        app_client.post("/ingestion/index")  # warms up
        captured.clear()
        app_client.post(
            "/ingestion",
            json={"chunks": [{"content": "z", "metadata": {"doc_id": "d", "doc_title": "t"}}]},
            headers={"X-Correlation-ID": cid},
        )
    finally:
        logging.getLogger().removeHandler(handler)

    # At least one log emitted during the request should have seen our cid.
    assert cid in captured, f"correlation id not propagated to logs (saw {captured!r})"


def test_health_endpoint_does_not_require_es(app_client):
    """A liveness check should never depend on a downstream being healthy."""
    resp = app_client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
