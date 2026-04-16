
from __future__ import annotations

import hashlib
from typing import Any

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from app.embeddings import Embedder
from app.logging_config import configure_logging


# ---------------------------------------------------------------------------
# Logging: configure once, very quietly
# ---------------------------------------------------------------------------
configure_logging(level="WARNING")


# ---------------------------------------------------------------------------
# Fake embedder
# ---------------------------------------------------------------------------


class FakeEmbedder:
    """Deterministic embedder: hashes text into a fixed-length float vector."""

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            # Spread bytes across the requested dimension.
            vec = [
                (digest[i % len(digest)] / 255.0) for i in range(self._dimension)
            ]
            out.append(vec)
        return out


@pytest.fixture
def fake_embedder() -> Embedder:
    return FakeEmbedder()


# ---------------------------------------------------------------------------
# Fake async Elasticsearch
# ---------------------------------------------------------------------------


class _Indices:
    def __init__(self, parent: "FakeAsyncES") -> None:
        self._p = parent

    async def exists(self, index: str) -> bool:
        return index in self._p.indices_data

    async def create(self, index: str, mappings: dict[str, Any] | None = None) -> dict:
        if index in self._p.indices_data:
            raise RuntimeError(f"index {index} already exists")
        self._p.indices_data[index] = {"mappings": mappings or {}, "docs": {}}
        return {"acknowledged": True}

    async def delete(self, index: str) -> dict:
        self._p.indices_data.pop(index, None)
        return {"acknowledged": True}

    async def refresh(self, index: str) -> dict:
        return {"acknowledged": True}


class FakeAsyncES:
    """Minimal async ES stub. Stores docs in a dict keyed by chunk_id."""

    def __init__(self) -> None:
        self.indices_data: dict[str, dict] = {}
        self.indices = _Indices(self)
        # Hooks tests can flip to simulate failures.
        self.fail_on_search = False
        self.fail_on_bulk = False

    def _store(self, index: str) -> dict:
        return self.indices_data[index]["docs"]

    async def search(
        self,
        index: str,
        query: dict | None = None,
        body: dict | None = None,
        size: int = 10,
        _source: bool = True,
        **_kwargs,
    ) -> dict:
        if self.fail_on_search:
            raise RuntimeError("simulated ES search failure")
        if index not in self.indices_data:
            from elasticsearch import NotFoundError
            raise NotFoundError("index_not_found", {}, {})

        docs = list(self._store(index).values())

        # Tests use either a top-level `query` (term match for dedup) or
        # `body` (full hybrid query). We support both, narrowly.
        if query and "term" in query:
            field, value = next(iter(query["term"].items()))
            hits = [d for d in docs if d.get(field) == value][:size]
        elif body and "query" in body:
            # Treat as match-all for the fake; real scoring is irrelevant.
            hits = docs[: body.get("size", size)]
        else:
            hits = docs[:size]

        return {
            "hits": {
                "total": {"value": len(hits)},
                "hits": [
                    {"_id": d["chunk_id"], "_score": 1.0, "_source": d}
                    for d in hits
                ],
            }
        }

    async def index(self, *, index: str, id: str, document: dict) -> dict:
        self._store(index)[id] = document
        return {"_id": id, "result": "created"}

    async def bulk(self, operations: list[dict] | None = None, **_kwargs) -> dict:
        if self.fail_on_bulk:
            raise RuntimeError("simulated ES bulk failure")
        ops = operations or _kwargs.get("body") or []
        items = []
        # operations is a flat list alternating action / doc.
        it = iter(ops)
        for action in it:
            doc = next(it)
            meta = action.get("index") or action.get("create") or {}
            idx = meta["_index"]
            doc_id = meta.get("_id") or doc.get("chunk_id")
            self._store(idx)[doc_id] = doc
            items.append({"index": {"_id": doc_id, "status": 201}})
        return {"errors": False, "items": items}

    async def close(self) -> None:
        return None


@pytest_asyncio.fixture
async def fake_es() -> FakeAsyncES:
    es = FakeAsyncES()
    return es


# ---------------------------------------------------------------------------
# FastAPI test client with overridden dependencies
# ---------------------------------------------------------------------------


@pytest.fixture
def app_client(fake_es, fake_embedder) -> TestClient:
    """TestClient with ES + embedder swapped for fakes."""
    from main import app
    from app.es_client import get_es_client
    from app.routers.ingestion_router import get_embedder as get_ing_embedder
    from app.routers.search_router import get_embedder as get_search_embedder

    async def _es_override():
        return fake_es

    def _embedder_override():
        return fake_embedder

    app.dependency_overrides[get_es_client] = _es_override
    app.dependency_overrides[get_ing_embedder] = _embedder_override
    app.dependency_overrides[get_search_embedder] = _embedder_override

    with TestClient(app) as client:
        yield client

    app.dependency_overrides.clear()
