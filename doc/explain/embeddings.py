

from __future__ import annotations

import os
from typing import Protocol

import httpx

from app.exceptions import EmbeddingError
from app.logging_config import get_logger

log = get_logger(__name__)


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def dimension(self) -> int: ...


class RouterEmbedder:
    """Calls the existing /ai-router-service/embedding/batch endpoint."""

    def __init__(
        self,
        router_endpoint: str | None = None,
        model: str | None = None,
        dimension: int = 1536,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._endpoint = (
            router_endpoint
            or os.getenv("ROUTER_SERVICE_ENDPOINT", "http://127.0.0.1:8002")
        ).rstrip("/")
        self._model = model or os.getenv("EMBED_MODEL", "text-embedding-3-small")
        self._dimension = dimension
        self._timeout = timeout_seconds

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        url = f"{self._endpoint}/ai-router-service/embedding/batch"
        payload = {"model": self._model, "data_batch": texts}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
        except httpx.HTTPError as exc:
            log.error("embedding_transport_error", extra={"url": url, "error": str(exc)})
            raise EmbeddingError(f"Failed to reach embedding service: {exc}") from exc

        if resp.status_code in (401, 403):
            raise EmbeddingError(
                "Embedding authentication failed. Check router credentials.",
                details={"status_code": resp.status_code},
            )
        if resp.status_code != 200:
            raise EmbeddingError(
                f"Embedding service returned {resp.status_code}",
                details={"status_code": resp.status_code, "body": resp.text[:500]},
            )

        embeddings = resp.json().get("embeddings") or []
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                "Embedding count mismatch",
                details={"expected": len(texts), "got": len(embeddings)},
            )
        return embeddings
