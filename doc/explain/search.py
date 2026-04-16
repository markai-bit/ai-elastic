

from __future__ import annotations

from typing import Any

from elasticsearch import AsyncElasticsearch
from elasticsearch import NotFoundError as ESNotFoundError

from app.config import get_settings
from app.embeddings import Embedder
from app.exceptions import IndexNotFoundError, SearchError
from app.logging_config import get_logger
from app.models import SearchHit, SearchRequest, SearchResponse

log = get_logger(__name__)


def _filter_clauses(filters: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate {field: value} into ES term clauses on `metadata.<field>`."""
    clauses: list[dict[str, Any]] = []
    for field, value in filters.items():
        if isinstance(value, list):
            clauses.append({"terms": {f"metadata.{field}": value}})
        else:
            clauses.append({"term": {f"metadata.{field}": value}})
    return clauses


def build_hybrid_query(
    *,
    query_text: str,
    query_vector: list[float],
    top_k: int,
    filters: dict[str, Any],
    bm25_boost: float,
    knn_boost: float,
    knn_num_candidates: int,
) -> dict[str, Any]:
    filter_clauses = _filter_clauses(filters)

    bm25_query: dict[str, Any] = {
        "bool": {
            "must":   [{"match": {"content": {"query": query_text, "boost": bm25_boost}}}],
            "filter": filter_clauses,
        }
    }

    knn_query: dict[str, Any] = {
        "field": "embedding",
        "query_vector": query_vector,
        "k": top_k,
        "num_candidates": max(knn_num_candidates, top_k * 10),
        "boost": knn_boost,
    }
    if filter_clauses:
        knn_query["filter"] = filter_clauses

    return {"size": top_k, "query": bm25_query, "knn": knn_query}


async def hybrid_search(
    request: SearchRequest,
    es: AsyncElasticsearch,
    embedder: Embedder,
    index_name: str | None = None,
) -> SearchResponse:
    settings = get_settings()
    index = index_name or settings.index_name

    try:
        vectors = await embedder.embed([request.query])
    except Exception:
        # EmbeddingError already logged in embedder; re-raise so router maps it.
        raise

    if not vectors or not vectors[0]:
        raise SearchError("Embedder returned no vector for query")
    query_vector = vectors[0]

    body = build_hybrid_query(
        query_text=request.query,
        query_vector=query_vector,
        top_k=request.top_k,
        filters=request.filters,
        bm25_boost=settings.bm25_boost,
        knn_boost=settings.knn_boost,
        knn_num_candidates=settings.knn_num_candidates,
    )

    try:
        resp = await es.search(index=index, body=body)
    except ESNotFoundError as exc:
        raise IndexNotFoundError(f"Index not found: {index}") from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("search_failed", extra={"index": index, "query": request.query})
        raise SearchError(f"Elasticsearch search failed: {exc}") from exc

    hits_raw = resp.get("hits", {}).get("hits", [])
    hits = [
        SearchHit(
            chunk_id=h["_source"].get("chunk_id", h["_id"]),
            score=h["_score"],
            content=h["_source"].get("content", ""),
            metadata=h["_source"].get("metadata", {}),
        )
        for h in hits_raw
    ]

    log.info(
        "search_completed",
        extra={"index": index, "query": request.query, "hits": len(hits), "top_k": request.top_k},
    )

    return SearchResponse(query=request.query, total_hits=len(hits), hits=hits)
