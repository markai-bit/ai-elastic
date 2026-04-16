

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.embeddings import Embedder, RouterEmbedder
from app.es_client import get_es_client
from app.logging_config import get_logger
from app.models import SearchRequest, SearchResponse
from app.search import hybrid_search

log = get_logger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


def get_embedder() -> Embedder:
    """Override in tests via app.dependency_overrides."""
    return RouterEmbedder()


@router.post(
    "",
    response_model=SearchResponse,
    summary="Hybrid BM25 + KNN search with optional metadata filters.",
)
async def search(
    payload: SearchRequest,
    es=Depends(get_es_client),
    embedder: Embedder = Depends(get_embedder),
) -> SearchResponse:
    log.info(
        "search_request_received",
        extra={"top_k": payload.top_k, "filters": list(payload.filters.keys())},
    )
    return await hybrid_search(payload, es=es, embedder=embedder)
