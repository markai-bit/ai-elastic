

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.embeddings import Embedder, RouterEmbedder
from app.es_client import get_es_client
from app.index_manager import ensure_index
from app.ingestion import ingest_chunks
from app.logging_config import get_logger
from app.models import IngestRequest, IngestResponse, SchemaResponse, schema_response

log = get_logger(__name__)

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


def get_embedder() -> Embedder:
    """Override in tests via app.dependency_overrides."""
    return RouterEmbedder()


@router.get("/schema", response_model=SchemaResponse)
async def get_schema() -> SchemaResponse:
    """Expose the current metadata schema so clients know what they may send."""
    return schema_response()


@router.post(
    "/index",
    status_code=status.HTTP_201_CREATED,
    summary="Create the index if it doesn't exist (idempotent).",
)
async def create_index(recreate: bool = False, es=Depends(get_es_client)):
    return await ensure_index(es, recreate=recreate)


@router.post(
    "",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest chunks with hash-based dedup (semantic routing).",
)
async def ingest(
    payload: IngestRequest,
    es=Depends(get_es_client),
    embedder: Embedder = Depends(get_embedder),
) -> IngestResponse:
    log.info(
        "ingest_request_received",
        extra={"chunks": len(payload.chunks), "force_on_metadata_change": payload.force_on_metadata_change},
    )
    return await ingest_chunks(payload, es=es, embedder=embedder)
