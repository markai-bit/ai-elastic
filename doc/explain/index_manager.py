

from __future__ import annotations

from typing import Any

from elasticsearch import AsyncElasticsearch
from elasticsearch import NotFoundError as ESNotFoundError

from app import config
from app.config import get_settings
from app.exceptions import IndexOperationError
from app.logging_config import get_logger

log = get_logger(__name__)


# Reserved fields that are *always* present, regardless of metadata schema.
def _core_properties(embed_dimension: int) -> dict[str, Any]:
    return {
        "chunk_id":     {"type": "keyword"},
        "content":      {"type": "text"},
        "content_hash": {"type": "keyword"},  # used by semantic routing / dedup
        "embedding":    {
            "type": "dense_vector",
            "dims": embed_dimension,
            "index": True,
            "similarity": "cosine",
        },
        "ingested_at":  {"type": "date"},
    }


def build_index_mapping(embed_dimension: int) -> dict[str, Any]:
    """Compose the full ES mapping: core fields + dynamic metadata fields."""
    properties = _core_properties(embed_dimension)
    metadata_props = {f.name: f.to_es_mapping() for f in config.METADATA_SCHEMA}

    # Namespace metadata under `metadata.*` so we can never collide with core fields.
    properties["metadata"] = {"properties": metadata_props}

    return {
        # Strict so that anything not in the schema is rejected at ingest time --
        # forces schema changes through config.py instead of silently sprawling.
        "dynamic": "strict",
        "properties": properties,
    }


async def ensure_index(
    es: AsyncElasticsearch,
    index_name: str | None = None,
    embed_dimension: int | None = None,
    recreate: bool = False,
) -> dict[str, Any]:
    """
    Idempotent index creation.

    If `recreate=True`, the index is dropped and rebuilt -- destructive,
    intended for POC/dev only. Production should use reindex flows.
    """
    settings = get_settings()
    name = index_name or settings.index_name
    dims = embed_dimension or settings.embed_dimension

    try:
        exists = await es.indices.exists(index=name)
        if exists:
            if not recreate:
                log.info("index_already_exists", extra={"index": name})
                return {"index": name, "created": False, "recreated": False}
            await es.indices.delete(index=name)
            log.info("index_deleted_for_recreate", extra={"index": name})

        mapping = build_index_mapping(dims)
        await es.indices.create(index=name, mappings=mapping)
        log.info("index_created", extra={"index": name, "embed_dimension": dims})
        return {"index": name, "created": True, "recreated": exists}
    except ESNotFoundError as exc:
        raise IndexOperationError(f"Index not found: {name}") from exc
    except Exception as exc:  # noqa: BLE001 -- map-and-rethrow boundary
        log.exception("index_create_failed", extra={"index": name})
        raise IndexOperationError(f"Failed to ensure index '{name}': {exc}") from exc
