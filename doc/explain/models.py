

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from app import config
from app.config import (
    metadata_field_names,
    required_metadata_field_names,
)


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


def _validate_metadata(value: dict[str, Any]) -> dict[str, Any]:
    """Reject unknown fields and check required ones. Type checks are deferred
    to ES (which will reject mis-typed values via strict mapping)."""
    allowed = metadata_field_names()
    required = required_metadata_field_names()

    unknown = set(value.keys()) - allowed
    if unknown:
        raise ValueError(
            f"Unknown metadata fields: {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )

    missing = required - set(value.keys())
    if missing:
        raise ValueError(f"Missing required metadata fields: {sorted(missing)}")

    return value


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class ChunkIn(BaseModel):
    """A single text chunk + its metadata."""
    content: str = Field(..., min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def _check_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _validate_metadata(v)


class IngestRequest(BaseModel):
    chunks: list[ChunkIn] = Field(..., min_length=1)
    # Default: skip if a chunk's content_hash already exists.
    # If True: re-ingest as a new doc when an incoming chunk's metadata
    # differs from any existing doc with the same content_hash.
    force_on_metadata_change: bool = False


class ChunkOutcome(BaseModel):
    chunk_index: int
    status: Literal["ingested", "skipped_duplicate", "ingested_new_metadata", "failed"]
    chunk_id: str | None = None
    content_hash: str
    reason: str | None = None


class IngestResponse(BaseModel):
    index: str
    total: int
    ingested: int
    skipped: int
    failed: int
    outcomes: list[ChunkOutcome]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)
    # e.g. {"jurisdiction": "UK", "regulation_name": "CRR"}
    filters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("filters")
    @classmethod
    def _check_filters(cls, v: dict[str, Any]) -> dict[str, Any]:
        allowed = metadata_field_names()
        unknown = set(v.keys()) - allowed
        if unknown:
            raise ValueError(
                f"Cannot filter on unknown fields: {sorted(unknown)}. "
                f"Allowed: {sorted(allowed)}"
            )
        return v


class SearchHit(BaseModel):
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    total_hits: int
    hits: list[SearchHit]


# ---------------------------------------------------------------------------
# Errors / introspection
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None


class SchemaField(BaseModel):
    name: str
    es_type: str
    required: bool
    description: str


class SchemaResponse(BaseModel):
    metadata_fields: list[SchemaField]


def schema_response() -> SchemaResponse:
    return SchemaResponse(
        metadata_fields=[
            SchemaField(
                name=f.name,
                es_type=f.es_type,
                required=f.required,
                description=f.description,
            )
            for f in config.METADATA_SCHEMA
        ]
    )
