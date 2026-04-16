
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Metadata schema -- the *only* place to edit when compliance requirements
# evolve. Field types map 1:1 to Elasticsearch field types.
# ---------------------------------------------------------------------------

ESFieldType = Literal["keyword", "text", "integer", "long", "float", "date", "boolean"]


@dataclass(frozen=True)
class MetadataField:
    name: str
    es_type: ESFieldType
    required: bool = False
    description: str = ""

    def to_es_mapping(self) -> dict[str, Any]:
        mapping: dict[str, Any] = {"type": self.es_type}
        # `text` fields get a `.keyword` sub-field for exact-match filters/aggs.
        if self.es_type == "text":
            mapping["fields"] = {"keyword": {"type": "keyword", "ignore_above": 256}}
        return mapping


# Edit this list to add / remove metadata fields. Nothing else needs to change.
METADATA_SCHEMA: list[MetadataField] = [
    MetadataField("doc_id",          "keyword", required=True,  description="Stable document identifier"),
    MetadataField("doc_title",       "text",    required=True,  description="Human-readable title"),
    MetadataField("doc_type",        "keyword", required=False, description="policy | regulation | guideline | standard"),
    MetadataField("regulation_name", "keyword", required=False, description="e.g. CRR, Basel III, MiFID II"),
    MetadataField("article_number",  "keyword", required=False, description="e.g. '143(1)'"),
    MetadataField("section",         "keyword", required=False, description="Section identifier"),
    MetadataField("paragraph",       "keyword", required=False, description="Paragraph identifier"),
    MetadataField("page_number",     "integer", required=False, description="Source PDF page"),
    MetadataField("effective_date",  "date",    required=False, description="When the rule comes into force"),
    MetadataField("jurisdiction",    "keyword", required=False, description="UK | EU | US | ..."),
    MetadataField("version",         "keyword", required=False, description="Document version"),
    MetadataField("source_url",      "keyword", required=False, description="Origin URL or filesystem path"),
    MetadataField("chunk_index",     "integer", required=False, description="Index of this chunk in the doc"),
    MetadataField("total_chunks",    "integer", required=False, description="Total chunks in the doc"),
]


def metadata_field_names() -> set[str]:
    return {f.name for f in METADATA_SCHEMA}


def required_metadata_field_names() -> set[str]:
    return {f.name for f in METADATA_SCHEMA if f.required}


# ---------------------------------------------------------------------------
# Application / ES configuration
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    # Elasticsearch
    es_url: str = field(default_factory=lambda: os.getenv("ES_URL", "http://localhost:9200"))
    es_username: str = field(default_factory=lambda: os.getenv("ES_USERNAME", "elastic"))
    # IMPORTANT: never hardcode. Read from env / secret manager only.
    es_password: str = field(default_factory=lambda: os.getenv("ES_PASSWORD", ""))
    index_name: str = field(default_factory=lambda: os.getenv("INDEX_NAME", "poc_policy_index"))

    # Embedding
    embed_dimension: int = field(default_factory=lambda: int(os.getenv("EMBED_DIMENSION", "1536")))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "text-embedding-3-small"))

    # Hybrid search weighting
    bm25_boost: float = field(default_factory=lambda: float(os.getenv("BM25_BOOST", "0.5")))
    knn_boost: float = field(default_factory=lambda: float(os.getenv("KNN_BOOST", "0.5")))
    knn_num_candidates: int = field(default_factory=lambda: int(os.getenv("KNN_NUM_CANDIDATES", "100")))

    # Service
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    request_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    )


def get_settings() -> Settings:
    return Settings()
