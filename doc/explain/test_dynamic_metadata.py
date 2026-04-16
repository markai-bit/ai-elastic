

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app import config as config_module
from app.config import METADATA_SCHEMA, MetadataField, metadata_field_names
from app.index_manager import build_index_mapping
from app.models import ChunkIn, SearchRequest, schema_response


# ---------------------------------------------------------------------------
# Index mapping is generated from schema
# ---------------------------------------------------------------------------


def test_index_mapping_contains_core_fields():
    mapping = build_index_mapping(embed_dimension=8)
    props = mapping["properties"]
    assert props["chunk_id"]["type"] == "keyword"
    assert props["content"]["type"] == "text"
    assert props["content_hash"]["type"] == "keyword"
    assert props["embedding"]["type"] == "dense_vector"
    assert props["embedding"]["dims"] == 8
    assert props["embedding"]["similarity"] == "cosine"
    assert props["ingested_at"]["type"] == "date"


def test_index_mapping_contains_every_metadata_field():
    mapping = build_index_mapping(embed_dimension=8)
    md_props = mapping["properties"]["metadata"]["properties"]
    for field in METADATA_SCHEMA:
        assert field.name in md_props, f"missing {field.name} in mapping"
        assert md_props[field.name]["type"] == field.es_type


def test_text_metadata_fields_get_keyword_subfield():
    """`text` is great for ranking, but you need `.keyword` for filters/aggs."""
    mapping = build_index_mapping(embed_dimension=8)
    md_props = mapping["properties"]["metadata"]["properties"]
    text_fields = [f for f in METADATA_SCHEMA if f.es_type == "text"]
    assert text_fields, "schema should have at least one text field for this assertion to be meaningful"
    for f in text_fields:
        assert "fields" in md_props[f.name]
        assert md_props[f.name]["fields"]["keyword"]["type"] == "keyword"


def test_index_mapping_is_strict():
    """Strict mapping prevents silent metadata sprawl -- changes go through config."""
    mapping = build_index_mapping(embed_dimension=8)
    assert mapping["dynamic"] == "strict"


# ---------------------------------------------------------------------------
# Ingestion validates metadata against the schema
# ---------------------------------------------------------------------------


def test_ingestion_rejects_unknown_metadata_field():
    with pytest.raises(ValidationError) as excinfo:
        ChunkIn(content="x", metadata={"doc_id": "d", "doc_title": "t", "color": "red"})
    assert "Unknown metadata fields" in str(excinfo.value)
    assert "color" in str(excinfo.value)


def test_ingestion_rejects_missing_required_metadata():
    with pytest.raises(ValidationError) as excinfo:
        ChunkIn(content="x", metadata={"doc_id": "only-doc-id-no-title"})
    assert "Missing required metadata" in str(excinfo.value)
    assert "doc_title" in str(excinfo.value)


def test_ingestion_accepts_required_only():
    chunk = ChunkIn(content="x", metadata={"doc_id": "d", "doc_title": "t"})
    assert chunk.metadata == {"doc_id": "d", "doc_title": "t"}


def test_ingestion_accepts_full_schema():
    md = {f.name: ("v" if f.es_type in ("keyword", "text") else 1) for f in METADATA_SCHEMA}
    chunk = ChunkIn(content="x", metadata=md)
    assert set(chunk.metadata.keys()) == metadata_field_names()


# ---------------------------------------------------------------------------
# Search validates filter fields against the schema
# ---------------------------------------------------------------------------


def test_search_rejects_unknown_filter_field():
    with pytest.raises(ValidationError) as excinfo:
        SearchRequest(query="q", filters={"not_a_real_field": "x"})
    assert "Cannot filter on unknown fields" in str(excinfo.value)


def test_search_accepts_valid_filter_field():
    req = SearchRequest(query="q", filters={"jurisdiction": "UK"})
    assert req.filters == {"jurisdiction": "UK"}


def test_search_accepts_list_filter_value():
    req = SearchRequest(query="q", filters={"jurisdiction": ["UK", "EU"]})
    assert req.filters["jurisdiction"] == ["UK", "EU"]


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------


def test_schema_response_lists_every_field():
    resp = schema_response()
    assert {f.name for f in resp.metadata_fields} == metadata_field_names()


# ---------------------------------------------------------------------------
# The headline contract: editing METADATA_SCHEMA propagates everywhere
# ---------------------------------------------------------------------------


def test_adding_a_field_propagates_to_mapping_and_validation(monkeypatch):
    """Simulate a compliance team adding `risk_category` and verify it just works."""
    new_field = MetadataField("risk_category", "keyword", required=False, description="Risk class")
    extended = list(METADATA_SCHEMA) + [new_field]
    monkeypatch.setattr(config_module, "METADATA_SCHEMA", extended)

    # Re-import the modules that snapshot the schema at call time.
    # build_index_mapping reads METADATA_SCHEMA each call, so this just works.
    mapping = build_index_mapping(embed_dimension=8)
    assert "risk_category" in mapping["properties"]["metadata"]["properties"]

    # And a chunk with the new field is now valid.
    chunk = ChunkIn(
        content="x",
        metadata={"doc_id": "d", "doc_title": "t", "risk_category": "high"},
    )
    assert chunk.metadata["risk_category"] == "high"

    # And we can filter on it.
    req = SearchRequest(query="q", filters={"risk_category": "high"})
    assert req.filters["risk_category"] == "high"


def test_removing_a_field_makes_it_invalid_everywhere(monkeypatch):
    """If `jurisdiction` is removed from schema, both ingestion and search should reject it."""
    reduced = [f for f in METADATA_SCHEMA if f.name != "jurisdiction"]
    monkeypatch.setattr(config_module, "METADATA_SCHEMA", reduced)

    mapping = build_index_mapping(embed_dimension=8)
    assert "jurisdiction" not in mapping["properties"]["metadata"]["properties"]

    with pytest.raises(ValidationError):
        ChunkIn(content="x", metadata={"doc_id": "d", "doc_title": "t", "jurisdiction": "UK"})

    with pytest.raises(ValidationError):
        SearchRequest(query="q", filters={"jurisdiction": "UK"})
