

from __future__ import annotations

import pytest

from app.hashing import content_hash, normalize_text
from app.index_manager import ensure_index
from app.ingestion import ingest_chunks
from app.models import ChunkIn, IngestRequest


# ---------------------------------------------------------------------------
# Pure-function tests for the hash itself
# ---------------------------------------------------------------------------


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  hello \n  world\t\t!  ") == "hello world !"


def test_hash_is_whitespace_insensitive():
    assert content_hash("Article 143(1)\n  of the CRR") == content_hash("Article 143(1) of the CRR")


def test_hash_is_case_sensitive():
    # Compliance-relevant: MUST vs must.
    assert content_hash("Firms MUST comply.") != content_hash("Firms must comply.")


def test_hash_is_deterministic_and_64_chars():
    h = content_hash("anything")
    assert h == content_hash("anything")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# Ingestion behavior
# ---------------------------------------------------------------------------


def _chunk(content: str, **metadata) -> ChunkIn:
    md = {"doc_id": "doc-1", "doc_title": "Test Doc"}
    md.update(metadata)
    return ChunkIn(content=content, metadata=md)


@pytest.mark.asyncio
async def test_first_ingest_writes_new_chunk(fake_es, fake_embedder):
    await ensure_index(fake_es, index_name="t1", embed_dimension=fake_embedder.dimension)
    req = IngestRequest(chunks=[_chunk("Article 143 governs IRB approach.")])
    resp = await ingest_chunks(req, es=fake_es, embedder=fake_embedder, index_name="t1")

    assert resp.ingested == 1
    assert resp.skipped == 0
    assert resp.outcomes[0].status == "ingested"
    assert resp.outcomes[0].chunk_id is not None
    # Verify it's actually in the fake store.
    assert len(fake_es.indices_data["t1"]["docs"]) == 1


@pytest.mark.asyncio
async def test_duplicate_is_skipped_by_default(fake_es, fake_embedder):
    await ensure_index(fake_es, index_name="t1", embed_dimension=fake_embedder.dimension)
    req = IngestRequest(chunks=[_chunk("Same content here.")])
    await ingest_chunks(req, es=fake_es, embedder=fake_embedder, index_name="t1")

    # Re-ingest the exact same chunk.
    resp = await ingest_chunks(req, es=fake_es, embedder=fake_embedder, index_name="t1")
    assert resp.ingested == 0
    assert resp.skipped == 1
    assert resp.outcomes[0].status == "skipped_duplicate"
    assert "content_hash already present" in resp.outcomes[0].reason
    assert len(fake_es.indices_data["t1"]["docs"]) == 1


@pytest.mark.asyncio
async def test_force_flag_with_same_metadata_still_skips(fake_es, fake_embedder):
    """force_on_metadata_change should *only* re-ingest when metadata differs."""
    await ensure_index(fake_es, index_name="t1", embed_dimension=fake_embedder.dimension)
    chunk = _chunk("Identical content", page_number=5)
    await ingest_chunks(IngestRequest(chunks=[chunk]), es=fake_es, embedder=fake_embedder, index_name="t1")

    resp = await ingest_chunks(
        IngestRequest(chunks=[chunk], force_on_metadata_change=True),
        es=fake_es, embedder=fake_embedder, index_name="t1",
    )
    assert resp.ingested == 0
    assert resp.skipped == 1
    assert len(fake_es.indices_data["t1"]["docs"]) == 1


@pytest.mark.asyncio
async def test_force_flag_with_different_metadata_ingests(fake_es, fake_embedder):
    """Same text, different metadata, force flag on -> new doc gets written."""
    await ensure_index(fake_es, index_name="t1", embed_dimension=fake_embedder.dimension)
    original = _chunk("Reusable boilerplate clause.", page_number=5, jurisdiction="UK")
    variant  = _chunk("Reusable boilerplate clause.", page_number=12, jurisdiction="EU")

    await ingest_chunks(IngestRequest(chunks=[original]), es=fake_es, embedder=fake_embedder, index_name="t1")
    resp = await ingest_chunks(
        IngestRequest(chunks=[variant], force_on_metadata_change=True),
        es=fake_es, embedder=fake_embedder, index_name="t1",
    )
    assert resp.ingested == 1
    assert resp.skipped == 0
    assert resp.outcomes[0].status == "ingested_new_metadata"
    assert len(fake_es.indices_data["t1"]["docs"]) == 2


@pytest.mark.asyncio
async def test_force_flag_off_skips_even_with_different_metadata(fake_es, fake_embedder):
    """Default behavior: hash match alone is enough to skip."""
    await ensure_index(fake_es, index_name="t1", embed_dimension=fake_embedder.dimension)
    original = _chunk("Same words.", page_number=5)
    variant  = _chunk("Same words.", page_number=99)

    await ingest_chunks(IngestRequest(chunks=[original]), es=fake_es, embedder=fake_embedder, index_name="t1")
    resp = await ingest_chunks(
        IngestRequest(chunks=[variant]),  # force flag default False
        es=fake_es, embedder=fake_embedder, index_name="t1",
    )
    assert resp.ingested == 0
    assert resp.skipped == 1
    assert len(fake_es.indices_data["t1"]["docs"]) == 1


@pytest.mark.asyncio
async def test_mixed_batch_partitions_correctly(fake_es, fake_embedder):
    """A batch with some new and some duplicate chunks should ingest the new ones only."""
    await ensure_index(fake_es, index_name="t1", embed_dimension=fake_embedder.dimension)
    # Pre-load one chunk so it's a "duplicate" in the next call.
    pre = _chunk("Already indexed sentence.")
    await ingest_chunks(IngestRequest(chunks=[pre]), es=fake_es, embedder=fake_embedder, index_name="t1")

    batch = IngestRequest(chunks=[
        _chunk("Brand new sentence A."),
        pre,                                  # duplicate
        _chunk("Brand new sentence B."),
    ])
    resp = await ingest_chunks(batch, es=fake_es, embedder=fake_embedder, index_name="t1")

    assert resp.total == 3
    assert resp.ingested == 2
    assert resp.skipped == 1
    statuses = [o.status for o in resp.outcomes]
    assert statuses == ["ingested", "skipped_duplicate", "ingested"]
    assert len(fake_es.indices_data["t1"]["docs"]) == 3  # 1 pre-existing + 2 new


@pytest.mark.asyncio
async def test_whitespace_only_difference_is_treated_as_duplicate(fake_es, fake_embedder):
    """PDF re-extraction often introduces extra whitespace; we want that to be a no-op."""
    await ensure_index(fake_es, index_name="t1", embed_dimension=fake_embedder.dimension)
    a = _chunk("The firm shall maintain capital.")
    b = _chunk("The  firm   shall\nmaintain  capital.")
    await ingest_chunks(IngestRequest(chunks=[a]), es=fake_es, embedder=fake_embedder, index_name="t1")
    resp = await ingest_chunks(IngestRequest(chunks=[b]), es=fake_es, embedder=fake_embedder, index_name="t1")
    assert resp.skipped == 1
