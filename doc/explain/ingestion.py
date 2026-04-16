

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from elasticsearch import AsyncElasticsearch

from app.config import get_settings
from app.embeddings import Embedder
from app.exceptions import IndexOperationError
from app.hashing import content_hash
from app.logging_config import get_logger
from app.models import ChunkIn, ChunkOutcome, IngestRequest, IngestResponse

log = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metadata_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Strict equality on the union of keys. None == missing for robustness."""
    keys = set(a.keys()) | set(b.keys())
    return all(a.get(k) == b.get(k) for k in keys)


async def _existing_docs_for_hash(
    es: AsyncElasticsearch, index: str, hash_value: str
) -> list[dict[str, Any]]:
    """Return _source for all existing docs whose content_hash matches."""
    resp = await es.search(
        index=index,
        query={"term": {"content_hash": hash_value}},
        size=50,  # cap; same-hash collisions should be tiny in practice
        _source=True,
    )
    return [hit["_source"] for hit in resp["hits"]["hits"]]


def _build_doc(chunk: ChunkIn, hash_value: str, embedding: list[float]) -> dict[str, Any]:
    return {
        "chunk_id": uuid.uuid4().hex,
        "content": chunk.content,
        "content_hash": hash_value,
        "embedding": embedding,
        "metadata": chunk.metadata,
        "ingested_at": _now_iso(),
    }


async def ingest_chunks(
    request: IngestRequest,
    es: AsyncElasticsearch,
    embedder: Embedder,
    index_name: str | None = None,
) -> IngestResponse:
    settings = get_settings()
    index = index_name or settings.index_name

    # ---- Pass 1: hash + dedup classification ---------------------------------
    decisions: list[dict[str, Any]] = []  # parallel to request.chunks
    chunks_to_embed: list[tuple[int, ChunkIn, str]] = []  # (idx, chunk, hash)

    for i, chunk in enumerate(request.chunks):
        h = content_hash(chunk.content)
        try:
            existing = await _existing_docs_for_hash(es, index, h)
        except Exception as exc:  # noqa: BLE001
            log.exception("dedup_lookup_failed", extra={"chunk_index": i, "hash": h})
            decisions.append({"action": "fail", "hash": h, "reason": f"dedup lookup failed: {exc}"})
            continue

        if not existing:
            decisions.append({"action": "ingest", "hash": h, "status": "ingested"})
            chunks_to_embed.append((i, chunk, h))
            continue

        if not request.force_on_metadata_change:
            decisions.append({
                "action": "skip", "hash": h, "status": "skipped_duplicate",
                "reason": f"content_hash already present ({len(existing)} doc(s))",
            })
            continue

        # force_on_metadata_change=True path
        if any(_metadata_equal(chunk.metadata, doc.get("metadata", {})) for doc in existing):
            decisions.append({
                "action": "skip", "hash": h, "status": "skipped_duplicate",
                "reason": "content_hash and metadata both already present",
            })
        else:
            decisions.append({
                "action": "ingest", "hash": h, "status": "ingested_new_metadata",
            })
            chunks_to_embed.append((i, chunk, h))

    # ---- Pass 2: embed only what we're actually ingesting --------------------
    embeddings: list[list[float]] = []
    if chunks_to_embed:
        embeddings = await embedder.embed([c.content for _, c, _ in chunks_to_embed])

    # ---- Pass 3: bulk index --------------------------------------------------
    operations: list[dict[str, Any]] = []
    chunk_ids: dict[int, str] = {}
    for (orig_idx, chunk, h), emb in zip(chunks_to_embed, embeddings):
        doc = _build_doc(chunk, h, emb)
        chunk_ids[orig_idx] = doc["chunk_id"]
        operations.append({"index": {"_index": index, "_id": doc["chunk_id"]}})
        operations.append(doc)

    if operations:
        try:
            resp = await es.bulk(operations=operations, refresh="wait_for")
        except Exception as exc:  # noqa: BLE001
            log.exception("bulk_ingest_failed", extra={"index": index, "ops": len(operations)})
            raise IndexOperationError(f"Bulk ingest failed: {exc}") from exc
        if resp.get("errors"):
            # Surface the first error for diagnosis but don't blow up the whole batch.
            first_err = next(
                (item for item in resp.get("items", []) if "error" in item.get("index", {})),
                None,
            )
            log.warning("bulk_ingest_partial_failure", extra={"first_error": first_err})

    # ---- Build response ------------------------------------------------------
    outcomes: list[ChunkOutcome] = []
    ingested = skipped = failed = 0
    for i, decision in enumerate(decisions):
        if decision["action"] == "ingest":
            outcomes.append(ChunkOutcome(
                chunk_index=i,
                status=decision["status"],
                chunk_id=chunk_ids.get(i),
                content_hash=decision["hash"],
            ))
            ingested += 1
        elif decision["action"] == "skip":
            outcomes.append(ChunkOutcome(
                chunk_index=i,
                status="skipped_duplicate",
                content_hash=decision["hash"],
                reason=decision.get("reason"),
            ))
            skipped += 1
        else:
            outcomes.append(ChunkOutcome(
                chunk_index=i,
                status="failed",
                content_hash=decision["hash"],
                reason=decision.get("reason"),
            ))
            failed += 1

    log.info("ingest_summary", extra={
        "index": index, "total": len(request.chunks),
        "ingested": ingested, "skipped": skipped, "failed": failed,
    })

    return IngestResponse(
        index=index,
        total=len(request.chunks),
        ingested=ingested,
        skipped=skipped,
        failed=failed,
        outcomes=outcomes,
    )
