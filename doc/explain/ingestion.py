from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from elasticsearch import AsyncElasticsearch

from app.config import get_settings
from app.embeddings import Embedder
from app.exceptions import IndexOperationError
from app.hashing import content_hash
from app.logging_config import get_logger
from app.models import ChunkIn, ChunkOutcome, IngestRequest, IngestResponse

log = get_logger(__name__)


# ES limits worth respecting:
#   - index.max_terms_count (default 65,536)  -> cap on `terms` clause size
#   - index.max_result_window (default 10,000) -> cap on `size`
# We use 5,000 to stay well clear of both with headroom for future schema churn.
_MAX_TERMS_PER_QUERY = 5000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _metadata_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Strict equality on the union of keys. None == missing for robustness."""
    keys = set(a.keys()) | set(b.keys())
    return all(a.get(k) == b.get(k) for k in keys)


def _batched(values: list, size: int) -> Iterable[list]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


async def _existing_docs_by_hash(
    es: AsyncElasticsearch, index: str, hashes: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """
    One (or a few) `terms` query instead of N `term` queries.
    Returns {content_hash: [source_doc, ...]} for every hash that had matches.
    """
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not hashes:
        return result

    # Preserve order while deduping -- lets us not over-query on repeated hashes.
    unique = list(dict.fromkeys(hashes))

    for sub in _batched(unique, _MAX_TERMS_PER_QUERY):
        resp = await es.search(
            index=index,
            query={"terms": {"content_hash": sub}},
            size=_MAX_TERMS_PER_QUERY,
            _source=True,
        )
        for hit in resp["hits"]["hits"]:
            doc = hit["_source"]
            h = doc.get("content_hash")
            if h:
                result[h].append(doc)
    return result


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

    # ---- Pass 1: hash every chunk (CPU only, no IO) --------------------------
    chunk_hashes = [content_hash(c.content) for c in request.chunks]

    # ---- Pass 2: ONE batched lookup (was N+1 before) -------------------------
    try:
        existing_by_hash = await _existing_docs_by_hash(es, index, chunk_hashes)
    except Exception as exc:  # noqa: BLE001 -- boundary with a flaky dependency
        log.exception(
            "dedup_lookup_failed",
            extra={"index": index, "unique_hashes": len(set(chunk_hashes))},
        )
        # Treat whole-batch lookup failure as a hard error rather than marking
        # every chunk as "failed" -- gives a more actionable response.
        raise IndexOperationError(f"Dedup lookup failed: {exc}") from exc

    # ---- Pass 3: route each chunk in memory (no IO) --------------------------
    decisions: list[dict[str, Any]] = []
    chunks_to_embed: list[tuple[int, ChunkIn, str]] = []  # (orig_idx, chunk, hash)

    for i, (chunk, h) in enumerate(zip(request.chunks, chunk_hashes)):
        existing = existing_by_hash.get(h, [])

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

        # force_on_metadata_change=True
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

    # ---- Pass 4: embed only what we're actually ingesting -------------------
    embeddings: list[list[float]] = []
    if chunks_to_embed:
        embeddings = await embedder.embed([c.content for _, c, _ in chunks_to_embed])

    # ---- Pass 5: bulk index --------------------------------------------------
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
    # Note: per-chunk "failed" outcomes are no longer produced -- a dedup lookup
    # failure now raises IndexOperationError (502), and bulk failures likewise.
    # Left in the union type so we can reintroduce partial-failure reporting
    # later without an API break.
    outcomes: list[ChunkOutcome] = []
    ingested = skipped = 0
    for i, decision in enumerate(decisions):
        if decision["action"] == "ingest":
            outcomes.append(ChunkOutcome(
                chunk_index=i,
                status=decision["status"],
                chunk_id=chunk_ids.get(i),
                content_hash=decision["hash"],
            ))
            ingested += 1
        else:  # skip
            outcomes.append(ChunkOutcome(
                chunk_index=i,
                status="skipped_duplicate",
                content_hash=decision["hash"],
                reason=decision.get("reason"),
            ))
            skipped += 1

    log.info("ingest_summary", extra={
        "index": index, "total": len(request.chunks),
        "ingested": ingested, "skipped": skipped,
    })

    return IngestResponse(
        index=index,
        total=len(request.chunks),
        ingested=ingested,
        skipped=skipped,
        failed=0,
        outcomes=outcomes,
    )
