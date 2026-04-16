# RAG Service 

focused module + a focused test file.

| step    | problem                             | Code                                           | Test                                  |
| --------- | ------------------------------------ | ---------------------------------------------- | ------------------------------------- |
| DADV-3409 | Semantic routing (hash dedup)        | `app/hashing.py`, `app/ingestion.py`           | `tests/test_semantic_routing.py`      |
| DADV-3394 | Dynamic metadata mapping             | `app/config.py`, `app/index_manager.py`, `app/models.py` | `tests/test_dynamic_metadata.py` |
| DADV-3389 | Ingestion API                        | `app/routers/ingestion_router.py`              | `tests/test_ingestion_api.py`         |
| DADV-3391 | Ingestion error handling + logging   | `app/middleware.py`, `app/exceptions.py`, `app/logging_config.py` | `tests/test_ingestion_errors.py` |
| DADV-3384 | Search API                           | `app/routers/search_router.py`, `app/search.py`| `tests/test_search_api.py`            |
| DADV-3392 | Search error handling + logging     | `app/middleware.py`, `app/exceptions.py`, `app/logging_config.py` | `tests/test_search_errors.py`   |

Tickets DADV-3387 (DB schema/logging strategy) and DADV-3396 (integration)
are not addressed by code in this drop — 3387 is a design decision and 3396
is the final wiring step.

---

## Architecture in 30 seconds

```
                  ┌─────────────────────────────────────────┐
HTTP request ───▶ │ FastAPI router (thin)                   │
                  │   - validates body via Pydantic         │
                  │   - delegates to service                │
                  └────────────────┬────────────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────────────┐
                  │ Service layer                           │
                  │   - app/ingestion.py  (dedup + write)   │
                  │   - app/search.py     (hybrid query)    │
                  │ Raises RagError subclasses on failure.  │
                  └────────────────┬────────────────────────┘
                                   │
                                   ▼
                  ┌──────────────────┐  ┌──────────────────┐
                  │ Elasticsearch    │  │ Embedder         │
                  │ (AsyncES client) │  │ (httpx → router) │
                  └──────────────────┘  └──────────────────┘
```

Cross-cutting (`app/middleware.py`):

- Correlation ID (`X-Correlation-ID`) read from header or minted; propagated
  via `contextvars` so logs in async tasks still see it.
- Global exception handler maps `RagError` subclasses to JSON responses with
  the right HTTP status and includes the correlation ID in the body.
- Per-request access log line in JSON.

---

## Single source of truth: metadata

Edit `METADATA_SCHEMA` in `app/config.py`. That's it. Index mapping,
ingestion validation, search filter validation, and the `/ingestion/schema`
introspection endpoint all derive from this list.

```python
METADATA_SCHEMA = [
    MetadataField("doc_id",       "keyword", required=True),
    MetadataField("doc_title",    "text",    required=True),
    MetadataField("regulation_name", "keyword"),
    MetadataField("article_number",  "keyword"),
    MetadataField("page_number",     "integer"),
    MetadataField("effective_date",  "date"),
    MetadataField("jurisdiction",    "keyword"),
    # ... add or remove fields here, nowhere else.
]
```

Tests `test_adding_a_field_propagates_*` and `test_removing_a_field_*` in
`tests/test_dynamic_metadata.py` lock this contract in place.

---

## Semantic routing (dedup) algorithm

Per chunk, on ingest:

1. `content_hash = SHA256(normalize_whitespace(text))`
2. Search the index for documents with the same `content_hash`.
3. If none → ingest as new.
4. If matches found:
   - **Default behavior** (`force_on_metadata_change=False`): skip.
   - **Force flag on**: if any existing match has equal metadata, skip;
     otherwise ingest as a new document with status `ingested_new_metadata`.

Why hash whitespace-normalized text? PDF re-extraction is noisy — runs of
spaces and stray newlines shouldn't trigger re-ingest. Case is preserved
because compliance docs are case-sensitive ("MUST" ≠ "must").

---

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # then edit ES_PASSWORD etc.

# Start a local Elasticsearch (8.x) however you normally do.
# Then:
uvicorn main:app --reload --port 8000

# Sanity check:
curl localhost:8000/health
curl -X POST localhost:8000/ingestion/index
curl localhost:8000/ingestion/schema | jq

curl -X POST localhost:8000/ingestion \
  -H 'Content-Type: application/json' \
  -H 'X-Correlation-ID: my-test-001' \
  -d '{
    "chunks": [
      {"content": "Article 143(1) of the CRR ...",
       "metadata": {"doc_id": "crr-1", "doc_title": "CRR", "page_number": 1}}
    ]
  }'

curl -X POST localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "internal ratings based approach", "top_k": 5,
       "filters": {"jurisdiction": "UK"}}'
```


---

## Running the tests

```bash
pytest                              # all tests
pytest tests/test_semantic_routing.py -v   # one ticket at a time
pytest -k "force_flag" -v           # by name pattern
```

The tests **do not** require a real Elasticsearch — `tests/conftest.py`
provides `FakeAsyncES` (in-memory async stub) and `FakeEmbedder`
(deterministic SHA-derived vectors). Integration tests against a real
cluster should live in a separate `tests/integration/` folder run from CI
with docker-compose.

>  **The author of this drop did not run pytest** — the iteration
> environment had no internet and no preinstalled deps, so the test files
> are correct-by-inspection only. The first thing you should do is
> `pip install -r requirements.txt && pytest`. If anything breaks it will
> almost certainly be in `tests/conftest.py::FakeAsyncES` (the surface most
> likely to drift from the real `elasticsearch[async]` client).

---

## Known gaps / follow-ups

- **No retries / circuit breaking** on the embedder. Tenacity around
  `RouterEmbedder.embed` is the first thing I'd add for production.
- **`tiktoken`-accurate token counts** are not in this drop. The POC's
  `len(text)//3` heuristic was a footgun at scale; replace before turning
  on the real router under load.
- **RRF (reciprocal rank fusion)** is the right hybrid scoring primitive on
  ES 8.9+. Current code uses simple boost weighting because it works on
  older clusters. Swap when you confirm cluster version.
- **Audit log table (DADV-3387)** is not implemented — needs a Postgres
  schema decision first. The structured JSON access log is the placeholder.
- **`AsyncElasticsearch` lifecycle** — currently a process-global lazy
  singleton. Fine for uvicorn workers, would need adjustment for testing
  alternative ASGI servers.

---

## Layout

```
rag_service/
├── app/
│   ├── config.py              # METADATA_SCHEMA + Settings  ← single source of truth
│   ├── hashing.py             # SHA-256 over normalized text
│   ├── embeddings.py          # Embedder protocol + httpx implementation
│   ├── es_client.py           # AsyncElasticsearch factory
│   ├── exceptions.py          # RagError hierarchy
│   ├── index_manager.py       # ensure_index() + build_index_mapping()
│   ├── ingestion.py           # ingest_chunks() with dedup
│   ├── search.py              # hybrid_search() + build_hybrid_query()
│   ├── models.py              # Pydantic request/response models
│   ├── logging_config.py      # JSON logs + correlation IDs
│   ├── middleware.py          # FastAPI middleware + error handlers
│   └── routers/
│       ├── ingestion_router.py
│       └── search_router.py
├── tests/
│   ├── conftest.py            # FakeAsyncES, FakeEmbedder, app_client
│   ├── test_semantic_routing.py
│   ├── test_dynamic_metadata.py
│   ├── test_ingestion_api.py
│   ├── test_ingestion_errors.py
│   ├── test_search_api.py
│   └── test_search_errors.py
├── main.py                    # FastAPI app + lifespan
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```
