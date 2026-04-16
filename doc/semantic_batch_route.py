import asyncio
import time
import hashlib
import re
import os
from elasticsearch import AsyncElasticsearch

# ---------------------------------------------------------------------------
# Core Hashing Logic (from hashing.py)
# ---------------------------------------------------------------------------
_WHITESPACE_RE = re.compile(r"\s+")

def normalize_text(text: str) -> str:
    """Collapse all runs of whitespace to a single space and strip."""
    if text is None:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()

def content_hash(text: str) -> str:
    """SHA-256 hex digest of normalized text."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Approach 1: The N+1 Problem (Slow)
# ---------------------------------------------------------------------------
async def route_n_plus_1(chunks: list[str], es: AsyncElasticsearch, index: str) -> float:
    start_time = time.perf_counter()
    
    for chunk in chunks:
        h = content_hash(chunk)
        
        # BOTTLENECK: Awaiting a network call inside a loop
        resp = await es.search(
            index=index,
            query={"term": {"content_hash": h}},
            size=50,
            _source=True,
        )
        existing_docs = [hit["_source"] for hit in resp["hits"]["hits"]]
        
        # (Semantic routing decision logic happens here based on existing_docs)
        if not existing_docs:
            pass # Mark for ingestion
        
    return time.perf_counter() - start_time

# ---------------------------------------------------------------------------
# Approach 2: The Batch Lookup (Fast)
# ---------------------------------------------------------------------------
async def route_batch(chunks: list[str], es: AsyncElasticsearch, index: str) -> float:
    start_time = time.perf_counter()
    
    # 1. CPU-bound hash extraction
    chunk_hashes = [content_hash(chunk) for chunk in chunks]
    
    # 2. SINGLE Network Call for the entire batch
    resp = await es.search(
        index=index,
        query={"terms": {"content_hash": chunk_hashes}},
        size=10000,
        _source=True,
    )
    
    # 3. Create O(1) lookup dictionary locally
    existing_docs_by_hash = {}
    for hit in resp["hits"]["hits"]:
        doc = hit["_source"]
        h = doc.get("content_hash")
        if h not in existing_docs_by_hash:
            existing_docs_by_hash[h] = []
        existing_docs_by_hash[h].append(doc)
    
    # 4. Instant memory-only routing loop
    for chunk, h in zip(chunks, chunk_hashes):
        existing_docs = existing_docs_by_hash.get(h, [])
        
        # (Semantic routing decision logic happens here based on existing_docs)
        if not existing_docs:
            pass # Mark for ingestion
            
    return time.perf_counter() - start_time

# ---------------------------------------------------------------------------
# Benchmark Execution
# ---------------------------------------------------------------------------
async def main():
    es_url = os.getenv("ES_URL", "http://localhost:9200")
    es_username = os.getenv("ES_USERNAME", "elastic")
    es_password = os.getenv("ES_PASSWORD", "")
    
    auth = (es_username, es_password) if es_password else None
    es = AsyncElasticsearch(es_url, basic_auth=auth)
    index_name = "poc_benchmark_index"
    
    print(f"Connecting to real Elasticsearch at {es_url}...")
    
    try:
        if await es.indices.exists(index=index_name):
            await es.indices.delete(index=index_name)
            
        await es.indices.create(
            index=index_name, 
            mappings={"properties": {"content_hash": {"type": "keyword"}}}
        )
        
        # Seed the database with 500 fake chunks so searches have actual work to do
        print("Seeding database with chunks...")
        chunks_to_ingest = [f"This is the content for chunk number {i}" for i in range(500)]
        
        operations = []
        for i, chunk in enumerate(chunks_to_ingest):
            operations.append({"index": {"_index": index_name, "_id": str(i)}})
            operations.append({"content": chunk, "content_hash": content_hash(chunk)})
            
        await es.bulk(operations=operations, refresh="wait_for")
        
        print(f"\nBenchmarking Semantic Routing over {len(chunks_to_ingest)} chunks...\n")
        
        # 1. Run N+1 Approach
        time_n1 = await route_n_plus_1(chunks_to_ingest, es, index_name)
        print(f"[Method 1] N+1 Query Loop took:     {time_n1:.4f} seconds")
        
        # 2. Run Batch Approach
        time_batch = await route_batch(chunks_to_ingest, es, index_name)
        print(f"[Method 2] Batch 'terms' Lookup took: {time_batch:.4f} seconds")
        
        print("\nResult:")
        print(f"The batch lookup is {time_n1 / time_batch:.0f}x faster against real ES for {len(chunks_to_ingest)} chunks.")
        
    finally:
        # Clean up and close connection
        if await es.indices.exists(index=index_name):
            await es.indices.delete(index=index_name)
        await es.close()

if __name__ == "__main__":
    asyncio.run(main())
