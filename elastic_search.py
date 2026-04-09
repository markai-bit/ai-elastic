import os
import json
import asyncio
from typing import List, Dict, Any
import PyPDF2

# Use a local, open-source embedding model (runs on your CPU)
# all-MiniLM-L6-v2 generates 384-dimensional vectors
from sentence_transformers import SentenceTransformer
from elasticsearch import AsyncElasticsearch, helpers

# --- CONFIGURATION ---
ES_URL = "http://localhost:9200"
INDEX_NAME = "poc_policy_index"
DUMMY_PDF_PATH = "sample_policy.pdf"
EXPORT_JSON_PATH = "data_for_indexing.json"

# Initialize ES Client and Local Model
es = AsyncElasticsearch(ES_URL)
print("Loading local embedding model (this takes a few seconds the first time)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# ==========================================
# 1. PARSING & SEMANTIC CHUNKING
# ==========================================

def create_dummy_pdf_if_needed():
    """Helper: Creates a dummy PDF if you don't provide one."""
    if not os.path.exists(DUMMY_PDF_PATH):
        print(f"Creating a sample PDF at {DUMMY_PDF_PATH} for testing...")
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(DUMMY_PDF_PATH)
        c.drawString(100, 800, "Corporate Data Retention Policy 2026")
        c.drawString(100, 780, "Section 1: All employee emails must be retained for exactly 5 years.")
        c.drawString(100, 760, "Section 2: Financial records and tax documents must be kept for 7 years.")
        c.drawString(100, 740, "Section 3: Visitor logs at the front desk are purged every 30 days.")
        c.drawString(100, 720, "Section 4: Health and medical data of employees must be kept indefinitely")
        c.drawString(100, 700, "in a highly secure, encrypted server isolated from the main network.")
        c.save()

def parse_pdf(filepath: str) -> str:
    """Extracts raw text from a PDF."""
    print(f"\n[1] Parsing PDF: {filepath}")
    text = ""
    with open(filepath, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def semantic_chunking(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
    """
    Simulates semantic chunking (Service One technique).
    Splits text into meaningful overlapping windows so context isn't lost.
    """
    print(f"[2] Performing Semantic Chunking (Target Size: {chunk_size} chars)...")
    # Split by spaces to get words
    words = text.replace('\n', ' ').split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1 # +1 for space
        
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            # Keep the last few words for overlap
            overlap_words = current_chunk[-max(1, overlap//5):] 
            current_chunk = overlap_words
            current_length = sum(len(w) + 1 for w in overlap_words)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    print(f"    -> Generated {len(chunks)} chunks.")
    return chunks

# ==========================================
# 2. PREPARING DATA & EXPORTING FORMAT
# ==========================================

def prepare_data_for_indexing(chunks: List[str]) -> List[Dict[str, Any]]:
    """Generates vectors for each chunk and formats them for Elasticsearch."""
    print("[3] Generating Vector Embeddings using local MiniLM model...")
    
    # Generate embeddings (Returns a numpy array, we convert to list for JSON/ES)
    embeddings = model.encode(chunks)
    
    documents = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc = {
            "chunk_id": f"chunk_{i}",
            "content": chunk,
            "embedding": embedding.tolist() # Convert numpy array to standard python list
        }
        documents.append(doc)
        
    return documents

def export_to_json(documents: List[Dict[str, Any]]):
    """Downloads/Saves the exact payload going into Elasticsearch."""
    print(f"[4] Saving indexing payload format to {EXPORT_JSON_PATH}...")
    with open(EXPORT_JSON_PATH, "w") as f:
        # Don't print the whole vector in the file to keep it readable, just truncate for display
        display_docs = []
        for d in documents:
            display_doc = d.copy()
            display_doc["embedding"] = f"[{d['embedding'][0]:.4f}, {d['embedding'][1]:.4f}, ... (384 dimensions)]"
            display_docs.append(display_doc)
            
        json.dump(display_docs, f, indent=4)

# ==========================================
# 3. ELASTICSEARCH INGESTION & SEARCH
# ==========================================

async def setup_elasticsearch_index():
    """Creates the index with Dense Vector mapping for KNN search."""
    print(f"\n[5] Setting up Elasticsearch Index '{INDEX_NAME}'...")
    
    # Delete if exists for clean POC run
    if await es.indices.exists(index=INDEX_NAME):
        await es.indices.delete(index=INDEX_NAME)
        
    mapping = {
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "content": {"type": "text"}, # For BM25 Lexical Search
                "embedding": {
                    "type": "dense_vector",  # For Vector KNN Search
                    "dims": 384,             # MiniLM model dimension size
                    "index": True,
                    "similarity": "cosine"   # How we measure distance
                }
            }
        }
    }
    await es.indices.create(index=INDEX_NAME, body=mapping)
    print("    -> Index created successfully.")

async def ingest_to_elasticsearch(documents: List[Dict[str, Any]]):
    """Pushes the documents into ES in bulk."""
    print(f"[6] Ingesting {len(documents)} documents into Elasticsearch...")
    actions = []
    for doc in documents:
        action = {
            "_op_type": "index",
            "_index": INDEX_NAME,
            "_id": doc["chunk_id"],
            "_source": doc
        }
        actions.append(action)
        
    await helpers.async_bulk(es, actions)
    await es.indices.refresh(index=INDEX_NAME) # Force refresh so it's searchable immediately
    print("    -> Ingestion complete!")

async def hybrid_search(query: str, top_k: int = 2):
    """
    Performs true Hybrid Search combining BM25 (Exact text) and KNN (Vector meaning).
    """
    print(f"\n[7] Executing Hybrid Search for query: '{query}'")
    
    # 1. Vectorize the search query
    query_vector = model.encode(query).tolist()
    
    # 2. Build Hybrid Query (Supported in ES 8.x)
    es_query = {
        "size": top_k,
        "query": {
            # LEXICAL (BM25) SEARCH
            "match": {
                "content": {
                    "query": query,
                    "boost": 0.5 # Weight for text match
                }
            }
        },
        # VECTOR (KNN) SEARCH
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": 50,
            "boost": 0.5 # Weight for vector meaning
        }
    }
    
    response = await es.search(index=INDEX_NAME, body=es_query)
    
    print("\n" + "="*40)
    print("🏆 HYBRID SEARCH RESULTS")
    print("="*40)
    for i, hit in enumerate(response["hits"]["hits"]):
        score = hit["_score"]
        content = hit["_source"]["content"]
        print(f"Rank {i+1} | Score: {score:.4f}")
        print(f"Content: {content}\n")

# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
async def main():
    try:
        # Step 0: Ensure we have a PDF to read
        try:
            import reportlab
            create_dummy_pdf_if_needed()
        except ImportError:
            print("Reportlab not installed, skipping PDF creation. Please ensure a sample PDF exists.")

        # Step 1 & 2: Parse and Chunk
        raw_text = parse_pdf(DUMMY_PDF_PATH)
        chunks = semantic_chunking(raw_text, chunk_size=150, overlap=30)
        
        # Step 3 & 4: Embed and Export Format
        documents = prepare_data_for_indexing(chunks)
        export_to_json(documents)
        
        # Step 5 & 6: Elasticsearch Setup and Ingest
        await setup_elasticsearch_index()
        await ingest_to_elasticsearch(documents)
        
        # Step 7: Perform Hybrid Search Tests
        await hybrid_search("How long do I need to keep tax records?")
        await hybrid_search("Rules regarding medical and health information servers.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Make sure Elasticsearch is running on localhost:9200!")
    finally:
        await es.close()

if __name__ == "__main__":
    # Workaround for async loop on Windows if applicable
    import platform
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())
