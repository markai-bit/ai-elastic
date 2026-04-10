import os
import json
from typing import List, Dict, Any

import PyPDF2
from sentence_transformers import SentenceTransformer

# Import our custom search engine that mirrors Elasticsearch
from elastipy import ElastiPy

# ==============================================================================
# POC SCRIPT: Parsing, Semantic Chunking, Local Indexing, Saving, & Hybrid Search
# ==============================================================================

INDEX_NAME = "poc_policy_index"
DUMMY_PDF_PATH = "sample_policy.pdf"
SAVED_INDEX_PATH = "my_local_index.json"

print("Loading local embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_dummy_pdf():
    """Generates a sample PDF if one doesn't exist."""
    if not os.path.exists(DUMMY_PDF_PATH):
        print(f"Creating sample PDF at {DUMMY_PDF_PATH}...")
        try:
            from reportlab.pdfgen import canvas
            c = canvas.Canvas(DUMMY_PDF_PATH)
            c.drawString(100, 800, "Corporate Data Retention Policy 2026")
            c.drawString(100, 780, "Section 1: All employee emails must be retained for exactly 5 years.")
            c.drawString(100, 760, "Section 2: Financial records and tax documents must be kept for 7 years.")
            c.drawString(100, 740, "Section 3: Visitor logs at the front desk are purged every 30 days.")
            c.drawString(100, 720, "Section 4: Health and medical data of employees must be kept indefinitely")
            c.drawString(100, 700, "in a highly secure, encrypted server isolated from the main network.")
            c.save()
        except ImportError:
            print("Error: Could not create dummy PDF. Please run: pip install reportlab")
            exit(1)

def parse_pdf(filepath: str) -> str:
    """Reads raw text from a PDF."""
    print(f"\n[1] Parsing PDF: {filepath}")
    text = ""
    with open(filepath, "rb") as file:
        for page in PyPDF2.PdfReader(file).pages:
            text += page.extract_text() + "\n"
    return text

def semantic_chunking(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    """Splits text into overlapping windows."""
    print(f"[2] Performing Semantic Chunking...")
    words = text.replace('\n', ' ').split()
    chunks, curr, curr_len = [], [], 0
    for w in words:
        curr.append(w)
        curr_len += len(w) + 1
        if curr_len >= chunk_size:
            chunks.append(" ".join(curr))
            curr = curr[-max(1, overlap//5):]
            curr_len = sum(len(x)+1 for x in curr)
    if curr: 
        chunks.append(" ".join(curr))
    return chunks

def prepare_data_for_indexing(chunks: List[str]) -> List[Dict[str, Any]]:
    """Generates vectors for each chunk and formats them for the search engine."""
    print("[3] Generating Vector Embeddings using local MiniLM model...")
    embeddings = model.encode(chunks)
    documents = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        doc = {
            "chunk_id": f"chunk_{i}",
            "content": chunk,
            "embedding": embedding.tolist()
        }
        documents.append(doc)
    return documents

def run_poc():
    # Initialize our pure-Python Elasticsearch clone
    es = ElastiPy()

    # ==============================================================
    # 1. LOAD OR BUILD THE INDEX
    # ==============================================================
    if os.path.exists(SAVED_INDEX_PATH):
        print(f"\n[INFO] Found saved index! Loading from {SAVED_INDEX_PATH}...")
        
        # This one line loads the text JSON AND the HNSW binary graph!
        loaded_name = es.load_index(SAVED_INDEX_PATH)
        print(f"    -> Successfully loaded index: '{loaded_name}'")
        
    else:
        print("\n[INFO] No saved index found. Building from scratch...")
        create_dummy_pdf()
        text = parse_pdf(DUMMY_PDF_PATH)
        chunks = semantic_chunking(text)
        documents = prepare_data_for_indexing(chunks)
        
        print(f"[4] Creating mapping and indexing into ElastiPy...")
        # Create the index and tell it exactly which field holds our AI vectors
        es.indices.create(INDEX_NAME, mappings={
            "properties": {
                "chunk_id": {"type": "keyword"},
                "content": {"type": "text"},
                "embedding": {"type": "dense_vector", "dims": 384} # MiniLM uses 384 dimensions
            }
        })

        # Bulk insert the documents
        for doc in documents:
            es.index(INDEX_NAME, doc, id=doc["chunk_id"])
            
        print(f"    -> Successfully indexed {len(documents)} vectors.")
        
        # SAVE THE INDEX TO DISK!
        print(f"[5] Saving the generated index to disk at '{SAVED_INDEX_PATH}'...")
        es.save_index(INDEX_NAME, SAVED_INDEX_PATH)
        print("    -> Save complete! Run this script again to see it load instantly.")


    # ==============================================================
    # 2. PERFORM SOTA HYBRID SEARCH (RRF + HNSW)
    # ==============================================================
    query = "How long do I need to keep tax records?"
    print(f"\n[6] Executing Hybrid Search: '{query}'")
    
    # Vectorize the user's question
    query_vector = model.encode(query).tolist()
    
    # Construct the payload mimicking the real Elasticsearch structure
    search_payload = {
        "size": 2,
        "_source": ["chunk_id", "content"], # Don't print the massive vector array to the console
        "query": {
            "match": {
                "content": query # Lexical Keyword Search
            }
        },
        "knn": {
            "field": "embedding",
            "query_vector": query_vector, # Semantic Meaning Search
            "k": 2
        }
    }
    
    # Pass payload into our custom engine
    results = es.search(INDEX_NAME, search_payload)
    
    # Display the results
    print("\n" + "="*50)
    print("🏆 LOCAL ELASTIPY HYBRID SEARCH RESULTS")
    print("="*50)
    for i, hit in enumerate(results["hits"]["hits"]):
        print(f"Rank {i+1} | Document ID: {hit['_id']}")
        print(f"Content: {hit['_source']['content']}")
        
        # ElastiPy explicitly breaks down the RRF scores for you!
        if "_rrf_score" in hit:
            print(f"  ↳ RRF Fused Score: {hit['_rrf_score']:.4f}")
            print(f"  ↳ HNSW Vector Score: {hit['_vector_score']:.4f}")
            print(f"  ↳ BM25 Keyword Score: {hit['_keyword_score']:.4f}")
        print("-" * 50)

if __name__ == "__main__":
    run_poc()
