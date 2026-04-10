class SOTAServerlessHybridDB:
    """
    Implements Reciprocal Rank Fusion (RRF) and HNSW (Hierarchical Navigable Small World).
    This exactly mirrors the internal architecture of modern Elasticsearch 8+.
    """
    def __init__(self):
        self.chunks = []
        self.vectors = None
        self.bm25_index = None
        self.hnsw_index = None

    def ingest(self, chunks: List[str]):
        print(f"\n[1] Building SOTA In-Memory Database for {len(chunks)} chunks...")
        self.chunks = chunks
        
        # 1. Generate AI Vectors
        print("    -> Generating AI Vectors...")
        raw_vectors = model.encode(chunks)
        # Pre-normalize vectors to make Cosine Similarity math blazingly fast
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        self.vectors = raw_vectors / np.maximum(norms, 1e-9) 
        
        # 2. Build Vector DB (HNSW Graph)
        if HNSW_AVAILABLE:
            print("    -> Building HNSW Graph (Hierarchical Navigable Small World)...")
            dim = self.vectors.shape[1]
            # Initialize HNSW index using Cosine space
            self.hnsw_index = hnswlib.Index(space='cosine', dim=dim)
            self.hnsw_index.init_index(max_elements=len(chunks), ef_construction=200, M=16)
            self.hnsw_index.add_items(self.vectors, np.arange(len(chunks)))
            self.hnsw_index.set_ef(50) # Set query parameter
        else:
            print("    -> HNSW skipped (hnswlib not installed). Prepping Exact KNN Matrix.")

        # 3. Build Keyword DB (Optimized BM25)
        print("    -> Building BM25 Keyword Index...")
        tokenized_corpus = [clean_text_for_bm25(chunk) for chunk in chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)

    def hybrid_search(self, query: str, top_k: int = 2, rrf_k: int = 60) -> List[Dict[str, Any]]:
        """
        Executes Hybrid Search utilizing the HNSW graph and RRF ranking.
        Returns explicit indexes and score values.
        """
        print(f"\n[2] Executing HNSW Hybrid Search for: '{query}'")
        
        num_docs = len(self.chunks)
        if num_docs == 0: return []

        # --- 1. VECTOR SEARCH (Meaning via HNSW) ---
        query_vector = model.encode([query])[0]
        query_vector = query_vector / np.maximum(np.linalg.norm(query_vector), 1e-9)
        
        vector_scores_dict = {}
        
        if HNSW_AVAILABLE:
            # HNSW traverses the graph and returns the closest candidates instantly
            labels, distances = self.hnsw_index.knn_query(query_vector, k=min(num_docs, rrf_k))
            vector_ranks = labels[0]
            # Convert Cosine Distance back to a Similarity Score (1 - distance)
            for idx, dist in zip(labels[0], distances[0]):
                vector_scores_dict[int(idx)] = 1.0 - float(dist)
        else:
            # Fallback to Brute-Force Dot Product if HNSW isn't installed
            v_scores = np.dot(self.vectors, query_vector)
            vector_ranks = np.argsort(v_scores)[::-1][:rrf_k]
            for idx in vector_ranks:
                vector_scores_dict[int(idx)] = float(v_scores[idx])

        # --- 2. KEYWORD SEARCH (BM25) ---
        tokenized_query = clean_text_for_bm25(query)
        bm25_scores = np.array(self.bm25_index.get_scores(tokenized_query))
        bm25_ranks = np.argsort(bm25_scores)[::-1][:rrf_k]

        # --- 3. RECIPROCAL RANK FUSION (RRF) ---
        rrf_scores = np.zeros(num_docs)

        # Apply RRF penalty curve for HNSW Vector rankings
        for rank, doc_idx in enumerate(vector_ranks):
            rrf_scores[doc_idx] += 1.0 / (rrf_k + rank + 1)

        # Apply RRF penalty curve for Keyword rankings
        for rank, doc_idx in enumerate(bm25_ranks):
            if bm25_scores[doc_idx] > 0:
                rrf_scores[doc_idx] += 1.0 / (rrf_k + rank + 1)

        # --- 4. FINAL SORT & EXPORT ---
        final_top_indices = np.argsort(rrf_scores)[::-1][:top_k]

        print("\n" + "="*50)
        print("🏆 SOTA HYBRID SEARCH RESULTS (HNSW + RRF)")
        print("="*50)
        
        results = []
        for rank, idx in enumerate(final_top_indices):
            idx_int = int(idx)
            v_score = vector_scores_dict.get(idx_int, 0.0)
            k_score = float(bm25_scores[idx_int])
            r_score = float(rrf_scores[idx_int])
            
            print(f"Rank {rank+1} | Index: [{idx_int}] | RRF Score: {r_score:.5f}")
            print(f"     => (HNSW Vector: {v_score:.3f}, BM25 Keyword: {k_score:.3f})")
            print(f"Content: {self.chunks[idx_int]}\n")
            
            # Explicitly exporting the index and individual values
            results.append({
                "chunk_index": idx_int,
                "rrf_fusion_score": r_score,
                "hnsw_vector_value": v_score,
                "bm25_keyword_value": k_score,
                "content": self.chunks[idx_int]
            })
            
        return results
