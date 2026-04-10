"""
ElastiPy — A pure Python, zero-dependency search engine
Supports up to 1000 documents with full-text search, filtering,
aggregations, pagination, bulk indexing, and Hybrid Search via hnsw-lite.
"""

import json
import math
import os
import re
import time
import uuid
import pickle
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

# For HNSW Vector Search
import numpy as np

# Safely handle the hnsw-lite import based on official PyPI documentation
try:
    from hnsw.hnsw import HNSW
    from hnsw.node import Node
    HNSW_AVAILABLE = True
except ImportError:
    HNSW_AVAILABLE = False
    print("⚠️ hnsw-lite not found! Run 'pip install hnsw-lite' to enable fast Vector Search.")

# ─────────────────────────────────────────────
#  HNSW Lite Adaptive Wrapper
# ─────────────────────────────────────────────
class HNSWLiteWrapper:
    """
    Safely wraps the hnsw-lite PyPI package based on its official API.
    Handles the Node object requirements and negated distances automatically.
    """
    def __init__(self, dim: int):
        # The hnsw-lite package doesn't strictly need dim at init, but we keep the signature
        if HNSW_AVAILABLE:
            self.index = HNSW(space="cosine", M=16, ef_construction=200)
        else:
            self.index = None

    def add_items(self, vectors: np.ndarray, ids: np.ndarray):
        if not self.index: return
        for v, idx in zip(vectors, ids):
            # hnsw-lite expects native Python lists
            vec_list = [float(x) for x in v]
            # We store the internal hnsw_id directly inside the node's metadata
            self.index.insert(vec_list, {"id": int(idx)})

    def knn_query(self, query_vector: np.ndarray, k: int):
        if not self.index: 
            return np.array([]), np.array([])
            
        # Extract the 1D list from the numpy array
        if query_vector.ndim == 2:
            vec_list = [float(x) for x in query_vector[0]]
        else:
            vec_list = [float(x) for x in query_vector]
            
        # hnsw-lite requires wrapping queries in a Node object (level 0)
        query_node = Node(vec_list, 0)
        
        # Returns list of tuples: (negated_distance, Node)
        results = self.index.knn_search(query_node, k)
        
        labels = []
        distances = []
        
        for dist, node in results:
            labels.append(node.metadata["id"])
            # The documentation explicitly states: "Distances are negated in results"
            actual_distance = -dist
            distances.append(actual_distance)
            
        return np.array([labels]), np.array([distances])

    def save_index(self, filepath: str):
        # Pure Python fallback for saving the class object
        with open(filepath, 'wb') as f:
            pickle.dump(self.index, f)

    def load_index(self, filepath: str, max_elements: int):
        with open(filepath, 'rb') as f:
            self.index = pickle.load(f)

# ─────────────────────────────────────────────
#  Tokenizer & Text utilities
# ─────────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "has", "have", "he", "her", "his", "how", "i",
    "if", "in", "is", "it", "its", "of", "on", "or", "our", "she",
    "so", "that", "the", "their", "they", "this", "to", "was", "we",
    "were", "what", "when", "which", "who", "will", "with", "you",
}

def tokenize(text: str, remove_stop_words: bool = True) -> list[str]:
    """Lowercase, strip punctuation, optionally remove stop words."""
    if not isinstance(text, str):
        text = str(text)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if remove_stop_words:
        tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens

def stem(token: str) -> str:
    """Minimal suffix-stripping stemmer (Porter-lite)."""
    suffixes = ["ing", "tion", "ness", "ment", "ity", "ies", "es", "s"]
    for suffix in suffixes:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token

def analyze(text: str) -> list[str]:
    """Full analysis pipeline: tokenize → stem."""
    return [stem(t) for t in tokenize(text)]


# ─────────────────────────────────────────────
#  Inverted Index
# ─────────────────────────────────────────────

class InvertedIndex:
    """Maps token → {doc_id: [positions]} for BM25 scoring."""
    def __init__(self):
        self.index: dict[str, dict[str, list[int]]] = defaultdict(dict)
        self.doc_lengths: dict[str, int] = {}
        self.total_docs = 0
        self.avg_doc_length = 0.0

    def add(self, doc_id: str, tokens: list[str]):
        self.total_docs += 1
        self.doc_lengths[doc_id] = len(tokens)
        self._update_avg()
        positions: dict[str, list[int]] = defaultdict(list)
        for pos, token in enumerate(tokens):
            positions[token].append(pos)
        for token, pos_list in positions.items():
            self.index[token][doc_id] = pos_list

    def remove(self, doc_id: str):
        for token in list(self.index.keys()):
            self.index[token].pop(doc_id, None)
            if not self.index[token]:
                del self.index[token]
        if doc_id in self.doc_lengths:
            del self.doc_lengths[doc_id]
            self.total_docs = max(0, self.total_docs - 1)
            self._update_avg()

    def _update_avg(self):
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def posting(self, token: str) -> dict[str, list[int]]:
        return self.index.get(token, {})

    def df(self, token: str) -> int:
        """Document frequency — number of docs containing token."""
        return len(self.index.get(token, {}))


# ─────────────────────────────────────────────
#  BM25 Scorer
# ─────────────────────────────────────────────

class BM25:
    K1 = 1.5   # term frequency saturation
    B  = 0.75  # length normalisation

    @staticmethod
    def score(tf: int, df: int, total_docs: int,
              doc_len: int, avg_doc_len: float) -> float:
        if total_docs == 0 or df == 0:
            return 0.0
        idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
        norm_tf = (tf * (BM25.K1 + 1)) / (
            tf + BM25.K1 * (1 - BM25.B + BM25.B * doc_len / max(avg_doc_len, 1))
        )
        return idf * norm_tf


# ─────────────────────────────────────────────
#  Index (like an ES index)
# ─────────────────────────────────────────────

class Index:
    MAX_DOCS = 1000

    def __init__(self, name: str, mappings: dict | None = None):
        self.name = name
        self.mappings: dict = mappings or {}
        self.docs: dict[str, dict] = {}          
        self.meta: dict[str, dict] = {}          
        self.inv_index = InvertedIndex()
        self.field_indices: dict[str, dict] = defaultdict(dict)  
        self.created_at = datetime.now(timezone.utc).isoformat()
        
        # HNSW Integrations
        self.hnsw_indices: dict[str, HNSWLiteWrapper] = {}
        self.hnsw_mappings: dict[str, dict] = {}

        if HNSW_AVAILABLE:
            props = self.mappings.get("properties", {})
            for field, cfg in props.items():
                if cfg.get("type") == "dense_vector":
                    dim = cfg.get("dims")
                    if dim:
                        self.hnsw_indices[field] = HNSWLiteWrapper(dim=dim)
                        self.hnsw_mappings[field] = {"doc_to_hnsw": {}, "hnsw_to_doc": {}, "counter": 0}

    # ── helpers ───────────────────────────────

    def _text_fields(self) -> list[str]:
        return [
            f for f, cfg in self.mappings.get("properties", {}).items()
            if cfg.get("type", "text") == "text"
        ]

    def _get_field(self, doc: dict, field: str) -> Any:
        parts = field.split(".")
        val = doc
        for p in parts:
            if not isinstance(val, dict):
                return None
            val = val.get(p)
        return val

    def _extract_text(self, doc: dict) -> list[str]:
        tokens = []
        text_fields = self._text_fields() or list(doc.keys())
        for field in text_fields:
            val = self._get_field(doc, field)
            if isinstance(val, str):
                tokens.extend(analyze(val))
        return tokens

    def _build_field_index(self, doc_id: str, doc: dict):
        props = self.mappings.get("properties", {})
        for field, cfg in props.items():
            val = self._get_field(doc, field)
            if val is None:
                continue
            ftype = cfg.get("type", "text")
            if ftype in ("keyword", "boolean", "float", "integer", "date"):
                self.field_indices[field].setdefault(str(val), [])
                if doc_id not in self.field_indices[field][str(val)]:
                    self.field_indices[field][str(val)].append(doc_id)

    def _remove_field_index(self, doc_id: str, doc: dict):
        props = self.mappings.get("properties", {})
        for field in props:
            val = self._get_field(doc, field)
            if val is None:
                continue
            key = str(val)
            if doc_id in self.field_indices.get(field, {}).get(key, []):
                self.field_indices[field][key].remove(doc_id)

    # ── CRUD ──────────────────────────────────

    def add_document(self, doc: dict, doc_id: str | None = None) -> dict:
        if len(self.docs) >= self.MAX_DOCS:
            raise OverflowError(f"Index '{self.name}' is at the {self.MAX_DOCS}-document limit.")

        doc_id = doc_id or str(uuid.uuid4())
        self.docs[doc_id] = deepcopy(doc)
        self.meta[doc_id] = {
            "_id": doc_id,
            "_index": self.name,
            "_version": 1,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Lexical Indexing
        tokens = self._extract_text(doc)
        self.inv_index.add(doc_id, tokens)
        self._build_field_index(doc_id, doc)
        
        # HNSW Vector Indexing
        if HNSW_AVAILABLE:
            for field, wrapper in self.hnsw_indices.items():
                vec = self._get_field(doc, field)
                if vec is not None:
                    mapping = self.hnsw_mappings[field]
                    hnsw_id = mapping["counter"]
                    mapping["counter"] += 1
                    mapping["doc_to_hnsw"][doc_id] = hnsw_id
                    mapping["hnsw_to_doc"][str(hnsw_id)] = doc_id
                    
                    v = np.array(vec, dtype=np.float32)
                    v = v / max(np.linalg.norm(v), 1e-9)
                    wrapper.add_items(np.array([v]), np.array([hnsw_id]))

        return {"_id": doc_id, "result": "created"}

    def update_document(self, doc_id: str, partial: dict) -> dict:
        if doc_id not in self.docs:
            raise KeyError(f"Document '{doc_id}' not found.")
        old_doc = self.docs[doc_id]
        self._remove_field_index(doc_id, old_doc)
        self.inv_index.remove(doc_id)

        self.docs[doc_id] = {**old_doc, **partial}
        self.meta[doc_id]["_version"] += 1
        
        # Lexical Re-Indexing
        tokens = self._extract_text(self.docs[doc_id])
        self.inv_index.add(doc_id, tokens)
        self._build_field_index(doc_id, self.docs[doc_id])
        
        # HNSW Vector Re-Indexing
        if HNSW_AVAILABLE:
            for field, wrapper in self.hnsw_indices.items():
                vec = self._get_field(self.docs[doc_id], field)
                if vec is not None:
                    mapping = self.hnsw_mappings[field]
                    if doc_id in mapping["doc_to_hnsw"]:
                        hnsw_id = mapping["doc_to_hnsw"][doc_id]
                    else:
                        hnsw_id = mapping["counter"]
                        mapping["counter"] += 1
                        mapping["doc_to_hnsw"][doc_id] = hnsw_id
                        mapping["hnsw_to_doc"][str(hnsw_id)] = doc_id
                    
                    v = np.array(vec, dtype=np.float32)
                    v = v / max(np.linalg.norm(v), 1e-9)
                    wrapper.add_items(np.array([v]), np.array([hnsw_id]))

        return {"_id": doc_id, "result": "updated"}

    def delete_document(self, doc_id: str) -> dict:
        if doc_id not in self.docs:
            raise KeyError(f"Document '{doc_id}' not found.")
            
        # Soft delete handling for lite libraries (de-links it from mapping)
        if HNSW_AVAILABLE:
            for field, wrapper in self.hnsw_indices.items():
                mapping = self.hnsw_mappings[field]
                if doc_id in mapping["doc_to_hnsw"]:
                    hnsw_id = mapping["doc_to_hnsw"][doc_id]
                    del mapping["doc_to_hnsw"][doc_id]
                    if str(hnsw_id) in mapping["hnsw_to_doc"]:
                        del mapping["hnsw_to_doc"][str(hnsw_id)]

        self._remove_field_index(doc_id, self.docs[doc_id])
        self.inv_index.remove(doc_id)
        del self.docs[doc_id]
        del self.meta[doc_id]
        return {"_id": doc_id, "result": "deleted"}

    def get_document(self, doc_id: str) -> dict:
        if doc_id not in self.docs:
            raise KeyError(f"Document '{doc_id}' not found.")
        return {**self.meta[doc_id], "_source": self.docs[doc_id]}

    def bulk(self, documents: list[dict]) -> dict:
        success, failed = 0, []
        for doc in documents:
            doc = deepcopy(doc)
            doc_id = doc.pop("_id", None)
            try:
                self.add_document(doc, doc_id)
                success += 1
            except Exception as e:
                failed.append({"_id": doc_id, "error": str(e)})
        return {"indexed": success, "failed": failed}

    # ── Internal Lexical Engine ──────────────────────────
    def _lexical_search(self, query: dict) -> dict[str, float]:
        scores: dict[str, float] = {}
        qtype = list(query.keys())[0] if query else "match_all"

        if qtype == "match_all":
            for doc_id in self.docs:
                scores[doc_id] = 1.0

        elif qtype == "match":
            field, value = next(iter(query["match"].items()))
            tokens = analyze(str(value))
            for token in tokens:
                for doc_id, positions in self.inv_index.posting(token).items():
                    tf = len(positions)
                    df = self.inv_index.df(token)
                    score = BM25.score(tf, df, self.inv_index.total_docs,
                                       self.inv_index.doc_lengths.get(doc_id, 1),
                                       self.inv_index.avg_doc_length)
                    scores[doc_id] = scores.get(doc_id, 0) + score

        elif qtype == "multi_match":
            cfg = query["multi_match"]
            fields = cfg.get("fields", list(self.docs.values())[0].keys() if self.docs else [])
            tokens = analyze(str(cfg.get("query", "")))
            for token in tokens:
                for doc_id, positions in self.inv_index.posting(token).items():
                    boost = 1.0
                    for f in fields:
                        if "^" in f:
                            fname, b = f.split("^")
                            val = self._get_field(self.docs.get(doc_id, {}), fname)
                            if val and token in analyze(str(val)):
                                boost = max(boost, float(b))
                    tf = len(positions)
                    df = self.inv_index.df(token)
                    score = BM25.score(tf, df, self.inv_index.total_docs,
                                       self.inv_index.doc_lengths.get(doc_id, 1),
                                       self.inv_index.avg_doc_length) * boost
                    scores[doc_id] = scores.get(doc_id, 0) + score

        elif qtype == "term":
            field, value = next(iter(query["term"].items()))
            ids = self.field_indices.get(field, {}).get(str(value), [])
            for doc_id in ids: scores[doc_id] = 1.0

        elif qtype == "terms":
            field, values = next(iter(query["terms"].items()))
            for v in values:
                for doc_id in self.field_indices.get(field, {}).get(str(v), []):
                    scores[doc_id] = 1.0

        elif qtype == "bool":
            bool_q = query["bool"]
            must, filter_ = bool_q.get("must", []), bool_q.get("filter", [])
            should, must_not = bool_q.get("should", []), bool_q.get("must_not", [])
            candidate_sets = []

            for clause in must:
                sub = self.search(clause, size=self.MAX_DOCS)
                ids = {h["_id"] for h in sub["hits"]["hits"]}
                sc  = {h["_id"]: h["_score"] for h in sub["hits"]["hits"]}
                candidate_sets.append((ids, sc))

            for clause in filter_:
                sub = self.search(clause, size=self.MAX_DOCS)
                ids = {h["_id"] for h in sub["hits"]["hits"]}
                candidate_sets.append((ids, {}))

            if candidate_sets:
                common_ids = candidate_sets[0][0]
                for ids, _ in candidate_sets[1:]: common_ids &= ids
            else:
                common_ids = set(self.docs.keys())

            for clause in must_not:
                sub = self.search(clause, size=self.MAX_DOCS)
                common_ids -= {h["_id"] for h in sub["hits"]["hits"]}

            for doc_id in common_ids:
                scores[doc_id] = sum(sc.get(doc_id, 0) for _, sc in candidate_sets)

            for clause in should:
                sub = self.search(clause, size=self.MAX_DOCS)
                for h in sub["hits"]["hits"]:
                    if h["_id"] in scores:
                        scores[h["_id"]] += h["_score"]

        return scores

    # ── Search ────────────────────────────────
    def search(self, query: dict, knn: dict | None = None, size: int = 10,
               from_: int = 0, sort: list | None = None,
               source_fields: list | None = None) -> dict:
        t0 = time.perf_counter()
        
        # 1. Lexical Search
        lex_scores = {}
        qtype = list(query.keys())[0] if query else "match_all"
        is_match_all = (qtype == "match_all")
        
        if not (knn and is_match_all):
            lex_scores = self._lexical_search(query)

        # 2. HNSW Vector Search
        knn_scores = {}
        if knn and HNSW_AVAILABLE:
            field = knn["field"]
            vector = knn["query_vector"]
            k = knn.get("k", size)
            if field in self.hnsw_indices and vector:
                wrapper = self.hnsw_indices[field]
                mapping = self.hnsw_mappings[field]
                element_count = mapping.get("counter", 0)
                
                if element_count > 0:
                    v = np.array(vector, dtype=np.float32)
                    v = v / max(np.linalg.norm(v), 1e-9)
                    
                    try:
                        labels, distances = wrapper.knn_query(np.array([v]), k=min(k, element_count))
                        for label, dist in zip(labels[0], distances[0]):
                            doc_id = mapping["hnsw_to_doc"].get(str(label))
                            if doc_id:
                                # For cosine space, distance returned is usually 1 - cosine_sim
                                # Our distances from HNSW-lite were negated as actual positive distances
                                knn_scores[doc_id] = max(0, 1.0 - float(dist))
                    except Exception as e:
                        pass # Index not ready or unsupported dimension

        # 3. Reciprocal Rank Fusion (RRF) Hybrid Scoring
        scores = {}
        if lex_scores and knn_scores:
            rrf_k = 60
            lex_ranks = {d: r for r, d in enumerate(sorted(lex_scores, key=lex_scores.get, reverse=True))}
            knn_ranks = {d: r for r, d in enumerate(sorted(knn_scores, key=knn_scores.get, reverse=True))}
            
            for d in set(lex_scores) | set(knn_scores):
                s = 0.0
                if d in lex_ranks and lex_scores[d] > 0:
                    s += 1.0 / (rrf_k + lex_ranks[d] + 1)
                if d in knn_ranks:
                    s += 1.0 / (rrf_k + knn_ranks[d] + 1)
                scores[d] = s
        elif knn_scores:
            scores = knn_scores
        else:
            scores = lex_scores

        # 4. Sorting and Pagination
        if sort:
            field_sort, order = None, "asc"
            for s in sort:
                if isinstance(s, dict):
                    field_sort, cfg = next(iter(s.items()))
                    order = cfg.get("order", "asc") if isinstance(cfg, dict) else "asc"
                    break
            ranked = sorted(scores.keys(), key=lambda d: (self._get_field(self.docs[d], field_sort) or 0), reverse=(order == "desc"))
        else:
            ranked = sorted(scores.keys(), key=lambda d: scores.get(d, 0), reverse=True)

        total = len(ranked)
        page  = ranked[from_: from_ + size]

        # 5. Format Output
        hits = []
        for doc_id in page:
            source = deepcopy(self.docs[doc_id])
            if source_fields:
                source = {k: v for k, v in source.items() if k in source_fields}
                
            hit_payload = {
                "_index":  self.name,
                "_id":     doc_id,
                "_score":  round(scores.get(doc_id, 0), 4),
                "_source": source,
            }
            
            # Expose scores for analysis
            if lex_scores and knn_scores:
                hit_payload["_rrf_score"] = hit_payload["_score"]
                hit_payload["_vector_score"] = round(knn_scores.get(doc_id, 0), 4)
                hit_payload["_keyword_score"] = round(lex_scores.get(doc_id, 0), 4)
                
            hits.append(hit_payload)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "took": elapsed_ms,
            "hits": {
                "total": {"value": total, "relation": "eq"},
                "max_score": round(max(scores.values()), 4) if scores else 0,
                "hits": hits,
            },
        }

    # ── Aggregations ──────────────────────────
    def aggregate(self, aggs: dict, base_doc_ids: list | None = None) -> dict:
        doc_ids = base_doc_ids if base_doc_ids is not None else list(self.docs.keys())
        results = {}
        for agg_name, agg_body in aggs.items():
            agg_type = list(agg_body.keys())[0]
            agg_cfg  = agg_body[agg_type]
            
            if agg_type == "terms":
                field = agg_cfg["field"]
                counts: dict[str, int] = defaultdict(int)
                for doc_id in doc_ids:
                    val = self._get_field(self.docs[doc_id], field)
                    if val is not None: counts[str(val)] += 1
                buckets = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:agg_cfg.get("size", 10)]
                results[agg_name] = {"buckets": [{"key": k, "doc_count": v} for k, v in buckets]}
        return results

    # ── Persistence ───────────────────────────
    def save(self, filepath: str):
        data = {
            "name":      self.name,
            "mappings":  self.mappings,
            "docs":      self.docs,
            "meta":      self.meta,
            "created_at": self.created_at,
            "hnsw_mappings": getattr(self, "hnsw_mappings", {})
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
            
        if HNSW_AVAILABLE:
            for field, wrapper in self.hnsw_indices.items():
                try:
                    wrapper.save_index(f"{filepath}.{field}.bin")
                except Exception as e:
                    print(f"Warning: Could not save HNSW graph for field {field}: {e}")

    @classmethod
    def load(cls, filepath: str) -> "Index":
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        idx = cls(data["name"], data.get("mappings", {}))
        idx.created_at = data.get("created_at", "")
        idx.hnsw_mappings = data.get("hnsw_mappings", {})
        
        for doc_id, doc in data["docs"].items():
            idx.docs[doc_id] = doc
            idx.meta[doc_id] = data["meta"][doc_id]
            tokens = idx._extract_text(doc)
            idx.inv_index.add(doc_id, tokens)
            idx._build_field_index(doc_id, doc)
            
        if HNSW_AVAILABLE:
            for field, mapping in idx.hnsw_mappings.items():
                dim = idx.mappings.get("properties", {}).get(field, {}).get("dims")
                bin_path = f"{filepath}.{field}.bin"
                if dim and os.path.exists(bin_path):
                    wrapper = HNSWLiteWrapper(dim=dim)
                    try:
                        wrapper.load_index(bin_path, max_elements=idx.MAX_DOCS)
                        idx.hnsw_indices[field] = wrapper
                    except Exception as e:
                        print(f"Warning: Could not load HNSW graph for field {field}: {e}")
        return idx

    def stats(self) -> dict:
        return {
            "index": self.name,
            "doc_count": len(self.docs),
            "max_docs": self.MAX_DOCS,
            "has_vectors": len(self.hnsw_indices) > 0,
            "created_at": self.created_at,
        }


# ─────────────────────────────────────────────
#  ElastiPy — top-level client (like es.*)
# ─────────────────────────────────────────────

class ElastiPy:
    def __init__(self):
        self._indices: dict[str, Index] = {}
        self.indices = _IndicesNamespace(self)

    def index(self, index: str, body: dict, id: str | None = None) -> dict:
        self._require(index)
        return self._indices[index].add_document(body, id)

    def search(self, index: str, body: dict) -> dict:
        self._require(index)
        idx = self._indices[index]
        query  = body.get("query", {"match_all": {}})
        knn    = body.get("knn")
        size   = body.get("size", 10)
        from_  = body.get("from", 0)
        sort   = body.get("sort")
        source = body.get("_source")

        return idx.search(query=query, knn=knn, size=size, from_=from_, sort=sort, source_fields=source)

    def bulk(self, index: str, documents: list[dict]) -> dict:
        self._require(index)
        return self._indices[index].bulk(documents)

    def save_index(self, index: str, filepath: str):
        self._require(index)
        self._indices[index].save(filepath)

    def load_index(self, filepath: str) -> str:
        idx = Index.load(filepath)
        self._indices[idx.name] = idx
        return idx.name

    def _require(self, index: str):
        if index not in self._indices:
            raise KeyError(f"Index '{index}' does not exist. Create it first.")


class _IndicesNamespace:
    def __init__(self, client: ElastiPy):
        self._client = client

    def create(self, index: str, body: dict | None = None, mappings: dict | None = None) -> dict:
        mappings = mappings or (body or {}).get("mappings", {})
        self._client._indices[index] = Index(index, mappings)
        return {"acknowledged": True, "index": index}

    def exists(self, index: str) -> bool:
        return index in self._client._indices


# ─────────────────────────────────────────────
#  Demo / Proof of Concept
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  ElastiPy — SOTA hnsw-lite Hybrid Search")
    print("=" * 60)

    es = ElastiPy()

    # 1. Create Index with explicitly defined Vector dimensions
    es.indices.create("poc", mappings={
        "properties": {
            "text": {"type": "text"},
            "vector": {"type": "dense_vector", "dims": 3}
        }
    })

    # 2. Add Documents
    es.index("poc", {"text": "Financial tax records keep for 7 years.", "vector": [0.1, 0.9, 0.2]})
    es.index("poc", {"text": "Keep medical documents forever securely.", "vector": [0.8, 0.1, 0.3]})
    es.index("poc", {"text": "Visitor logs are kept for tax audits.", "vector": [0.2, 0.7, 0.8]})

    # 3. Hybrid Search Query
    print("\n🔍 Executing Hybrid Search (Matching 'tax' + semantic math):")
    res = es.search("poc", body={
        "query": {
            "match": {"text": "tax"}  # BM25 Lexical
        },
        "knn": {
            "field": "vector",
            "query_vector": [0.1, 0.85, 0.15], # Semantic Meaning
            "k": 2
        }
    })
    
    # 4. Display Results
    for rank, hit in enumerate(res["hits"]["hits"]):
        print(f"\n[Rank {rank+1}] ID: {hit['_id']}")
        print(f"Content: {hit['_source']['text']}")
        if "_rrf_score" in hit:
            print(f"RRF Fused Score: {hit['_rrf_score']:.4f}")
            print(f" ↳ HNSW Vector Score: {hit['_vector_score']:.4f}")
            print(f" ↳ BM25 Keyword Score: {hit['_keyword_score']:.4f}")
