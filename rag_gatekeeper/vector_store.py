"""Native in-memory hybrid search vector store with Reciprocal Rank Fusion (RRF).

Enables RAG-Gatekeeper to operate as an all-in-one autonomous RAG engine without
mandatory external vector database dependencies.
"""

import math
import re
from typing import Optional
from collections import Counter
from rag_gatekeeper.models import ChunkItem


class SimpleBM25:
    """Lightweight pure-python BM25 scoring implementation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_freqs: list[dict[str, int]] = []
        self.idf: dict[str, float] = {}
        self.doc_lens: list[int] = []

    def _tokenize(self, text: str) -> list[str]:
        # Simple word & character tokenization for cross-language compatibility
        return re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", text.lower())

    def fit(self, docs: list[str]):
        self.corpus_size = len(docs)
        if self.corpus_size == 0:
            return
        total_len = 0
        df: dict[str, int] = Counter()
        self.doc_freqs = []
        self.doc_lens = []

        for doc in docs:
            tokens = self._tokenize(doc)
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total_len += doc_len
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            for word in freqs:
                df[word] += 1

        self.avg_doc_len = total_len / self.corpus_size
        self.idf = {}
        for word, freq in df.items():
            self.idf[word] = math.log(1.0 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query: str) -> list[float]:
        q_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size
        for idx in range(self.corpus_size):
            doc_len = self.doc_lens[idx]
            freqs = self.doc_freqs[idx]
            score = 0.0
            for token in q_tokens:
                if token not in freqs:
                    continue
                tf = freqs[token]
                idf = self.idf.get(token, 0.0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                score += idf * (numerator / denominator)
            scores[idx] = score
        return scores


class InMemoryHybridStore:
    """Hybrid (BM25 + Semantic overlap) in-memory document store with RRF ranking."""

    def __init__(self):
        self._chunks: list[ChunkItem] = []
        self._bm25 = SimpleBM25()
        self._fitted = False

    def upsert_chunks(self, chunks: list[ChunkItem]):
        """Add or update chunks in the store."""
        existing_ids = {c.id for c in self._chunks}
        for chunk in chunks:
            if chunk.id not in existing_ids:
                self._chunks.append(chunk)
                existing_ids.add(chunk.id)
        # Refit BM25
        self._bm25.fit([c.content for c in self._chunks])
        self._fitted = True

    def clear(self):
        self._chunks.clear()
        self._fitted = False

    def search(self, query: str, top_k: int = 10) -> list[ChunkItem]:
        """Perform hybrid search with Reciprocal Rank Fusion (RRF)."""
        if not self._chunks or not self._fitted:
            return []

        # 1. Sparse BM25 retrieval ranking
        bm25_scores = self._bm25.get_scores(query)
        bm25_ranked = sorted(range(len(self._chunks)), key=lambda i: bm25_scores[i], reverse=True)

        # 2. Simulated Dense / Token overlap ranking
        def overlap_score(idx: int) -> float:
            q_set = set(re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", query.lower()))
            c_set = set(re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", self._chunks[idx].content.lower()))
            if not q_set:
                return 0.0
            return len(q_set & c_set) / len(q_set)

        dense_scores = [overlap_score(i) for i in range(len(self._chunks))]
        dense_ranked = sorted(range(len(self._chunks)), key=lambda i: dense_scores[i], reverse=True)

        # 3. Reciprocal Rank Fusion (RRF) with constant k=60
        k = 60
        rrf_scores: dict[int, float] = Counter()
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[idx] += 1.0 / (k + rank + 1)
        for rank, idx in enumerate(dense_ranked):
            rrf_scores[idx] += 1.0 / (k + rank + 1)

        # Sort by final RRF score
        final_ranked = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)
        return [self._chunks[i] for i in final_ranked[:top_k]]
