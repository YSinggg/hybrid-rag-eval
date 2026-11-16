"""Hybrid retrieval engine combining Okapi BM25, dense cosine search, and RRF."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from hybrid_rag.models import DocumentChunk, RetrievalMode, SearchResult
from hybrid_rag.providers import EmbeddingProvider

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
_SUBTOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text while preserving compound error/severity identifiers and sub-tokens."""
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        tokens.append(match)
        if "_" in match or "-" in match:
            parts = _SUBTOKEN_RE.findall(match)
            if len(parts) > 1:
                tokens.extend(parts)
    return tokens


class BM25Index:
    """In-memory Okapi BM25 lexical index."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._doc_term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._doc_freqs: Counter[str] = Counter()
        self._avg_doc_len: float = 0.0

    def add_texts(self, texts: list[str]) -> None:
        """Index new chunk texts incrementally."""
        for text in texts:
            tokens = tokenize(text)
            tf = Counter(tokens)
            self._doc_term_freqs.append(tf)
            self._doc_lengths.append(len(tokens))
            for term in tf:
                self._doc_freqs[term] += 1
        total_docs = len(self._doc_lengths)
        self._avg_doc_len = sum(self._doc_lengths) / total_docs if total_docs else 0.0

    def score_all(self, query: str) -> list[float]:
        """Compute BM25 scores for all indexed chunks against query."""
        num_docs = len(self._doc_term_freqs)
        if num_docs == 0 or self._avg_doc_len == 0.0:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return [0.0] * num_docs

        scores = [0.0] * num_docs
        for term in query_terms:
            df = self._doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))
            for idx, tf_map in enumerate(self._doc_term_freqs):
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                dl = self._doc_lengths[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self._avg_doc_len))
                scores[idx] += idf * ((tf * (self.k1 + 1.0)) / denom)
        return scores


class VectorIndex:
    """In-memory L2-normalized dense vector index using cosine similarity."""

    def __init__(self, embedder: EmbeddingProvider) -> None:
        self.embedder = embedder
        self._matrix: np.ndarray | None = None

    def add_texts(self, texts: list[str]) -> None:
        """Embed and append new chunk texts to the vector matrix."""
        if not texts:
            return
        embeddings = self.embedder.embed_texts(texts)
        if self._matrix is None:
            self._matrix = embeddings
        else:
            self._matrix = np.vstack([self._matrix, embeddings])

    def score_all(self, query: str) -> list[float]:
        """Compute cosine similarity between query and all indexed vectors."""
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        query_vec = self.embedder.embed_texts([query])[0]
        sims = self._matrix @ query_vec
        return [float(max(0.0, score)) for score in sims]


class HybridRetriever:
    """Combines BM25 and dense vector search using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, embedder: EmbeddingProvider, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k
        self.chunks: list[DocumentChunk] = []
        self._bm25 = BM25Index()
        self._vector = VectorIndex(embedder)

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add DocumentChunks to both lexical and vector indices."""
        if not chunks:
            return
        texts = [f"{chunk.section}\n{chunk.text}" for chunk in chunks]
        self.chunks.extend(chunks)
        self._bm25.add_texts(texts)
        self._vector.add_texts(texts)

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        mode: RetrievalMode = "hybrid",
    ) -> list[SearchResult]:
        """Retrieve top-k chunks under 'hybrid', 'bm25', or 'vector' mode."""
        if not self.chunks or not query.strip():
            return []

        bm25_scores = self._bm25.score_all(query)
        vector_scores = self._vector.score_all(query)

        if mode == "bm25":
            ranked_indices = [
                idx
                for idx in sorted(
                    range(len(self.chunks)), key=lambda i: bm25_scores[i], reverse=True
                )
                if bm25_scores[idx] > 0.0
            ][:top_k]
            return [
                SearchResult(
                    chunk=self.chunks[idx],
                    bm25_score=round(bm25_scores[idx], 4),
                    vector_score=round(vector_scores[idx], 4),
                    rrf_score=round(bm25_scores[idx], 4),
                    rank=rank,
                )
                for rank, idx in enumerate(ranked_indices, start=1)
            ]

        if mode == "vector":
            ranked_indices = [
                idx
                for idx in sorted(
                    range(len(self.chunks)), key=lambda i: vector_scores[i], reverse=True
                )
                if vector_scores[idx] > 0.05
            ][:top_k]
            return [
                SearchResult(
                    chunk=self.chunks[idx],
                    bm25_score=round(bm25_scores[idx], 4),
                    vector_score=round(vector_scores[idx], 4),
                    rrf_score=round(vector_scores[idx], 4),
                    rank=rank,
                )
                for rank, idx in enumerate(ranked_indices, start=1)
            ]

        # Hybrid mode: Reciprocal Rank Fusion over non-trivial candidates from each retriever.
        candidate_depth = max(top_k * 3, 10)
        bm25_ranked = [
            idx
            for idx in sorted(range(len(self.chunks)), key=lambda i: bm25_scores[i], reverse=True)
            if bm25_scores[idx] > 0.0
        ][:candidate_depth]
        vector_ranked = [
            idx
            for idx in sorted(range(len(self.chunks)), key=lambda i: vector_scores[i], reverse=True)
            if vector_scores[idx] > 0.05
        ][:candidate_depth]

        rrf_scores: dict[int, float] = {}
        for rank, idx in enumerate(bm25_ranked, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, idx in enumerate(vector_ranked, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        fused_indices = sorted(
            rrf_scores,
            key=lambda i: (rrf_scores[i], bm25_scores[i] + vector_scores[i]),
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                chunk=self.chunks[idx],
                bm25_score=round(bm25_scores[idx], 4),
                vector_score=round(vector_scores[idx], 4),
                rrf_score=round(rrf_scores[idx], 6),
                rank=rank,
            )
            for rank, idx in enumerate(fused_indices, start=1)
        ]
