"""Unit tests for BM25, dense vector, and Reciprocal Rank Fusion retrieval."""

from hybrid_rag.models import DocumentChunk
from hybrid_rag.providers import LocalHashingEmbedder
from hybrid_rag.retriever import HybridRetriever, tokenize


def test_tokenize_preserves_error_codes_and_subtokens() -> None:
    tokens = tokenize("Fix ERR_RATE_429 and SEV-1 outages")
    assert "err_rate_429" in tokens
    assert "rate" in tokens
    assert "429" in tokens
    assert "sev-1" in tokens


def test_hybrid_retriever_ranks_relevant_chunk_first() -> None:
    retriever = HybridRetriever(embedder=LocalHashingEmbedder(), rrf_k=60)
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            source="rate.md",
            section="Throttling",
            text="Gateway returns ERR_RATE_429 with a Retry-After header.",
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="d2",
            source="db.md",
            section="Postgres",
            text="Check pg_stat_activity when connections saturate.",
        ),
    ]
    retriever.add_chunks(chunks)

    for mode in ("bm25", "vector", "hybrid"):
        results = retriever.search("How to handle ERR_RATE_429?", top_k=2, mode=mode)  # type: ignore[arg-type]
        assert results
        assert results[0].chunk.chunk_id == "c1"
