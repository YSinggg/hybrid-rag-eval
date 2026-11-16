"""Pydantic data models for chunks, retrieval results, API requests, and responses."""

from typing import Literal

from pydantic import BaseModel, Field

RetrievalMode = Literal["hybrid", "bm25", "vector"]


class DocumentChunk(BaseModel):
    """A single indexed text chunk with provenance metadata."""

    chunk_id: str
    doc_id: str
    source: str
    section: str = "Introduction"
    page: int | None = None
    text: str


class SearchResult(BaseModel):
    """A retrieved chunk accompanied by component and fused relevance scores."""

    chunk: DocumentChunk
    bm25_score: float = 0.0
    vector_score: float = 0.0
    rrf_score: float = 0.0
    rank: int


class Citation(BaseModel):
    """A verifiable source reference backing a generated answer."""

    chunk_id: str
    source: str
    section: str
    page: int | None = None
    quote: str


class AskRequest(BaseModel):
    """Request body for querying the RAG pipeline."""

    question: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: RetrievalMode = "hybrid"


class AskResponse(BaseModel):
    """Grounded answer with citations, retrieved evidence, and latency telemetry."""

    answer: str
    citations: list[Citation]
    results: list[SearchResult]
    grounded: bool
    provider: str
    mode: RetrievalMode
    latency_ms: float


class IngestTextRequest(BaseModel):
    """Request body for ingesting raw text or Markdown content."""

    source: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)


class IngestResponse(BaseModel):
    """Summary returned after ingesting one or more documents."""

    documents_added: int
    chunks_added: int


class StatsResponse(BaseModel):
    """Current index and provider status."""

    total_documents: int
    total_chunks: int
    sources: list[str]
    provider: str
    embedding_dim: int
