"""End-to-end RAG orchestration: ingestion, retrieval, citation mapping, and telemetry."""

from __future__ import annotations

import time
from pathlib import Path

from hybrid_rag.chunking import chunk_text, load_text_from_file, make_doc_id
from hybrid_rag.config import Settings
from hybrid_rag.models import (
    AskResponse,
    Citation,
    DocumentChunk,
    IngestResponse,
    RetrievalMode,
    SearchResult,
    StatsResponse,
)
from hybrid_rag.providers import EmbeddingProvider, LLMProvider, build_providers
from hybrid_rag.retriever import HybridRetriever


class RagPipeline:
    """Coordinates document ingestion, hybrid search, and grounded answer generation."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: EmbeddingProvider | None = None,
        llm: LLMProvider | None = None,
    ) -> None:
        self.settings = settings or Settings()
        default_embedder, default_llm = build_providers(self.settings)
        self.embedder = embedder or default_embedder
        self.llm = llm or default_llm
        self.retriever = HybridRetriever(embedder=self.embedder, rrf_k=self.settings.rrf_k)
        self._doc_sources: dict[str, str] = {}

    def ingest_text(self, *, source: str, content: str) -> IngestResponse:
        """Chunk and index raw text or Markdown content."""
        doc_id = make_doc_id(source, content)
        if doc_id in self._doc_sources:
            return IngestResponse(documents_added=0, chunks_added=0)

        chunks = chunk_text(
            content,
            source=source,
            doc_id=doc_id,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.retriever.add_chunks(chunks)
        self._doc_sources[doc_id] = source
        return IngestResponse(documents_added=1, chunks_added=len(chunks))

    def ingest_file(self, path: Path, *, source_name: str | None = None) -> IngestResponse:
        """Load, chunk, and index a .md, .txt, or .pdf file."""
        source = source_name or path.name
        pages = load_text_from_file(path)
        combined = "\n".join(text for _, text in pages)
        doc_id = make_doc_id(source, combined)
        if doc_id in self._doc_sources:
            return IngestResponse(documents_added=0, chunks_added=0)

        all_chunks: list[DocumentChunk] = []
        for page_num, text in pages:
            all_chunks.extend(
                chunk_text(
                    text,
                    source=source,
                    doc_id=doc_id,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                    page=page_num,
                )
            )
        self.retriever.add_chunks(all_chunks)
        self._doc_sources[doc_id] = source
        return IngestResponse(documents_added=1, chunks_added=len(all_chunks))

    def ingest_directory(self, directory: Path) -> IngestResponse:
        """Recursively ingest all supported files in a directory."""
        if not directory.exists():
            return IngestResponse(documents_added=0, chunks_added=0)

        docs_added = 0
        chunks_added = 0
        supported = {".md", ".markdown", ".txt", ".pdf"}
        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in supported:
                result = self.ingest_file(file_path, source_name=file_path.name)
                docs_added += result.documents_added
                chunks_added += result.chunks_added
        return IngestResponse(documents_added=docs_added, chunks_added=chunks_added)

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        mode: RetrievalMode = "hybrid",
    ) -> list[SearchResult]:
        """Run retrieval only without synthesis (useful for retrieval-level evals)."""
        return self.retriever.search(
            question,
            top_k=top_k or self.settings.top_k,
            mode=mode,
        )

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        mode: RetrievalMode = "hybrid",
    ) -> AskResponse:
        """Retrieve relevant chunks and synthesize a grounded answer with citations."""
        start = time.perf_counter()
        results = self.retrieve(question, top_k=top_k, mode=mode)
        relevant = [r for r in results if r.rrf_score >= self.settings.min_relevance_score]

        answer, cited_ids, grounded = self.llm.generate_grounded_answer(question, relevant)
        chunk_lookup = {r.chunk.chunk_id: r.chunk for r in relevant}
        citations = [
            Citation(
                chunk_id=cid,
                source=chunk_lookup[cid].source,
                section=chunk_lookup[cid].section,
                page=chunk_lookup[cid].page,
                quote=chunk_lookup[cid].text[:240].strip(),
            )
            for cid in cited_ids
            if cid in chunk_lookup
        ]
        latency_ms = round((time.perf_counter() - start) * 1000.0, 2)

        return AskResponse(
            answer=answer,
            citations=citations,
            results=results,
            grounded=grounded,
            provider=f"{self.embedder.name}+{self.llm.name}",
            mode=mode,
            latency_ms=latency_ms,
        )

    def stats(self) -> StatsResponse:
        """Return index statistics and active provider metadata."""
        return StatsResponse(
            total_documents=len(self._doc_sources),
            total_chunks=len(self.retriever.chunks),
            sources=sorted(set(self._doc_sources.values())),
            provider=f"{self.embedder.name}+{self.llm.name}",
            embedding_dim=self.embedder.dimension,
        )
