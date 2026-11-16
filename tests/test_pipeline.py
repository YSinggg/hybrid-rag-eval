"""Integration tests for RagPipeline ingestion, citations, and abstention."""

from pathlib import Path

from hybrid_rag.config import Settings
from hybrid_rag.pipeline import RagPipeline


def _build_sample_pipeline() -> RagPipeline:
    root = Path(__file__).resolve().parent.parent
    settings = Settings(rag_provider="local", sample_docs_dir=root / "data" / "sample_docs")
    pipeline = RagPipeline(settings=settings)
    pipeline.ingest_directory(settings.sample_docs_dir)
    return pipeline


def test_pipeline_answers_with_verified_citations() -> None:
    pipeline = _build_sample_pipeline()
    res = pipeline.ask("How often must production API keys be rotated?")
    assert res.grounded is True
    assert "90 days" in res.answer
    assert res.citations
    assert res.citations[0].source == "api_rate_limits.md"


def test_pipeline_abstains_on_out_of_scope_question() -> None:
    pipeline = _build_sample_pipeline()
    res = pipeline.ask("What is the dental insurance deductible for contractors?")
    assert res.grounded is False
    assert res.citations == []
