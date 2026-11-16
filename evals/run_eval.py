"""Automated evaluation runner comparing BM25, Vector, and Hybrid (RRF) retrieval."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hybrid_rag.config import Settings
from hybrid_rag.models import RetrievalMode
from hybrid_rag.pipeline import RagPipeline


def evaluate_mode(
    pipeline: RagPipeline,
    cases: list[dict],
    mode: RetrievalMode,
    top_k: int = 4,
) -> dict:
    """Evaluate retrieval Hit@K, MRR, keyword recall, and abstention accuracy."""
    answerable = [c for c in cases if c["expected_grounded"]]
    hits = 0
    mrr_total = 0.0
    keyword_recall_total = 0.0
    grounding_correct = 0
    latencies: list[float] = []

    for case in cases:
        response = pipeline.ask(case["question"], top_k=top_k, mode=mode)
        latencies.append(response.latency_ms)

        if response.grounded == case["expected_grounded"]:
            grounding_correct += 1

        if case["expected_grounded"]:
            sources = [r.chunk.source for r in response.results]
            expected_source = case["expected_source"]
            if expected_source in sources:
                hits += 1
                rank = sources.index(expected_source) + 1
                mrr_total += 1.0 / rank

            expected_kw = case["expected_keywords"]
            if expected_kw:
                answer_lower = response.answer.lower()
                matched = sum(1 for kw in expected_kw if kw.lower() in answer_lower)
                keyword_recall_total += matched / len(expected_kw)

    num_answerable = max(1, len(answerable))
    num_total = max(1, len(cases))
    return {
        "mode": mode,
        "cases": len(cases),
        f"hit_at_{top_k}": round(hits / num_answerable, 3),
        "mrr": round(mrr_total / num_answerable, 3),
        "keyword_recall": round(keyword_recall_total / num_answerable, 3),
        "grounding_accuracy": round(grounding_correct / num_total, 3),
        "avg_latency_ms": round(sum(latencies) / num_total, 2),
    }


def main() -> int:
    """Run comparison across retrieval modes and print a Markdown summary table."""
    root = Path(__file__).resolve().parent.parent
    dataset_path = root / "evals" / "dataset.json"
    cases = json.loads(dataset_path.read_text(encoding="utf-8"))

    settings = Settings(sample_docs_dir=root / "data" / "sample_docs")
    pipeline = RagPipeline(settings=settings)
    ingest_summary = pipeline.ingest_directory(settings.sample_docs_dir)

    print(
        f"Indexed {ingest_summary.documents_added} docs "
        f"({ingest_summary.chunks_added} chunks) using {pipeline.stats().provider}\n"
    )

    modes: list[RetrievalMode] = ["bm25", "vector", "hybrid"]
    reports = [evaluate_mode(pipeline, cases, mode=m, top_k=4) for m in modes]

    print("| Mode | Hit@4 | MRR | Keyword Recall | Grounding Acc | Avg Latency (ms) |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in reports:
        print(
            f"| {row['mode']} | {row['hit_at_4']:.3f} | {row['mrr']:.3f} | "
            f"{row['keyword_recall']:.3f} | {row['grounding_accuracy']:.3f} | "
            f"{row['avg_latency_ms']:.2f} |"
        )

    out_path = root / "evals" / "latest_report.json"
    out_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
