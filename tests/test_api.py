"""End-to-end API tests using FastAPI TestClient."""

from pathlib import Path

from fastapi.testclient import TestClient

from hybrid_rag.api import create_app
from hybrid_rag.config import Settings


def _make_client() -> TestClient:
    root = Path(__file__).resolve().parent.parent
    settings = Settings(rag_provider="local", sample_docs_dir=root / "data" / "sample_docs")
    app = create_app(settings=settings)
    return TestClient(app)


def test_health_and_stats_endpoints() -> None:
    with _make_client() as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        stats = client.get("/stats")
        assert stats.status_code == 200
        payload = stats.json()
        assert payload["total_documents"] == 3
        assert payload["total_chunks"] > 3


def test_ingest_text_and_query_endpoint() -> None:
    with _make_client() as client:
        ingest = client.post(
            "/ingest/text",
            json={
                "source": "backup_policy.md",
                "content": (
                    "# Backups\n\n"
                    "Automated snapshots run every 6 hours and retain for 35 days."
                ),
            },
        )
        assert ingest.status_code == 200
        assert ingest.json()["documents_added"] == 1

        query = client.post(
            "/query",
            json={"question": "How long are automated snapshots retained?", "mode": "hybrid"},
        )
        assert query.status_code == 200
        data = query.json()
        assert data["grounded"] is True
        assert "35 days" in data["answer"]
        assert any(c["source"] == "backup_policy.md" for c in data["citations"])
