"""Unit tests for Markdown section parsing and overlapping text chunking."""

from pathlib import Path

from hybrid_rag.chunking import chunk_text, load_text_from_file


def test_chunk_text_tracks_markdown_sections() -> None:
    doc = (
        "# Service Guide\n\n"
        "Overview paragraph.\n\n"
        "## Error Codes\n\n"
        "ERR_RATE_429 indicates the organization exceeded its per-minute quota.\n"
    )
    chunks = chunk_text(doc, source="guide.md", chunk_size=200, chunk_overlap=30)
    assert len(chunks) == 2
    assert chunks[0].section == "Service Guide"
    assert chunks[1].section == "Error Codes"
    assert "ERR_RATE_429" in chunks[1].text


def test_chunk_text_splits_oversized_paragraph_with_overlap() -> None:
    long_para = " ".join([f"token{i}" for i in range(120)])
    chunks = chunk_text(long_para, source="long.txt", chunk_size=120, chunk_overlap=25)
    assert len(chunks) > 2
    assert all(len(c.text) <= 125 for c in chunks)


def test_load_text_from_markdown_file(tmp_path: Path) -> None:
    sample = tmp_path / "notes.md"
    sample.write_text("# Title\n\nBody text.", encoding="utf-8")
    pages = load_text_from_file(sample)
    assert pages == [(None, "# Title\n\nBody text.")]
