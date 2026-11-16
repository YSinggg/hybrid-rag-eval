"""Document loading and heading-aware overlapping text chunking."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pypdf import PdfReader

from hybrid_rag.models import DocumentChunk

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_WHITESPACE_RE = re.compile(r"\n{3,}")


def make_doc_id(source: str, content: str) -> str:
    """Create a short deterministic document identifier from source name and content."""
    digest = hashlib.sha256(f"{source}\n{content}".encode()).hexdigest()
    return digest[:12]


def load_text_from_file(path: Path) -> list[tuple[int | None, str]]:
    """Extract (page_number, text) segments from Markdown, plain text, or PDF files."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return [(None, path.read_text(encoding="utf-8"))]
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages: list[tuple[int | None, str]] = []
        for index, page in enumerate(reader.pages, start=1):
            extracted = (page.extract_text() or "").strip()
            if extracted:
                pages.append((index, extracted))
        return pages
    raise ValueError(f"Unsupported file type '{suffix}'. Supported: .md, .txt, .pdf")


def _split_long_block(block: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split an oversized paragraph on word boundaries while preserving overlap."""
    if len(block) <= chunk_size:
        return [block]

    slices: list[str] = []
    start = 0
    step_floor = max(1, chunk_size - chunk_overlap)

    while start < len(block):
        end = min(len(block), start + chunk_size)
        if end < len(block):
            split_at = block.rfind(" ", start + step_floor // 2, end)
            if split_at > start:
                end = split_at
        piece = block[start:end].strip()
        if piece:
            slices.append(piece)
        if end >= len(block):
            break
        start = max(start + 1, end - chunk_overlap)
    return slices


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Group Markdown paragraphs under their nearest heading."""
    normalized = _WHITESPACE_RE.sub("\n\n", text.replace("\r\n", "\n")).strip()
    if not normalized:
        return []

    sections: list[tuple[str, list[str]]] = []
    current_heading = "Introduction"
    current_lines: list[str] = []

    for line in normalized.split("\n"):
        heading_match = _HEADING_RE.match(line)
        if heading_match:
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_heading, [body]))
                current_lines = []
            current_heading = heading_match.group(2).strip()
        else:
            current_lines.append(line)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, [body]))

    return [(heading, "\n".join(parts)) for heading, parts in sections]


def chunk_text(
    text: str,
    *,
    source: str,
    doc_id: str | None = None,
    chunk_size: int = 450,
    chunk_overlap: int = 80,
    page: int | None = None,
) -> list[DocumentChunk]:
    """Chunk text into overlapping, section-tagged DocumentChunks."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    resolved_doc_id = doc_id or make_doc_id(source, text)
    sections = _split_into_sections(text)
    if not sections:
        return []

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for section_title, section_body in sections:
        paragraphs = [p.strip() for p in section_body.split("\n\n") if p.strip()]
        buffer = ""

        for paragraph in paragraphs:
            if len(paragraph) > chunk_size:
                if buffer:
                    chunks.append(
                        _make_chunk(
                            resolved_doc_id, chunk_index, source, section_title, page, buffer
                        )
                    )
                    chunk_index += 1
                    buffer = ""
                for piece in _split_long_block(paragraph, chunk_size, chunk_overlap):
                    chunks.append(
                        _make_chunk(
                            resolved_doc_id, chunk_index, source, section_title, page, piece
                        )
                    )
                    chunk_index += 1
                continue

            candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            if len(candidate) <= chunk_size:
                buffer = candidate
            else:
                chunks.append(
                    _make_chunk(resolved_doc_id, chunk_index, source, section_title, page, buffer)
                )
                chunk_index += 1
                tail = buffer[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
                combined_tail = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
                if len(combined_tail) <= chunk_size:
                    buffer = combined_tail
                else:
                    buffer = paragraph

        if buffer:
            chunks.append(
                _make_chunk(resolved_doc_id, chunk_index, source, section_title, page, buffer)
            )
            chunk_index += 1

    return chunks


def _make_chunk(
    doc_id: str,
    index: int,
    source: str,
    section: str,
    page: int | None,
    text: str,
) -> DocumentChunk:
    page_tag = f"p{page}-" if page is not None else ""
    return DocumentChunk(
        chunk_id=f"{doc_id}-{page_tag}c{index:03d}",
        doc_id=doc_id,
        source=source,
        section=section,
        page=page,
        text=text,
    )
