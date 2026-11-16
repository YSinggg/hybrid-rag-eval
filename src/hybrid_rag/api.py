"""FastAPI web service and interactive playground for Hybrid-RAG."""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from hybrid_rag.config import Settings, get_settings
from hybrid_rag.models import (
    AskRequest,
    AskResponse,
    IngestResponse,
    IngestTextRequest,
    StatsResponse,
)
from hybrid_rag.pipeline import RagPipeline

_HTML_PLAYGROUND = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hybrid-RAG Evaluation Playground</title>
  <style>
    body {
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 900px;
      margin: 2rem auto;
      padding: 0 1rem;
      color: #1f2937;
      background: #f9fafb;
    }
    h1 { margin-bottom: 0.25rem; }
    .sub { color: #4b5563; margin-bottom: 1.5rem; }
    .card {
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 10px;
      padding: 1.25rem;
      margin-bottom: 1rem;
      box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    textarea, select, input[type="file"] {
      width: 100%;
      box-sizing: border-box;
      padding: 0.65rem;
      border: 1px solid #d1d5db;
      border-radius: 8px;
      font: inherit;
    }
    .row { display: flex; gap: 0.75rem; margin-top: 0.75rem; align-items: center; }
    button {
      background: #111827;
      color: #fff;
      border: 0;
      border-radius: 8px;
      padding: 0.65rem 1.1rem;
      cursor: pointer;
      font-weight: 600;
    }
    button:hover { background: #374151; }
    .badge {
      display: inline-block;
      font-size: 0.78rem;
      padding: 0.2rem 0.55rem;
      border-radius: 999px;
      background: #e5e7eb;
      margin-right: 0.4rem;
    }
    .grounded { background: #d1fae5; color: #065f46; }
    .ungrounded { background: #fee2e2; color: #991b1b; }
    .cite {
      border-left: 3px solid #2563eb;
      padding-left: 0.75rem;
      margin-top: 0.65rem;
      font-size: 0.92rem;
    }
    pre { white-space: pre-wrap; margin: 0.25rem 0 0; color: #374151; font-family: inherit; }
  </style>
</head>
<body>
  <h1>Evaluated Hybrid-RAG Playground</h1>
  <p class="sub">BM25 + Dense Vector + Reciprocal Rank Fusion (RRF) with grounded citations.</p>

  <div class="card">
    <strong>Ask a question</strong>
    <textarea id="q" rows="3" placeholder="e.g. How should clients handle ERR_RATE_429?"></textarea>
    <div class="row">
      <select id="mode" style="max-width: 180px;">
        <option value="hybrid">hybrid (BM25 + Vector)</option>
        <option value="bm25">bm25 only</option>
        <option value="vector">vector only</option>
      </select>
      <button onclick="ask()">Ask</button>
      <span id="stats" style="font-size: 0.85rem; color: #6b7280;"></span>
    </div>
  </div>

  <div id="output" class="card" style="display:none;"></div>

  <div class="card">
    <strong>Upload a document (.md, .txt, .pdf)</strong>
    <div class="row">
      <input type="file" id="fileInput" accept=".md,.markdown,.txt,.pdf" />
      <button onclick="uploadFile()">Ingest</button>
    </div>
    <p id="uploadMsg" style="font-size:0.85rem;color:#4b5563;margin:0.5rem 0 0;"></p>
  </div>

  <script>
    async function loadStats() {
      const res = await fetch('/stats');
      const s = await res.json();
      document.getElementById('stats').textContent =
        `${s.total_documents} docs · ${s.total_chunks} chunks · ${s.provider}`;
    }
    async function ask() {
      const question = document.getElementById('q').value.trim();
      const mode = document.getElementById('mode').value;
      if (!question) return;
      const out = document.getElementById('output');
      out.style.display = 'block';
      out.innerHTML = 'Searching and synthesizing...';
      const res = await fetch('/query', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question, mode})
      });
      const data = await res.json();
      const badgeClass = data.grounded ? 'grounded' : 'ungrounded';
      const cites = (data.citations || []).map(c =>
        `<div class="cite"><strong>${c.source}</strong> · ${c.section} ` +
        `(${c.chunk_id})<pre>${c.quote}</pre></div>`
      ).join('');
      out.innerHTML = `
        <div>
          <span class="badge ${badgeClass}">${data.grounded ? 'Grounded' : 'Abstained'}</span>
          <span class="badge">${data.mode}</span>
          <span class="badge">${data.latency_ms} ms</span>
        </div>
        <p style="line-height:1.5;">${data.answer}</p>
        ${cites ? '<strong>Citations</strong>' + cites : ''}
      `;
    }
    async function uploadFile() {
      const input = document.getElementById('fileInput');
      if (!input.files.length) return;
      const form = new FormData();
      form.append('file', input.files[0]);
      const res = await fetch('/ingest/file', {method: 'POST', body: form});
      const data = await res.json();
      document.getElementById('uploadMsg').textContent =
        res.ok ? `Added ${data.documents_added} doc (${data.chunks_added} chunks).` : data.detail;
      loadStats();
    }
    loadStats();
  </script>
</body>
</html>
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    resolved_settings = settings or get_settings()
    pipeline = RagPipeline(settings=resolved_settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        pipeline.ingest_directory(resolved_settings.sample_docs_dir)
        yield

    app = FastAPI(
        title="Evaluated Hybrid-RAG Service",
        version="0.1.0",
        description="Hybrid BM25 + Vector + RRF document Q&A API with automated evaluation.",
        lifespan=lifespan,
    )
    app.state.pipeline = pipeline

    @app.get("/", response_class=HTMLResponse)
    def playground() -> str:
        return _HTML_PLAYGROUND

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        return pipeline.stats()

    @app.post("/ingest/text", response_model=IngestResponse)
    def ingest_text(payload: IngestTextRequest) -> IngestResponse:
        return pipeline.ingest_text(source=payload.source, content=payload.content)

    @app.post("/ingest/file", response_model=IngestResponse)
    async def ingest_file(file: UploadFile = File(...)) -> IngestResponse:  # noqa: B008
        filename = file.filename or "upload.txt"
        suffix = Path(filename).suffix.lower()
        if suffix not in {".md", ".markdown", ".txt", ".pdf"}:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension '{suffix}'. Allowed: .md, .txt, .pdf",
            )
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(raw)
            tmp_path = Path(tmp.name)
        try:
            return pipeline.ingest_file(tmp_path, source_name=filename)
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.post("/query", response_model=AskResponse)
    def query(payload: AskRequest) -> AskResponse:
        return pipeline.ask(payload.question, top_k=payload.top_k, mode=payload.mode)

    return app


app = create_app()
