"""Pluggable embedding and grounded answer providers (Local offline + Google Gemini)."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from hybrid_rag.config import Settings
from hybrid_rag.models import SearchResult

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "must",
        "of",
        "on",
        "or",
        "our",
        "should",
        "that",
        "the",
        "their",
        "there",
        "these",
        "this",
        "to",
        "we",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "you",
        "your",
    }
)

_ABSTENTION_MSG = (
    "I could not find enough supporting evidence in the indexed documents to answer that question."
)


class EmbeddingProvider(Protocol):
    """Interface for text embedding backends."""

    name: str
    dimension: int

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return an L2-normalized float32 matrix of shape (len(texts), dimension)."""


class LLMProvider(Protocol):
    """Interface for grounded answer generation backends."""

    name: str

    def generate_grounded_answer(
        self, question: str, contexts: list[SearchResult]
    ) -> tuple[str, list[str], bool]:
        """Return (answer, cited_chunk_ids, grounded)."""


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / safe_norms).astype(np.float32)


class LocalHashingEmbedder:
    """Deterministic offline embedder combining word, bigram, and character n-gram hashing."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.name = "local-hashing-384d"

    def _feature_tokens(self, text: str) -> list[tuple[str, float]]:
        words = re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower())
        features: list[tuple[str, float]] = []
        for word in words:
            weight = 0.35 if word in _STOPWORDS else 1.0
            features.append((f"w:{word}", weight))
            if len(word) >= 4:
                padded = f"<{word}>"
                for size in (3, 4):
                    for i in range(len(padded) - size + 1):
                        features.append((f"c{size}:{padded[i : i + size]}", 0.35))
        for left, right in zip(words, words[1:], strict=False):
            features.append((f"b:{left}_{right}", 0.8))
        return features

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Project texts into a normalized fixed-dimension vector space."""
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            for token, weight in self._feature_tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                raw = int.from_bytes(digest, "little")
                bucket = raw % self.dimension
                sign = 1.0 if ((raw >> 17) & 1) == 0 else -1.0
                matrix[row_idx, bucket] += sign * weight
        return _l2_normalize(matrix)


def _content_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in re.findall(r"[a-z0-9][a-z0-9_-]*", text.lower()):
        if token in _STOPWORDS:
            continue
        terms.add(token)
        if "_" in token or "-" in token:
            for part in re.findall(r"[a-z0-9]+", token):
                if part not in _STOPWORDS:
                    terms.add(part)
    return terms


def _split_candidate_sentences(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    sentences: list[str] = []
    for line in lines:
        cleaned = re.sub(r"^[-*•]\s+", "", line)
        cleaned = re.sub(r"^\d+\.\s+", "", cleaned)
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`])", cleaned)
        for part in parts:
            stripped = part.strip()
            if len(stripped) >= 15:
                sentences.append(stripped)
    return sentences


class LocalExtractiveLLM:
    """Zero-dependency extractive answer synthesizer with strict citation & abstention checks."""

    def __init__(self) -> None:
        self.name = "local-extractive"

    def generate_grounded_answer(
        self, question: str, contexts: list[SearchResult]
    ) -> tuple[str, list[str], bool]:
        """Select top supporting sentences from retrieved chunks or abstain if unsupported."""
        if not contexts:
            return _ABSTENTION_MSG, [], False

        query_terms = _content_terms(question)
        if not query_terms:
            return _ABSTENTION_MSG, [], False

        scored_sentences: list[tuple[float, int, str, str, str]] = []
        for result in contexts:
            chunk = result.chunk
            for sentence in _split_candidate_sentences(chunk.text):
                sent_terms = _content_terms(sentence)
                exact_hits = len(query_terms & sent_terms)
                stem_hits = sum(
                    1
                    for qt in query_terms
                    if len(qt) >= 4 and any(st.startswith(qt[:4]) for st in sent_terms)
                )
                overlap = exact_hits + 0.5 * max(0, stem_hits - exact_hits)
                if overlap <= 0:
                    continue
                rank_bonus = 1.0 / (result.rank + 1)
                score = (overlap / max(1, len(query_terms))) + 0.35 * rank_bonus
                scored_sentences.append((score, result.rank, chunk.chunk_id, sentence, chunk.text))

        if not scored_sentences:
            return _ABSTENTION_MSG, [], False

        scored_sentences.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        _, _, _, best_sentence, best_chunk_text = scored_sentences[0]
        # Require meaningful term/stem coverage in the winning chunk before answering.
        chunk_terms = _content_terms(best_chunk_text)
        matched_chunk_terms = sum(
            1
            for qt in query_terms
            if qt in chunk_terms
            or (len(qt) >= 4 and any(ct.startswith(qt[:4]) for ct in chunk_terms))
        )
        chunk_coverage = matched_chunk_terms / max(1, len(query_terms))
        best_sentence_terms = _content_terms(best_sentence)
        direct_term_hits = len(query_terms & best_sentence_terms)
        if direct_term_hits == 0 or (len(query_terms) >= 4 and chunk_coverage < 0.35):
            return _ABSTENTION_MSG, [], False

        chosen_lines: list[str] = []
        cited_ids: list[str] = []
        seen_sentences: set[str] = set()

        for _, _, chunk_id, sentence, _ in scored_sentences:
            normalized = sentence.lower()
            if normalized in seen_sentences:
                continue
            seen_sentences.add(normalized)
            chosen_lines.append(f"{sentence} [{chunk_id}]")
            if chunk_id not in cited_ids:
                cited_ids.append(chunk_id)
            if len(chosen_lines) >= 3:
                break

        return " ".join(chosen_lines), cited_ids, True


class _GroundedAnswerPayload(BaseModel):
    answer: str = Field(description="Concise answer grounded strictly in the provided chunks.")
    cited_chunk_ids: list[str] = Field(
        description="Exact chunk_id values from the context used to answer."
    )
    grounded: bool = Field(
        description="False if the provided context does not contain enough evidence to answer."
    )


class GeminiEmbeddingProvider:
    """Google Gemini text embedding provider using the official google-genai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-embedding-001") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.name = model
        self.dimension = 3072

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts in batches via client.models.embed_content."""
        vectors: list[list[float]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            response = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=types.EmbedContentConfig(),
            )
            for emb in response.embeddings or []:
                vectors.append(list(emb.values or []))
        if not vectors:
            return np.zeros((0, self.dimension), dtype=np.float32)
        matrix = np.asarray(vectors, dtype=np.float32)
        self.dimension = matrix.shape[1]
        return _l2_normalize(matrix)


class GeminiLLMProvider:
    """Google Gemini grounded synthesis provider using the Interactions API."""

    def __init__(self, api_key: str, model: str = "gemini-3.8-flash") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self.name = model

    def generate_grounded_answer(
        self, question: str, contexts: list[SearchResult]
    ) -> tuple[str, list[str], bool]:
        """Generate a schema-validated grounded answer with verified chunk citations."""
        if not contexts:
            return _ABSTENTION_MSG, [], False

        valid_ids = {item.chunk.chunk_id for item in contexts}
        formatted_context = "\n\n".join(
            (
                f"[{item.chunk.chunk_id}] source={item.chunk.source} "
                f"section={item.chunk.section}\n{item.chunk.text}"
            )
            for item in contexts
        )

        system_instruction = (
            "You are a strict technical documentation assistant. "
            "Answer the user question using ONLY the provided context chunks. "
            "Cite supporting chunk_id values in cited_chunk_ids. "
            "If the context does not contain sufficient information, set grounded=false "
            "and state clearly that the documents do not contain enough evidence."
        )
        prompt = f"Question: {question}\n\nContext:\n{formatted_context}"

        interaction = self._client.interactions.create(
            model=self._model,
            system_instruction=system_instruction,
            input=prompt,
            response_format=_GroundedAnswerPayload.model_json_schema(),
            response_mime_type="application/json",
            store=False,
        )
        payload = _GroundedAnswerPayload.model_validate_json(interaction.output_text or "{}")
        verified_citations = [cid for cid in payload.cited_chunk_ids if cid in valid_ids]
        if not payload.grounded or not verified_citations:
            return payload.answer or _ABSTENTION_MSG, [], False
        return payload.answer, verified_citations, True


def build_providers(settings: Settings) -> tuple[EmbeddingProvider, LLMProvider]:
    """Instantiate embedding and LLM providers from settings."""
    mode = settings.resolved_provider()
    if mode == "gemini":
        assert settings.gemini_api_key is not None
        return (
            GeminiEmbeddingProvider(
                api_key=settings.gemini_api_key,
                model=settings.gemini_embed_model,
            ),
            GeminiLLMProvider(
                api_key=settings.gemini_api_key,
                model=settings.gemini_chat_model,
            ),
        )
    return LocalHashingEmbedder(), LocalExtractiveLLM()
