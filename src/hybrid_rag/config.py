"""Application configuration loaded from environment variables or .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for retrieval, chunking, and LLM/embedding providers."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    rag_provider: Literal["auto", "gemini", "local"] = "auto"
    gemini_api_key: str | None = None
    gemini_chat_model: str = "gemini-3.8-flash"
    gemini_embed_model: str = "gemini-embedding-001"

    chunk_size: int = Field(default=450, ge=100, le=4000)
    chunk_overlap: int = Field(default=80, ge=0, le=1000)
    top_k: int = Field(default=4, ge=1, le=20)
    rrf_k: int = Field(default=60, ge=1, le=200)
    min_relevance_score: float = Field(default=0.01, ge=0.0)
    sample_docs_dir: Path = Path("data/sample_docs")

    def resolved_provider(self) -> Literal["gemini", "local"]:
        """Resolve 'auto' to 'gemini' when an API key is configured, else 'local'."""
        if self.rag_provider == "gemini":
            if not self.gemini_api_key:
                raise ValueError("RAG_PROVIDER=gemini requires GEMINI_API_KEY to be set.")
            return "gemini"
        if self.rag_provider == "local":
            return "local"
        return "gemini" if self.gemini_api_key else "local"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
