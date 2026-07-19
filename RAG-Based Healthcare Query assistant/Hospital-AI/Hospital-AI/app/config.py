"""
Central application configuration.

All configuration is environment-driven (12-factor style) so the same code
runs unmodified across local dev, CI, and containerized deployment. Nothing
here ever hardcodes a secret -- see `.env.example` for the variables this
expects.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- General ---------------------------------------------------------
    app_name: str = "Hospital AI Assistant"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # ---- LLM provider ------------------------------------------------------
    # "anthropic" | "openai" | "groq" | "none" (extractive fallback, no LLM calls)
    llm_provider: Literal["anthropic", "openai", "groq", "none"] = "none"
    llm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)
    groq_api_key: str | None = Field(default=None)
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # ---- Embeddings ---------------------------------------------------------
    # "sentence-transformers" | "tfidf" (offline, dependency-light fallback)
    embedding_backend: Literal["sentence-transformers", "tfidf"] = "tfidf"
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # ---- Database ----------------------------------------------------------
    database_url: str = f"sqlite:///{BASE_DIR / 'data' / 'processed' / 'hospital.db'}"
    raw_csv_path: Path = BASE_DIR / "data" / "raw" / "healthcare_dataset.csv"

    # ---- RAG -----------------------------------------------------------------
    documents_dir: Path = BASE_DIR / "data" / "documents"
    vector_store_dir: Path = BASE_DIR / "vector_store"
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 4
    similarity_threshold: float = 0.15  # below this, RAG agent declines rather than hallucinate

    # ---- SQL Agent -----------------------------------------------------------
    sql_row_limit: int = 200
    sql_query_timeout_seconds: int = 10

    # ---- Cache / memory --------------------------------------------------
    query_cache_ttl_seconds: int = 300
    query_cache_max_size: int = 256
    conversation_memory_turns: int = 6

    # ---- API -----------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = ["*"]
    api_key: str | None = None  # if set, required via X-API-Key header


@lru_cache
def get_settings() -> Settings:
    """Settings are cheap to build but env parsing is not free -- cache it."""
    return Settings()
