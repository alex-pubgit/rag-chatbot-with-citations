"""Settings, read from environment variables (prefix ``RAGBOT_``) or a local ``.env`` file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

OPENAI_BASE_URL = "https://api.openai.com/v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAGBOT_", env_file=".env", extra="ignore")

    index_dir: Path = Path(".ragbot")
    chunk_size: int = Field(120, ge=20, le=2000)  # words per passage
    chunk_overlap: int = Field(20, ge=0, le=500)
    top_k: int = Field(5, ge=1, le=20)
    retrieval: Literal["hybrid", "bm25", "dense"] = "hybrid"
    # Passages scoring below this are not used to answer. Unset: the embedder's calibrated default.
    min_score: float | None = Field(None, ge=0.0, le=1.0)

    embedder: Literal["hashing", "fastembed", "openai"] = "hashing"
    embedding_model: str | None = None
    embedding_base_url: str = OPENAI_BASE_URL
    embedding_api_key: SecretStr | None = None

    answerer: Literal["extractive", "openai"] = "extractive"
    llm_model: str | None = None
    llm_base_url: str = OPENAI_BASE_URL
    llm_api_key: SecretStr | None = None

    timeout_seconds: float = Field(30.0, gt=0)
