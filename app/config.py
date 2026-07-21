"""Runtime configuration, read from the environment with sensible defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = ""
    chat_model: str = "gemini-2.5-flash"
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    db_path: str = "data/grounded.db"
    corpus_dir: str = "data/corpus"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4
    grounding_threshold: float = 0.45
    request_timeout: float = 45.0

    @property
    def has_key(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def mode(self) -> str:
        return "gemini" if self.has_key else "offline"


def get_settings() -> Settings:
    """Build settings from the current environment (read fresh on every call)."""
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        chat_model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
        embed_model=os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        embed_dim=int(os.getenv("EMBED_DIM", "768")),
        db_path=os.getenv("DB_PATH", "data/grounded.db"),
        corpus_dir=os.getenv("CORPUS_DIR", "data/corpus"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "900")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "150")),
        top_k=int(os.getenv("TOP_K", "4")),
        grounding_threshold=float(os.getenv("GROUNDING_THRESHOLD", "0.45")),
        request_timeout=float(os.getenv("REQUEST_TIMEOUT", "45")),
    )
