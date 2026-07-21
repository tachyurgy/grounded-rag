"""Embedding backends implementing LangChain's ``Embeddings`` interface.

Two backends, selected automatically:

* ``GeminiEmbeddings`` — Google ``gemini-embedding-001`` over the public REST API
  (no SDK, works with the AQ.* API keys). Used when ``GEMINI_API_KEY`` is set.
* ``HashingEmbeddings`` — a deterministic, dependency-free bag-of-words hashing
  vectorizer used offline / in CI so tests stay hermetic and need no network.
"""
from __future__ import annotations

import hashlib
import math
import re

import httpx
from langchain_core.embeddings import Embeddings

from .config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")
_GENAI = "https://generativelanguage.googleapis.com/v1beta"


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class HashingEmbeddings(Embeddings):
    """Deterministic hashing embedding — no model, no network, no dependencies.

    Each token is hashed into one of ``dim`` buckets with a signed contribution,
    then the vector is L2-normalized. Retrieval quality is modest but perfectly
    adequate for a curated corpus with distinct per-document vocabulary, which is
    exactly what the test suite relies on to run without an API key.
    """

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall(text.lower()):
            digest = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = digest % self.dim
            sign = 1.0 if (digest >> 8) & 1 else -1.0
            vec[idx] += sign
        return _l2_normalize(vec)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)


class GeminiEmbeddings(Embeddings):
    """Gemini embeddings via REST. Vectors are L2-normalized so cosine == dot."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.model = settings.embed_model
        self.dim = settings.embed_dim

    def _embed(self, text: str) -> list[float]:
        url = f"{_GENAI}/models/{self.model}:embedContent?key={self._settings.gemini_api_key}"
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text[:8000]}]},
            "outputDimensionality": self.dim,
        }
        with httpx.Client(timeout=self._settings.request_timeout) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            values = resp.json()["embedding"]["values"]
        return _l2_normalize(values)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(settings: Settings) -> Embeddings:
    """Return the best available embedding backend for the current environment."""
    if settings.has_key:
        return GeminiEmbeddings(settings)
    return HashingEmbeddings(settings.embed_dim)
