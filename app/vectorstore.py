"""A dependency-free vector database backed by SQLite.

Embeddings are persisted as float32 blobs and queried by cosine similarity
(vectors are L2-normalized on write, so cosine reduces to a dot product against
a stacked NumPy matrix). This implements LangChain's ``VectorStore`` interface,
so ``.as_retriever()`` and any LCEL chain work against it unchanged.

To swap in a "real" vector database, implement the same interface — see
``app/chroma_store.py`` for a Chroma-backed adapter with identical semantics.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class SQLiteVectorStore(VectorStore):
    def __init__(self, embedding: Embeddings, db_path: str = ":memory:") -> None:
        self._embedding = embedding
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS docs ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  text TEXT NOT NULL,"
            "  metadata TEXT NOT NULL DEFAULT '{}',"
            "  vec BLOB NOT NULL,"
            "  dim INTEGER NOT NULL)"
        )
        self._conn.commit()
        self._matrix: np.ndarray | None = None
        self._rows: list[tuple[int, str, dict]] = []
        self._dirty = True

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0])

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        **kwargs,
    ) -> list[str]:
        texts = list(texts)
        metadatas = metadatas or [{} for _ in texts]
        vectors = self._embedding.embed_documents(texts)
        ids: list[str] = []
        cur = self._conn.cursor()
        for text, meta, vec in zip(texts, metadatas, vectors, strict=False):
            arr = np.asarray(vec, dtype="<f4")
            arr = arr / (float(np.linalg.norm(arr)) or 1.0)
            cur.execute(
                "INSERT INTO docs (text, metadata, vec, dim) VALUES (?,?,?,?)",
                (text, json.dumps(meta), arr.tobytes(), int(arr.shape[0])),
            )
            ids.append(str(cur.lastrowid))
        self._conn.commit()
        self._dirty = True
        return ids

    def _load(self) -> None:
        if not self._dirty and self._matrix is not None:
            return
        rows = self._conn.execute("SELECT id, text, metadata, vec FROM docs").fetchall()
        self._rows = [(r[0], r[1], json.loads(r[2])) for r in rows]
        self._matrix = (
            np.vstack([np.frombuffer(r[3], dtype="<f4") for r in rows]) if rows else None
        )
        self._dirty = False

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs
    ) -> list[tuple[Document, float]]:
        self._load()
        if self._matrix is None:
            return []
        q = np.asarray(self._embedding.embed_query(query), dtype="<f4")
        q = q / (float(np.linalg.norm(q)) or 1.0)
        sims = self._matrix @ q
        k = min(k, sims.shape[0])
        top = np.argsort(-sims)[:k]
        results: list[tuple[Document, float]] = []
        for i in top:
            doc_id, text, meta = self._rows[int(i)]
            results.append(
                (Document(page_content=text, metadata={**meta, "id": doc_id}), float(sims[int(i)]))
            )
        return results

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, **kwargs)]

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        db_path: str = ":memory:",
        **kwargs,
    ) -> SQLiteVectorStore:
        store = cls(embedding, db_path=db_path)
        store.add_texts(texts, metadatas=metadatas)
        return store
