"""Optional Chroma-backed vector store — proves the retrieval layer is a swappable
adapter over a named vector database, not tied to the bundled SQLite store.

Kept out of the runtime image (chromadb pulls onnxruntime, which is heavy for a
2GB box); the demo runs on ``SQLiteVectorStore``. Install the extra to use it:

    pip install "grounded-rag[chroma]"

Both stores implement LangChain's ``VectorStore`` interface, so the RAG engine
works against either unchanged.
"""
from __future__ import annotations

from collections.abc import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


class ChromaVectorStore(VectorStore):
    def __init__(self, embedding: Embeddings, collection: str = "grounded", persist_dir: str | None = None):
        import chromadb  # imported lazily so the base image needn't ship it

        self._embedding = embedding
        client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        # We supply our own embeddings, so disable Chroma's default embedding fn.
        self._col = client.get_or_create_collection(collection, embedding_function=None)
        self._counter = self._col.count()

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    def count(self) -> int:
        return self._col.count()

    def add_texts(self, texts: Iterable[str], metadatas: list[dict] | None = None, **kwargs) -> list[str]:
        texts = list(texts)
        vectors = self._embedding.embed_documents(texts)
        ids = [str(self._counter + i) for i in range(len(texts))]
        self._counter += len(texts)
        # Chroma rejects empty {} metadata; pass None unless real metadata exists.
        metas = metadatas if metadatas and any(metadatas) else None
        self._col.add(ids=ids, documents=texts, embeddings=vectors, metadatas=metas)
        return ids

    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs) -> list[tuple[Document, float]]:
        if self._col.count() == 0:
            return []
        q = self._embedding.embed_query(query)
        res = self._col.query(query_embeddings=[q], n_results=min(k, self._col.count()))
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out: list[tuple[Document, float]] = []
        for text, meta, dist in zip(docs, metas, dists, strict=False):
            out.append((Document(page_content=text, metadata=meta or {}), 1.0 - float(dist)))
        return out

    def similarity_search(self, query: str, k: int = 4, **kwargs) -> list[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, **kwargs)]

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs) -> ChromaVectorStore:
        store = cls(embedding, **kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store
