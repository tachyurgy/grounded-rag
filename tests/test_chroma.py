"""The Chroma adapter is interchangeable with the SQLite store.

Skips automatically when chromadb is not installed (it is an optional extra kept
out of the runtime image). Install with `pip install "grounded-rag[chroma]"`.
"""
import pytest

pytest.importorskip("chromadb")

from app.chroma_store import ChromaVectorStore  # noqa: E402
from app.embeddings import HashingEmbeddings  # noqa: E402

DOCS = [
    "HNSW is an approximate nearest neighbour graph index.",
    "BM25 is a sparse keyword ranking function used in hybrid search.",
]


def test_chroma_roundtrip_and_search():
    store = ChromaVectorStore.from_texts(
        DOCS, HashingEmbeddings(256), metadatas=[{"source": "a"}, {"source": "b"}]
    )
    assert store.count() == 2
    hit = store.similarity_search("keyword ranking sparse", k=1)[0]
    assert hit.page_content == DOCS[1]


def test_chroma_as_retriever():
    store = ChromaVectorStore.from_texts(DOCS, HashingEmbeddings(256))
    docs = store.as_retriever(search_kwargs={"k": 1}).invoke("nearest neighbour graph")
    assert docs[0].page_content == DOCS[0]
