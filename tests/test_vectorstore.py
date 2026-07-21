"""The SQLite vector store retrieves the right chunk and persists across reopen."""
import os
import tempfile

from app.embeddings import HashingEmbeddings
from app.vectorstore import SQLiteVectorStore

DOCS = [
    "HNSW is an approximate nearest neighbour graph index for vector search.",
    "Chunk overlap prevents a sentence from being lost at a passage boundary.",
    "Cosine similarity on normalized vectors reduces to a dot product.",
]


def _store(path=":memory:"):
    return SQLiteVectorStore.from_texts(
        DOCS, HashingEmbeddings(256), metadatas=[{"source": f"d{i}"} for i in range(len(DOCS))], db_path=path
    )


def test_retrieves_most_relevant_chunk():
    store = _store()
    hits = store.similarity_search("approximate nearest neighbour graph index", k=1)
    assert hits[0].page_content == DOCS[0]


def test_similarity_scores_are_ordered():
    store = _store()
    scored = store.similarity_search_with_score("boundary overlap passage", k=3)
    scores = [s for _, s in scored]
    assert scores == sorted(scores, reverse=True)
    assert scored[0][0].page_content == DOCS[1]


def test_as_retriever_langchain_interface():
    store = _store()
    retriever = store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("dot product normalized vectors")
    assert docs[0].page_content == DOCS[2]


def test_count_and_metadata():
    store = _store()
    assert store.count() == 3
    hit = store.similarity_search("HNSW graph", k=1)[0]
    assert hit.metadata["source"] == "d0"


def test_persistence_across_reopen():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "vec.db")
        _store(path)
        reopened = SQLiteVectorStore(HashingEmbeddings(256), db_path=path)
        assert reopened.count() == 3
        assert reopened.similarity_search("nearest neighbour", k=1)[0].page_content == DOCS[0]
