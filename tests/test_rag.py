"""RagEngine indexing: idempotent, and rebuilds when the corpus changes."""
from app.config import Settings
from app.rag import RagEngine


def _engine(corpus_dir: str) -> RagEngine:
    return RagEngine(Settings(corpus_dir=corpus_dir, db_path=":memory:", chunk_size=400, chunk_overlap=50))


def test_index_idempotent_and_rebuilds_on_change(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "01.md").write_text("HNSW is a graph index for vector search.\n\nBM25 is sparse keyword ranking.")
    eng = _engine(str(d))

    n1 = eng.index_corpus()
    assert n1 > 0
    assert eng.index_corpus() == n1  # same corpus -> no duplication

    (d / "02.md").write_text("Rerankers use a cross-encoder to score retrieved candidates.")
    n2 = eng.index_corpus()
    assert n2 > n1  # new doc -> rebuilt
    assert eng.retrieve("cross encoder reranker candidates", k=1)[0].source == "02"
