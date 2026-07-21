"""Corpus loading and chunking."""
from app.ingest import _split, load_corpus


def test_load_corpus_reads_markdown():
    chunks = load_corpus("data/corpus", size=900, overlap=150)
    assert len(chunks) >= 8
    sources = {c.source for c in chunks}
    assert "04-vector-databases" in sources
    assert all(c.text.strip() for c in chunks)


def test_split_respects_size_budget_with_overlap():
    text = "\n\n".join(f"Paragraph number {i} with some filler words here." for i in range(20))
    chunks = _split(text, size=120, overlap=20)
    assert len(chunks) > 1
    # No chunk is wildly over budget (a single paragraph may exceed, but not by much).
    assert max(len(c) for c in chunks) < 260


def test_ordinals_are_sequential_per_source():
    chunks = load_corpus("data/corpus", size=400, overlap=50)
    by_source: dict = {}
    for c in chunks:
        by_source.setdefault(c.source, []).append(c.ordinal)
    for ordinals in by_source.values():
        assert ordinals == list(range(len(ordinals)))


def test_corpus_fingerprint_changes_with_content(tmp_path):
    from app.ingest import corpus_fingerprint

    (tmp_path / "a.md").write_text("alpha content here")
    fp1 = corpus_fingerprint(str(tmp_path))
    assert corpus_fingerprint(str(tmp_path)) == fp1  # stable
    (tmp_path / "b.md").write_text("beta content here")
    assert corpus_fingerprint(str(tmp_path)) != fp1  # changed corpus -> new fp
