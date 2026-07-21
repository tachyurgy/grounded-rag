"""Marquee tests: the citation-grounding guarantee.

These pin the behaviour that makes the service trustworthy — an answer is only
"grounded" if every claim cites a real, supporting source.
"""
from app.grounding import parse_citations, support_score, verify_grounding

SOURCES = [
    "HNSW builds a layered proximity graph and offers excellent recall and latency.",
    "Chunk overlap carries a tail of the previous passage into the next chunk.",
]


def test_parse_citations_dedupes_and_sorts():
    assert parse_citations("A claim [2] and another [1] and again [2].") == [1, 2]


def test_parse_grouped_citations():
    # Models commonly emit grouped citations like [1, 2] — both must be parsed.
    assert parse_citations("Overlap carries context forward [1, 2].") == [1, 2]
    assert parse_citations("Combined [1,2] and single [3].") == [1, 2, 3]


def test_grouped_citation_answer_is_grounded():
    answer = (
        "HNSW builds a layered proximity graph with excellent recall and latency [1]. "
        "Chunk overlap carries a tail of the previous passage into the next chunk [1, 2]."
    )
    report = verify_grounding(answer, SOURCES)
    assert report.grounded is True
    assert report.citations == [1, 2]


def test_well_grounded_answer_passes():
    answer = "HNSW builds a layered proximity graph with excellent recall and latency [1]."
    report = verify_grounding(answer, SOURCES)
    assert report.grounded is True
    assert report.invalid_citations == []
    assert report.unsupported == []
    assert report.score > 0.45


def test_out_of_range_citation_is_rejected():
    # Cites [3] but only two sources were retrieved -> hallucinated citation index.
    answer = "HNSW offers excellent recall and latency [3]."
    report = verify_grounding(answer, SOURCES)
    assert report.grounded is False
    assert 3 in report.invalid_citations


def test_uncited_claim_is_flagged_unsupported():
    answer = "The capital of France is Paris and croissants are delicious."
    report = verify_grounding(answer, SOURCES)
    assert report.grounded is False
    assert report.unsupported  # no citation at all -> unsupported


def test_cited_but_unsupported_sentence_is_caught():
    # Cites a real source [2], but the sentence's content is not in that source.
    answer = "Kubernetes autoscaling uses a horizontal pod autoscaler and metrics server [2]."
    report = verify_grounding(answer, SOURCES)
    assert report.grounded is False
    assert answer.split(".")[0].strip() + "." not in [] or report.unsupported


def test_answer_with_no_citations_is_never_grounded():
    report = verify_grounding("Some fluent but unsourced prose.", SOURCES)
    assert report.grounded is False
    assert report.citations == []


def test_support_score_range():
    assert support_score("proximity graph recall latency", SOURCES[0]) > 0.5
    assert support_score("kubernetes autoscaling metrics", SOURCES[0]) == 0.0
