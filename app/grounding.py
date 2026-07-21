"""Citation grounding — the correctness core of the service.

A RAG answer is only trustworthy if every claim is *actually* supported by the
retrieved sources it cites. These are pure, deterministic functions (no LLM, no
network) so the guarantee is testable and cheap to enforce on every response:

1. Every inline citation ``[n]`` must reference a source that was actually
   retrieved (no out-of-range / hallucinated citation indices).
2. Every substantive sentence must carry a citation.
3. A cited sentence must lexically overlap the source it points at above a
   threshold, or it is flagged as unsupported.

An answer is ``grounded`` only if it has at least one citation, no invalid
citations, and no unsupported sentences. ``main.py`` surfaces the report to the
caller and the UI renders it as a trust badge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_CITE = re.compile(r"\[([\d,\s]+)\]")  # matches [1], [1,2], [1, 2]
_NUM = re.compile(r"\d+")
_WORD = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "is",
    "are", "was", "were", "be", "been", "it", "its", "this", "that", "these",
    "those", "as", "at", "by", "with", "from", "into", "your", "you", "can",
    "will", "which", "when", "how", "what", "why", "not", "no", "if", "then",
    "so", "than", "such", "also", "may", "should", "would", "could", "do",
    "does", "use", "used", "using", "they", "their", "them", "we", "our",
}


def parse_citations(answer: str) -> list[int]:
    """Return the sorted, unique 1-based citation indices found in the answer.

    Handles single ``[1]`` and grouped ``[1, 2]`` / ``[1,2]`` citation markers.
    """
    nums: set[int] = set()
    for group in _CITE.findall(answer):
        nums.update(int(n) for n in _NUM.findall(group))
    return sorted(nums)


def _content_tokens(text: str) -> set:
    return {t for t in _WORD.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2}


def support_score(claim: str, source: str) -> float:
    """Fraction of a claim's content tokens that appear in the source text."""
    claim_tokens = _content_tokens(claim)
    if not claim_tokens:
        return 1.0  # nothing substantive to support (e.g. a transitional phrase)
    source_tokens = _content_tokens(source)
    return len(claim_tokens & source_tokens) / len(claim_tokens)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]


@dataclass
class SentenceCheck:
    sentence: str
    citations: list[int]
    score: float
    supported: bool


@dataclass
class GroundingReport:
    grounded: bool
    score: float
    citations: list[int]
    invalid_citations: list[int]
    unsupported: list[str]
    sentences: list[SentenceCheck] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "grounded": self.grounded,
            "score": round(self.score, 3),
            "citations": self.citations,
            "invalid_citations": self.invalid_citations,
            "unsupported": self.unsupported,
            "sentences": [
                {
                    "sentence": s.sentence,
                    "citations": s.citations,
                    "score": round(s.score, 3),
                    "supported": s.supported,
                }
                for s in self.sentences
            ],
        }


def verify_grounding(
    answer: str, sources: list[str], threshold: float = 0.45
) -> GroundingReport:
    """Check that ``answer`` is grounded in the retrieved ``sources`` (1-indexed)."""
    all_citations = parse_citations(answer)
    invalid = [c for c in all_citations if c < 1 or c > len(sources)]

    checks: list[SentenceCheck] = []
    unsupported: list[str] = []
    scored: list[float] = []

    for sentence in _split_sentences(answer):
        cites = [c for c in parse_citations(sentence) if 1 <= c <= len(sources)]
        content = _content_tokens(_CITE.sub("", sentence))
        if not content:
            continue  # skip citation-only or trivial fragments
        if not cites:
            checks.append(SentenceCheck(sentence, [], 0.0, False))
            unsupported.append(sentence)
            continue
        best = max(support_score(sentence, sources[c - 1]) for c in cites)
        supported = best >= threshold
        scored.append(best)
        checks.append(SentenceCheck(sentence, cites, best, supported))
        if not supported:
            unsupported.append(sentence)

    score = sum(scored) / len(scored) if scored else 0.0
    grounded = bool(all_citations) and not invalid and not unsupported
    return GroundingReport(
        grounded=grounded,
        score=score,
        citations=all_citations,
        invalid_citations=invalid,
        unsupported=unsupported,
        sentences=checks,
    )
