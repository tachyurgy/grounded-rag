"""The RAG engine: index the corpus, retrieve, generate a cited answer, and
verify that the answer is grounded in what was retrieved."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from .agent import Agent, safe_calculator
from .config import Settings
from .embeddings import build_embeddings
from .grounding import GroundingReport, verify_grounding
from .ingest import load_corpus
from .llm import ChatGemini
from .vectorstore import SQLiteVectorStore

_SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered sources provided. "
    "Cite every claim inline with the matching source number in square brackets, "
    "e.g. [1] or [2]. Do not use any knowledge outside the sources. "
    "If the sources do not contain the answer, say so plainly and cite nothing. "
    "Be concise and specific."
)


@dataclass
class Source:
    n: int
    source: str
    score: float
    text: str

    def as_dict(self) -> dict:
        return {"n": self.n, "source": self.source, "score": round(self.score, 3), "text": self.text}


@dataclass
class Answer:
    answer: str
    sources: list[Source]
    grounding: GroundingReport

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": [s.as_dict() for s in self.sources],
            "grounding": self.grounding.as_dict(),
        }


def _format_context(sources: list[Source]) -> str:
    return "\n\n".join(f"[{s.n}] (source: {s.source})\n{s.text}" for s in sources)


class RagEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embeddings = build_embeddings(settings)
        self.store = SQLiteVectorStore(self.embeddings, settings.db_path)
        self.llm = ChatGemini(
            api_key=settings.gemini_api_key,
            model=settings.chat_model,
            timeout=settings.request_timeout,
        )

    # -- indexing -----------------------------------------------------------
    def index_corpus(self, force: bool = False) -> int:
        if self.store.count() > 0 and not force:
            return self.store.count()
        chunks = load_corpus(
            self.settings.corpus_dir,
            size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if chunks:
            self.store.add_texts(
                [c.text for c in chunks],
                metadatas=[{"source": c.source, "ordinal": c.ordinal} for c in chunks],
            )
        return self.store.count()

    # -- retrieval ----------------------------------------------------------
    def retrieve(self, query: str, k: int | None = None) -> list[Source]:
        hits = self.store.similarity_search_with_score(query, k=k or self.settings.top_k)
        return [
            Source(n=i + 1, source=doc.metadata.get("source", "?"), score=score, text=doc.page_content)
            for i, (doc, score) in enumerate(hits)
        ]

    def _messages(self, query: str, sources: list[Source]):
        context = _format_context(sources)
        return [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=f"Sources:\n{context}\n\nQuestion: {query}"),
        ]

    # -- generation ---------------------------------------------------------
    def answer(self, query: str, k: int | None = None) -> Answer:
        sources = self.retrieve(query, k=k)
        text = self.llm._generate(self._messages(query, sources)).generations[0].message.content
        report = verify_grounding(
            text, [s.text for s in sources], threshold=self.settings.grounding_threshold
        )
        return Answer(answer=text, sources=sources, grounding=report)

    def stream_answer(self, query: str, k: int | None = None):
        """Yield (sources, token_iterator) so the API can stream and then verify."""
        sources = self.retrieve(query, k=k)

        def tokens() -> Iterator[str]:
            for chunk in self.llm._stream(self._messages(query, sources)):
                content = chunk.message.content
                if content:
                    yield content

        return sources, tokens

    def ground(self, text: str, sources: list[Source]) -> GroundingReport:
        return verify_grounding(
            text, [s.text for s in sources], threshold=self.settings.grounding_threshold
        )

    # -- agent --------------------------------------------------------------
    def build_agent(self) -> Agent:
        def search_kb(q: str) -> str:
            hits = self.retrieve(q, k=3)
            if not hits:
                return "no results"
            return " | ".join(f"[{h.source}] {h.text[:200]}" for h in hits)

        tools = {"search_kb": search_kb, "calculator": safe_calculator}
        return Agent(generate=self.llm.complete, tools=tools)
