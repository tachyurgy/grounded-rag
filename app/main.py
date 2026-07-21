"""FastAPI surface for the grounded-rag service."""
from __future__ import annotations

import json
import os

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .rag import RagEngine

settings = get_settings()
engine = RagEngine(settings)
engine.index_corpus()  # index once at worker boot (idempotent if already populated)
WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")

app = FastAPI(title="grounded-rag", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = 4


@app.get("/up")
def up() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/stats")
def stats() -> dict:
    return {
        "chunks": engine.store.count(),
        "chat_model": settings.chat_model if settings.has_key else "offline",
        "embed_model": settings.embed_model if settings.has_key else "hashing (offline)",
        "mode": settings.mode,
        "grounding_threshold": settings.grounding_threshold,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.post("/search")
def search(req: SearchRequest) -> dict:
    sources = engine.retrieve(req.query, k=req.k)
    return {"query": req.query, "sources": [s.as_dict() for s in sources]}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return engine.answer(req.question, k=req.k).as_dict()


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    sources, tokens = engine.stream_answer(req.question, k=req.k)

    def gen():
        yield f"event: sources\ndata: {json.dumps([s.as_dict() for s in sources])}\n\n"
        collected = []
        for tok in tokens():
            collected.append(tok)
            yield f"event: token\ndata: {json.dumps(tok)}\n\n"
        report = engine.ground("".join(collected), sources)
        yield f"event: grounding\ndata: {json.dumps(report.as_dict())}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@app.post("/agent")
def agent(req: AskRequest) -> dict:
    result = engine.build_agent().run(req.question)
    return {
        "answer": result.answer,
        "steps": [
            {"thought": s.thought, "tool": s.tool, "tool_input": s.tool_input, "observation": s.observation}
            for s in result.steps
        ],
    }
