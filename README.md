# grounded-rag

**A retrieval-augmented Q&A service that refuses to hallucinate.** Every answer is
generated *only* from retrieved sources, cited inline, and then verified — a
deterministic grounding check rejects out-of-range citations and any claim that
doesn't actually overlap the source it cites. Answers you can't ground come back
as an honest "the sources don't cover this," not a confident fabrication.

**Live demo:** https://rag.levelbrook.com

```
┌── indexing (offline) ──┐        ┌────────────── query (online) ──────────────┐
 corpus → chunk → embed →  vector    question → embed → retrieve top-k → generate
                            store                       (LangChain)   (Gemini)
                                                              │            │
                                                              ▼            ▼
                                                         cited answer → GROUNDING GATE
                                                                         ✓ / ✕ + report
```

## Why it exists

RAG systems fail quietly: the model cites a source that doesn't support the claim,
or invents a citation number entirely. grounded-rag makes grounding a **checked
post-condition** on every response, so the trust badge in the UI reflects a real
verification, not a vibe. That verification is pure, deterministic, and covered by
tests — see [`app/grounding.py`](app/grounding.py) and
[`tests/test_grounding.py`](tests/test_grounding.py).

An answer is **grounded** only if: it contains at least one citation, every cited
index refers to a source that was actually retrieved, and every substantive
sentence lexically overlaps the source it cites above a threshold.

## Stack

- **Python 3.12** · **FastAPI** (streaming SSE) · **LangChain** primitives
- **`ChatGemini`** — a first-class LangChain `BaseChatModel` over the Gemini REST
  API (no SDK; works with key formats the SDK rejects; streams token-by-token)
- **`GeminiEmbeddings`** / **`HashingEmbeddings`** — LangChain `Embeddings`; the
  hashing backend keeps tests hermetic with no network or key
- **`SQLiteVectorStore`** — a dependency-free vector database implementing
  LangChain's `VectorStore` interface (cosine over L2-normalized float32 blobs);
  a **Chroma** adapter with identical semantics is included and tested
  ([`app/chroma_store.py`](app/chroma_store.py))
- **ReAct-style agent** with tools (knowledge-base search + a sandboxed
  calculator), executor fully unit-tested with an injected scripted LLM
- **Docker** · **docker-compose** · **Kubernetes** manifest · **GitHub Actions**
  CI (ruff + pytest + container smoke test)

The service degrades gracefully: with no `GEMINI_API_KEY` it runs offline on the
hashing embeddings and a labelled stub LLM, so CI and local dev need no secrets.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                     # 30 tests, hermetic (no key needed)

export GEMINI_API_KEY=...  # optional: enables real grounded generation
uvicorn app.main:app --reload
# open http://localhost:8000
```

Or with Docker:

```bash
docker compose up --build          # http://localhost:8000
```

## API

| Method | Path          | Purpose |
|--------|---------------|---------|
| `GET`  | `/`           | Web UI |
| `GET`  | `/up`         | Health check |
| `GET`  | `/stats`      | Indexed chunk count, models, mode |
| `POST` | `/search`     | Vector retrieval only — `{query, k}` |
| `POST` | `/ask`        | Grounded answer + sources + grounding report |
| `POST` | `/ask/stream` | Same, streamed over SSE (`sources` → `token`* → `grounding`) |
| `POST` | `/agent`      | Tool-using agent — `{question}` → answer + steps |

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"What chunk size and overlap should I use, and why?"}' | jq
```

```jsonc
{
  "answer": "A practical default is 500 to 1000 characters ... with 10–20% overlap [1]. ...",
  "sources": [{ "n": 1, "source": "02-chunking", "score": 0.71, "text": "..." }],
  "grounding": { "grounded": true, "score": 0.88, "citations": [1, 2],
                 "invalid_citations": [], "unsupported": [] }
}
```

## Corpus

The demo indexes an 8-document handbook on RAG & LLM engineering
([`data/corpus/`](data/corpus)) — RAG overview, chunking, embeddings, vector
databases, retrieval strategies, grounding, agents, and evaluation. Drop your own
`.md` files in and restart to index a different knowledge base.

## Tests

```bash
pytest                 # 30 tests
ruff check app tests   # lint
```

The suite covers the grounding guarantee (grounded vs. invalid-citation vs.
unsupported-claim), vector-store retrieval + persistence, the agent executor and
its calculator sandbox (including a code-injection rejection test), corpus
chunking, the Chroma adapter, and the full API surface — all without a network.

## Deployment

Ships as a single lean container (no torch/onnxruntime) behind a reverse proxy;
health-checked at `/up`. `docker-compose.yml` runs it locally with a persistent
volume for the vector store; [`k8s/deployment.yaml`](k8s/deployment.yaml) shows
the same service with readiness/liveness probes and a horizontal replica set.

## License

MIT
