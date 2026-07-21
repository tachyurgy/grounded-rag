# Releases

## 2026-07-21 — v0.1.0 initial public release
- **What deployed:** https://rag.levelbrook.com (Docker container on Hetzner Box B
  behind kamal-proxy; auto Let's Encrypt TLS). Public repo:
  https://github.com/tachyurgy/grounded-rag
- **What it is:** a RAG Q&A service that refuses to hallucinate — FastAPI +
  LangChain over an 8-doc RAG/LLM-engineering corpus, Gemini for embeddings
  (`gemini-embedding-001`, 768-dim) and generation (`gemini-2.5-flash`), a
  dependency-free `SQLiteVectorStore` implementing LangChain's `VectorStore`, a
  deterministic citation-grounding gate, an SSE streaming endpoint, and a
  ReAct-style tool-calling agent (KB search + sandboxed calculator).
- **How:** `git clone` on Box B → `docker build -t grounded-rag .` →
  `docker run -d --network kamal -e GEMINI_API_KEY=... -v grounded_rag_data:/data` →
  `kamal-proxy deploy grounded-rag --target grounded-rag:8000 --host rag.levelbrook.com --tls`.
  DNS: A `rag` → 5.78.227.227 (DNS-only).
- **Verified:** 30/30 pytest green + ruff clean locally and in CI; live `/up` 200,
  `/stats` shows 25 chunks in gemini mode; `/ask` returns a grounded answer
  (`grounded: true`, citations `[1, 2]`, 0 unsupported); `/ask/stream` emits
  `sources → token → grounding → done`; `/agent` runs the calculator tool
  (256×4 → 1024); UI screenshot shows a clean grounded answer with 0 page errors.
