"""Force a hermetic, offline, in-memory environment for the whole test session.

Removing GEMINI_API_KEY makes the suite run on the deterministic hashing
embeddings and the offline LLM stub — no network, no quota, fully reproducible.
"""
import os

os.environ.pop("GEMINI_API_KEY", None)
os.environ["DB_PATH"] = ":memory:"
os.environ["CORPUS_DIR"] = "data/corpus"
