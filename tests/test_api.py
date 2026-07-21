"""End-to-end API smoke tests via FastAPI's TestClient (offline mode)."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_up():
    r = client.get("/up")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_stats_reports_indexed_corpus_offline():
    r = client.get("/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"] >= 8
    assert body["mode"] == "offline"


def test_search_returns_relevant_source():
    r = client.post("/search", json={"query": "approximate nearest neighbour index", "k": 3})
    assert r.status_code == 200
    sources = r.json()["sources"]
    assert len(sources) == 3
    assert any(s["source"] == "04-vector-databases" for s in sources)


def test_ask_returns_answer_and_grounding_shape():
    r = client.post("/ask", json={"question": "What is chunk overlap?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert len(body["sources"]) >= 1
    assert "grounded" in body["grounding"]
    assert isinstance(body["grounding"]["citations"], list)


def test_agent_endpoint_responds():
    r = client.post("/agent", json={"question": "What is 2 plus 2?"})
    assert r.status_code == 200
    assert "answer" in r.json()


def test_ask_validation_rejects_empty_question():
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422
