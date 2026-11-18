from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import SAMPLE_DOCS
from ragbot.api import MAX_BODY_BYTES, create_app
from ragbot.embeddings.hashing import HashingEmbedder
from ragbot.ingest import ingest
from ragbot.settings import Settings


@pytest.fixture(scope="module")
def client(tmp_path_factory) -> TestClient:
    index_dir = tmp_path_factory.mktemp("index")
    ingest(SAMPLE_DOCS, index_dir, HashingEmbedder(), chunk_size=120, chunk_overlap=20)
    return TestClient(create_app(Settings(_env_file=None, index_dir=index_dir)))


def test_ask_returns_the_documented_contract(client):
    response = client.post("/ask", json={"question": "What is the bridge fee?", "top_k": 5})
    body = response.json()
    assert response.status_code == 200
    assert set(body) == {"answer", "refused", "citations", "latency_ms"}
    assert body["refused"] is False
    assert "0.05%" in body["answer"]
    assert set(body["citations"][0]) == {"n", "source", "title", "heading", "snippet", "score"}
    assert body["citations"][0]["source"] == "bridge.md"
    assert body["citations"][0]["heading"] == "Bridge fee"


def test_unanswerable_question_is_refused_without_citations(client):
    body = client.post("/ask", json={"question": "Who founded AcmeSwap?"}).json()
    assert body["refused"] is True
    assert body["citations"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "x" * 1001},
        {"question": "ok", "top_k": 0},
        {"question": "ok", "top_k": 21},
    ],
)
def test_invalid_request_is_rejected(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_oversized_body_is_rejected_before_it_is_parsed(client):
    response = client.post(
        "/ask", content=b"x" * (MAX_BODY_BYTES + 1), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 413


def test_sources_lists_every_indexed_document(client):
    sources = client.get("/sources").json()
    assert len(sources) == 12
    assert {"source": "bridge.md", "title": "Bridge", "chunks": 4} in sources


def test_health_reports_the_index(client):
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["documents"] == 12
    assert health["embedder"] == "hashing-v1-2048"


def test_web_page_makes_no_external_requests(client):
    page = client.get("/").text
    assert "<title>Docs chatbot</title>" in page
    assert "http://" not in page
    assert "https://" not in page
