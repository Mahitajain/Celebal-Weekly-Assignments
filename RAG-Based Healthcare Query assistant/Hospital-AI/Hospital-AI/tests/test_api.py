from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["admissions_row_count"] > 0
    assert body["documents_indexed"] > 0


def test_chat_sql_route(client):
    resp = client.post("/api/v1/chat", json={"message": "How many emergency admissions are there?", "session_id": "t1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "sql"
    assert body["sql_detail"]["sql"] is not None


def test_chat_rag_route(client):
    resp = client.post("/api/v1/chat", json={"message": "What is the hospital visitor policy?", "session_id": "t2"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["route"] == "rag"
    assert body["rag_detail"]["citations"]


def test_chat_rejects_empty_message(client):
    resp = client.post("/api/v1/chat", json={"message": "", "session_id": "t3"})
    assert resp.status_code == 422


def test_clear_session(client):
    client.post("/api/v1/chat", json={"message": "How many patients have asthma?", "session_id": "t4"})
    resp = client.delete("/api/v1/chat/t4")
    assert resp.status_code == 200


def test_chat_caches_repeated_question(client):
    r1 = client.post("/api/v1/chat", json={"message": "How many patients have arthritis?", "session_id": "t5"})
    r2 = client.post("/api/v1/chat", json={"message": "How many patients have arthritis?", "session_id": "t5"})
    assert r1.json()["answer"] == r2.json()["answer"]
