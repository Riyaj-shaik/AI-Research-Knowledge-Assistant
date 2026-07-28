"""
test_api.py - Basic integration tests for the AI Research Assistant API.

Run with:
    pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "documents_in_store" in data


def test_list_documents_empty():
    response = client.get("/documents/")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total" in data


def test_get_nonexistent_document():
    response = client.get("/documents/nonexistent-id")
    assert response.status_code == 404


def test_delete_nonexistent_document():
    response = client.delete("/documents/nonexistent-id")
    assert response.status_code == 404


def test_search_no_documents():
    response = client.post("/search/", json={"query": "machine learning"})
    assert response.status_code in [200, 404]


def test_analytics():
    response = client.get("/analytics/")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data


def test_classifier_status_untrained():
    response = client.get("/ml/status")
    assert response.status_code == 200


def test_ask_no_documents():
    response = client.post("/assistant/ask", json={"question": "What is AI?"})
    assert response.status_code in [200, 404]


def test_session_empty():
    response = client.get("/assistant/session/test-session")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test-session"
    assert data["history"] == []


def test_clear_session():
    response = client.delete("/assistant/session/test-session")
    assert response.status_code == 200


def test_compare_insufficient_docs():
    response = client.post("/assistant/compare", json={"doc_ids": ["single-doc"]})
    assert response.status_code == 400
