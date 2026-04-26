from __future__ import annotations

from fastapi.testclient import TestClient


def test_root(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["name"]
    assert body["version"]
    assert body["docs_url"] == "/docs"


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["app_name"]


def test_classes(client: TestClient) -> None:
    r = client.get("/classes")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 4
    assert "glioma" in body["classes"]


def test_model_info(client: TestClient) -> None:
    r = client.get("/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["num_classes"] == 4
    assert body["architecture"]
    assert isinstance(body["weights_loaded"], bool)
    assert body["input_size"] == [224, 224]
