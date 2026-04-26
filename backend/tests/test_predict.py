from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

CLASSES = {"glioma", "meningioma", "notumor", "pituitary"}


def test_predict_jpeg(client: TestClient, jpeg_bytes: bytes) -> None:
    r = client.post(
        "/predict",
        files={"file": ("scan.jpg", jpeg_bytes, "image/jpeg")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"] == "scan.jpg"
    assert body["predicted_label"] in CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 4
    total = sum(p["probability"] for p in body["probabilities"])
    assert abs(total - 1.0) < 1e-3


def test_predict_png(client: TestClient, png_bytes: bytes) -> None:
    r = client.post(
        "/predict",
        files={"file": ("scan.png", png_bytes, "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_label"] in CLASSES


def test_predict_rejects_non_image(client: TestClient) -> None:
    r = client.post(
        "/predict",
        files={"file": ("note.txt", b"hello world", "text/plain")},
    )
    assert r.status_code == 415


def test_predict_rejects_corrupt_image(client: TestClient) -> None:
    r = client.post(
        "/predict",
        files={"file": ("broken.jpg", b"not really a jpeg", "image/jpeg")},
    )
    assert r.status_code == 400


def test_predict_rejects_empty_file(client: TestClient) -> None:
    r = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert r.status_code == 400


def test_predict_batch(client: TestClient, jpeg_bytes: bytes, png_bytes: bytes) -> None:
    files = [
        ("files", ("a.jpg", jpeg_bytes, "image/jpeg")),
        ("files", ("b.png", png_bytes, "image/png")),
    ]
    r = client.post("/predict/batch", files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 2
    assert {p["filename"] for p in body["predictions"]} == {"a.jpg", "b.png"}
    for p in body["predictions"]:
        assert p["predicted_label"] in CLASSES


def test_predict_batch_rejects_oversize_batch(client: TestClient, jpeg_bytes: bytes) -> None:
    # default max is 16 — send 17.
    files = [
        ("files", (f"img{i}.jpg", jpeg_bytes, "image/jpeg")) for i in range(17)
    ]
    r = client.post("/predict/batch", files=files)
    assert r.status_code == 400
    assert "Batch size" in r.json()["detail"]


def test_predict_grayscale_converted(client: TestClient) -> None:
    img = Image.new("L", (200, 200), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    r = client.post(
        "/predict",
        files={"file": ("gray.png", buf.getvalue(), "image/png")},
    )
    assert r.status_code == 200
