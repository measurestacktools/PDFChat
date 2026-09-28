"""API validation tests (no Groq key needed — error/validation paths only).

Run:  pytest tests/test_api.py
Live Groq tests are NOT included here; see README for the manual live check.
"""
import io
import os

import httpx
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

import app as m


def _client():
    m._session_key = None
    m._doc = None
    return TestClient(m.app)


def _blank_pdf_bytes():
    buf = io.BytesIO()
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    w.write(buf)
    buf.seek(0)
    return buf.getvalue()


def test_home_and_status():
    c = _client()
    r = c.get("/")
    assert r.status_code == 200 and "PDFChat" in r.text
    r = c.get("/api/status")
    d = r.json()
    assert r.status_code == 200
    assert d["configured"] is False and d["document"]["loaded"] is False
    assert d["model"] == "openai/gpt-oss-120b"


def test_static_assets():
    c = _client()
    for p in ("/static/styles.css", "/static/app.js"):
        r = c.get(p)
        assert r.status_code == 200 and len(r.text) > 1000


def _groq_reachable() -> bool:
    """Key verification calls the live Groq API — skip that assertion offline."""
    try:
        httpx.get("https://api.groq.com/openai/v1/models", timeout=5.0)
        return True
    except Exception:
        return False


def test_key_validation():
    c = _client()
    assert c.post("/api/key", json={"key": ""}).status_code == 400
    if not _groq_reachable():
        pytest.skip("no network: live key-verification test skipped")
    r = c.post("/api/key", json={"key": "gsk_fake_key_for_tests"})
    assert r.status_code == 401
    assert c.get("/api/status").json()["configured"] is False
    assert c.delete("/api/key").status_code == 200


def test_upload_validation():
    c = _client()
    assert c.post("/api/upload").status_code in (400, 422)
    r = c.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "PDF" in r.json()["error"]
    r = c.post("/api/upload", files={"file": ("empty.pdf", b"", "application/pdf")})
    assert r.status_code == 400
    r = c.post("/api/upload", files={"file": ("junk.pdf", b"%PDF-not-really" * 100, "application/pdf")})
    assert r.status_code == 400
    r = c.post("/api/upload", files={"file": ("big.pdf", os.urandom(26 * 1024 * 1024), "application/pdf")})
    assert r.status_code == 413
    # Valid PDF container but zero extractable text -> 422 scanned-PDF message.
    r = c.post("/api/upload", files={"file": ("blank.pdf", _blank_pdf_bytes(), "application/pdf")})
    assert r.status_code == 422 and "OCR" in r.json()["error"]
    assert c.get("/api/document").json()["loaded"] is False


def test_chat_validation_without_document():
    c = _client()
    assert c.post("/api/chat", json={"question": ""}).status_code == 400
    assert c.post("/api/chat", json={"question": "x" * 1001}).status_code == 400
    r = c.post("/api/chat", json={"question": "Summarize?"})
    assert r.status_code == 400 and "Upload a PDF" in r.json()["error"]


def test_delete_document_when_empty():
    c = _client()
    r = c.delete("/api/document")
    assert r.status_code == 200
