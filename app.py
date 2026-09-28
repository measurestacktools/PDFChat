"""PDFChat — chat with any PDF using local RAG + Groq.

Pipeline (all local except the final Groq call):
  PDF bytes
    -> per-page text extraction (pypdf)
    -> cleaning / normalization
    -> page-tracked chunking (sentence-aware, with overlap)
    -> TF-IDF index built in memory (pure Python, no downloads, works offline)
  Question
    -> TF-IDF cosine retrieval of the most relevant chunks
    -> retrieved context + question sent to Groq (OpenAI-compatible API)
    -> grounded answer with real page-number sources

Docs verified live 2026 (models endpoint + probe calls):
- Endpoint (OpenAI-compatible): https://api.groq.com/openai/v1
- Chat model: openai/gpt-oss-120b (Production general chat; Llama 3.x chat
  models are no longer served — live API returns 404 for them)
- Call shape: client.chat.completions.create(model, messages, temperature, max_tokens)
- Rate limits / error codes: https://console.groq.com/docs/rate-limits ,
  https://console.groq.com/docs/errors
"""

import hashlib
import io
import logging
import math
import os
import re
import time

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import (
    APIConnectionError,
    APIStatusError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)
from pypdf import PdfReader
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pdfchat")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip() or "openai/gpt-oss-120b"
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").strip()
try:
    MAX_PDF_MB = float(os.getenv("MAX_PDF_MB", "25"))
except ValueError:
    MAX_PDF_MB = 25.0
MAX_PDF_BYTES = int(MAX_PDF_MB * 1024 * 1024)

ALLOWED_MIME = {"application/pdf"}
ALLOWED_EXT = {".pdf"}

# Retrieval / document guardrails (practical for a local single-user app).
MAX_PAGES = 400          # refuse PDFs with more pages (memory bound)
MAX_TOTAL_CHARS = 600_000  # refuse documents with more extracted text
MIN_TEXT_CHARS = 200     # below this -> "no extractable text" (likely scanned)
CHUNK_TARGET = 700       # target chunk size in characters
CHUNK_OVERLAP = 120      # overlap between consecutive chunks
TOP_K = 4                # chunks sent to the model per question
MAX_CONTEXT_CHARS = 12_000  # cap on retrieved context sent to Groq
HISTORY_TURNS = 3        # previous Q/A pairs kept for follow-up questions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="PDFChat", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory API key entered via the Settings panel in the UI.
# - Takes precedence over the .env key for this server process only.
# - Never written to disk, never logged, never sent back to the browser.
# - Cleared when the server restarts (use .env for a permanent key).
_session_key: str | None = None

# The currently loaded document (single-document workspace, in memory only).
# None when no document is loaded.
_doc: dict | None = None


class KeyPayload(BaseModel):
    key: str = ""


class ChatPayload(BaseModel):
    question: str = ""


# --------------------------------------------------------------------------
# Text utilities
# --------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the and or but if then else when while of at by for with about into "
    "through during before after above below to from up down in out on off over "
    "under again further once here there all any both each few more most other "
    "some such no nor not only own same so than too very can will just don "
    "should now is are was were be been being have has had having do does did "
    "doing would could ought i you he she it we they them his her its our your "
    "their this that these those am as what which who whom how why where when "
    "because until".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens minus stopwords (for retrieval only)."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def _clean_text(text: str) -> str:
    """Normalize whitespace; keep the original wording otherwise."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_page(text: str, page: int,
               target: int = CHUNK_TARGET, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Split one page of text into overlapping, sentence-aware chunks.

    Every chunk records its 1-based page number, so sources are always real.
    """
    text = _clean_text(text)
    if len(text) < 40:
        return []
    sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s.strip()]
    if not sentences:
        return []
    chunks: list[dict] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        # A single pathological sentence is hard-cut to keep chunks bounded.
        while len(sent) > target:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.append(sent[:target])
            sent = sent[target:]
        if current and current_len + len(sent) + 1 > target:
            chunks.append(" ".join(current))
            # Overlap: carry the tail of the finished chunk forward.
            tail = chunks[-1][-overlap:]
            current = [tail] if tail.strip() else []
            current_len = len(current[0]) if current else 0
        current.append(sent)
        current_len += len(sent) + 1
    if current:
        joined = " ".join(current).strip()
        if len(joined) >= 40:
            chunks.append(joined)
    return [{"text": c, "page": page} for c in chunks if len(c) >= 40]


# --------------------------------------------------------------------------
# Local TF-IDF retrieval (pure Python — no model downloads, works offline)
# --------------------------------------------------------------------------

class TfIdfIndex:
    """Tiny TF-IDF vector space over the document's chunks."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        n = len(chunks)
        df: dict[str, int] = {}
        self._tf: list[dict[str, float]] = []
        for ch in chunks:
            counts: dict[str, int] = {}
            for tok in _tokenize(ch["text"]):
                counts[tok] = counts.get(tok, 0) + 1
            for tok in counts:
                df[tok] = df.get(tok, 0) + 1
            # Sublinear term frequency: 1 + log(tf)
            self._tf.append({tok: 1.0 + math.log(c) for tok, c in counts.items()})
        # Smoothed inverse document frequency.
        self._idf = {tok: math.log((1 + n) / (1 + f)) + 1.0 for tok, f in df.items()}

    def search(self, query: str, top_k: int = TOP_K) -> list[tuple[int, float]]:
        """Return (chunk_index, cosine_score) pairs, best first."""
        qcounts: dict[str, int] = {}
        for tok in _tokenize(query):
            if tok in self._idf:
                qcounts[tok] = qcounts.get(tok, 0) + 1
        if not qcounts:
            return []
        qvec = {tok: (1.0 + math.log(c)) * self._idf[tok] for tok, c in qcounts.items()}
        qmag = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        scored: list[tuple[int, float]] = []
        for i, tf in enumerate(self._tf):
            dot = 0.0
            dmag_sq = 0.0
            for tok, tfv in tf.items():
                w = tfv * self._idf[tok]
                dmag_sq += w * w
                if tok in qvec:
                    dot += w * qvec[tok]
            if dot > 0 and dmag_sq > 0:
                scored.append((i, dot / (math.sqrt(dmag_sq) * qmag)))
        scored.sort(key=lambda s: s[1], reverse=True)
        return scored[:top_k]


# --------------------------------------------------------------------------
# PDF ingestion
# --------------------------------------------------------------------------

def _extract_pages(raw: bytes) -> list[str]:
    """Extract raw text per page. Raises ValueError on corrupt/unreadable PDFs."""
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError(
                    "This PDF is password-protected. Please remove the password "
                    "and upload it again."
                )
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return pages
    except ValueError:
        raise
    except Exception:
        raise ValueError(
            "This file could not be read as a PDF. It may be corrupt or not "
            "a real PDF file — try re-exporting it and upload again."
        )


def ingest_pdf(raw: bytes, filename: str) -> dict:
    """Full local pipeline: extract -> clean -> chunk -> index. Returns doc dict."""
    t0 = time.time()
    pages = _extract_pages(raw)
    if len(pages) > MAX_PAGES:
        raise ValueError(
            f"This PDF has {len(pages)} pages (limit is {MAX_PAGES}). "
            "Please split it into smaller files and upload one at a time."
        )
    chunks: list[dict] = []
    total_chars = 0
    for num, text in enumerate(pages, start=1):
        for ch in chunk_page(text, num):
            chunks.append(ch)
            total_chars += len(ch["text"])
    if total_chars > MAX_TOTAL_CHARS:
        raise ValueError(
            "This document contains too much text to index locally "
            f"({total_chars:,} characters). Please use a shorter document."
        )
    if total_chars < MIN_TEXT_CHARS:
        raise ValueError(
            "No readable text could be extracted from this PDF. It is probably "
            "a scanned/image-only document, and PDFChat does not include OCR. "
            "Please upload a text-based PDF (one you can select and copy text from)."
        )
    index = TfIdfIndex(chunks)
    return {
        "doc_id": hashlib.sha256(raw).hexdigest()[:12],
        "filename": filename,
        "size_bytes": len(raw),
        "pages": len(pages),
        "chunks": len(chunks),
        "chars": total_chars,
        "chunks_data": chunks,
        "index": index,
        "history": [],
        "took_ms": int((time.time() - t0) * 1000),
    }


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n / (1024 * 1024):.2f} MB"


def _doc_meta(d: dict) -> dict:
    return {
        "loaded": True,
        "filename": d["filename"],
        "pages": d["pages"],
        "chunks": d["chunks"],
        "size": _fmt_size(d["size_bytes"]),
    }


# --------------------------------------------------------------------------
# Key management (same secure pattern as VisionAI)
# --------------------------------------------------------------------------

def _effective_key() -> str:
    return (_session_key or GROQ_API_KEY).strip()


def _key_source() -> str | None:
    if _session_key:
        return "settings"
    if GROQ_API_KEY:
        return "env"
    return None


def _verify_key(key: str) -> None:
    client = OpenAI(api_key=key, base_url=GROQ_BASE_URL, timeout=15.0)
    client.models.list()


# --------------------------------------------------------------------------
# Groq helpers
# --------------------------------------------------------------------------

def _friendly_groq_error(exc: Exception) -> tuple[int, str]:
    if isinstance(exc, AuthenticationError):
        return 401, (
            "Your Groq API key was rejected. Replace it via Settings (top right) "
            "or check GROQ_API_KEY in your .env file (no extra spaces or quotes), "
            "then try again. Get a free key at https://console.groq.com/keys"
        )
    if isinstance(exc, RateLimitError):
        return 429, (
            "Groq rate limit reached (too many requests or tokens). "
            "Wait about a minute and try again. If it keeps happening, "
            "check your plan limits at https://console.groq.com/docs/rate-limits"
        )
    if isinstance(exc, APIConnectionError):
        return 503, (
            "Could not reach the Groq API. Check your internet connection and "
            "try again. If Groq is having an outage, wait a few minutes."
        )
    if isinstance(exc, APIStatusError):
        status = exc.status_code or 502
        detail = ""
        try:
            detail = str(exc.response.json())[:400]
        except Exception:
            detail = str(exc)[:400]
        if status == 413:
            return 413, (
                "The request was too large for Groq to process. "
                "Try a shorter question or a smaller document."
            )
        if status in (498, 499):
            return 503, (
                "Groq is temporarily at capacity and did not process the request. "
                "Wait a minute and try again — you will not be charged for this."
            )
        if status == 404:
            return 502, (
                f"Model '{GROQ_MODEL}' was not found on Groq. It may have been "
                "renamed — check https://console.groq.com/docs/models for the current "
                "chat model and update GROQ_MODEL in your .env file."
            )
        return status, f"Groq API error (HTTP {status}). Details: {detail}"
    return 500, f"Unexpected server error: {str(exc)[:300]}"


SYSTEM_PROMPT = (
    "You are PDFChat, a precise document assistant. You answer questions using "
    "ONLY the document excerpts provided below. Rules:\n"
    "1. Base every claim on the excerpts. Do not use outside knowledge.\n"
    "2. When you state a fact from the document, cite the page like [p. 4].\n"
    "3. If the excerpts do not contain enough information to answer, say exactly: "
    "\"I couldn't find that information in this document.\" Then briefly say what "
    "the document does cover on the topic, if anything.\n"
    "4. Never invent page numbers, dates, names, or figures.\n"
    "5. Keep answers focused; use short paragraphs or bullets where helpful."
)


def _build_context(hits: list[tuple[int, float]], chunks: list[dict]) -> tuple[str, list[int]]:
    parts: list[str] = []
    pages: list[int] = []
    used = 0
    for idx, _score in hits:
        ch = chunks[idx]
        block = f"--- Page {ch['page']} ---\n{ch['text']}"
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        used += len(block)
        if ch["page"] not in pages:
            pages.append(ch["page"])
    pages.sort()
    return "\n\n".join(parts), pages


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/status")
def api_status():
    key = _effective_key()
    source = _key_source()
    if key:
        where = "Settings (this session)" if source == "settings" else ".env file"
        message = f"Connected — model {GROQ_MODEL} ready. Key from {where}."
    else:
        message = (
            "No API key found. Click Settings (top right) to paste your key, "
            "or copy .env.example to .env and add your key from "
            "https://console.groq.com/keys, then restart the app."
        )
    out = {
        "ok": True,
        "configured": bool(key),
        "source": source,
        "model": GROQ_MODEL,
        "max_pdf_mb": MAX_PDF_MB,
        "message": message,
    }
    out["document"] = _doc_meta(_doc) if _doc else {"loaded": False}
    return out


@app.post("/api/key")
def api_save_key(payload: KeyPayload):
    """Save the UI-entered key in server memory after verifying it with Groq."""
    global _session_key
    key = (payload.key or "").strip()
    if not key:
        return JSONResponse(status_code=400, content={"error": "Please paste your Groq API key first."})
    try:
        _verify_key(key)
    except AuthenticationError:
        return JSONResponse(
            status_code=401,
            content={"error": (
                "That key was rejected by Groq. Check for extra spaces, "
                "make sure it starts with gsk_, or create a new one at "
                "https://console.groq.com/keys")},
        )
    except APIConnectionError:
        _session_key = key
        return {"ok": True, "verified": False, "message": (
            "Key saved, but Groq could not be reached to verify it. "
            "Check your connection — your first question will confirm the key.")}
    except Exception as exc:
        log.warning("Key verification inconclusive: %s", exc)
        _session_key = key
        return {"ok": True, "verified": False,
                "message": "Key saved. Groq did not confirm it yet — try asking a question."}
    _session_key = key
    return {"ok": True, "verified": True, "message": "API key verified and saved for this session."}


@app.delete("/api/key")
def api_delete_key():
    global _session_key
    _session_key = None
    if GROQ_API_KEY:
        return {"ok": True, "message": "Session key removed. Using the key from your .env file."}
    return {"ok": True, "message": "Session key removed."}


@app.get("/api/document")
def api_get_document():
    if not _doc:
        return {"loaded": False}
    return _doc_meta(_doc)


@app.post("/api/upload")
async def api_upload(request: Request, file: UploadFile | None = File(default=None)):
    """Upload a PDF and run the local extract -> chunk -> index pipeline."""
    global _doc
    if file is None or not file.filename:
        return JSONResponse(status_code=400, content={"error": "Please choose a PDF file first."})

    mime = (file.content_type or "").lower().split(";")[0].strip()
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXT and mime not in ALLOWED_MIME:
        return JSONResponse(
            status_code=400,
            content={"error": (
                f"Unsupported file type '{file.filename}'. "
                "Please upload a PDF file (.pdf).")},
        )

    # Early guard: reject giant bodies before reading into memory.
    try:
        content_length = int(request.headers.get("content-length") or 0)
    except ValueError:
        content_length = 0
    if content_length and content_length > MAX_PDF_BYTES + 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={"error": (
                f"PDF is too large (limit is {MAX_PDF_MB:g}MB). "
                "Please split it or choose a smaller file.")},
        )

    raw = await file.read()
    if not raw:
        return JSONResponse(
            status_code=400,
            content={"error": "The uploaded file is empty (0 bytes). Please choose a real PDF."})
    if len(raw) > MAX_PDF_BYTES:
        mb = len(raw) / (1024 * 1024)
        return JSONResponse(
            status_code=413,
            content={"error": (
                f"PDF is too large ({mb:.1f}MB — limit is {MAX_PDF_MB:g}MB). "
                "Please split it or choose a smaller file.")},
        )
    # Filenames are never written to disk or executed — used for display only.
    safe_name = os.path.basename(file.filename or "document.pdf")[:120] or "document.pdf"

    try:
        _doc = ingest_pdf(raw, safe_name)
    except ValueError as ve:
        _doc = None
        msg = str(ve)
        code = 422 if "No readable text" in msg else 400
        return JSONResponse(status_code=code, content={"error": msg})

    meta = _doc_meta(_doc)
    meta["took_ms"] = _doc["took_ms"]
    meta["message"] = (
        f"Ready — {meta['pages']} pages, {meta['chunks']} passages indexed. Ask anything about it.")
    return meta


@app.post("/api/chat")
def api_chat(payload: ChatPayload):
    """Answer a question grounded in the retrieved document chunks."""
    global _doc
    question = (payload.question or "").strip()
    if not question:
        return JSONResponse(
            status_code=400, content={"error": "Please type a question first."})
    if len(question) > 1000:
        return JSONResponse(
            status_code=400,
            content={"error": "Your question is too long (max 1000 characters). Please shorten it."})
    if not _doc:
        return JSONResponse(
            status_code=400,
            content={"error": "No document loaded. Upload a PDF first, then ask your question."})

    api_key = _effective_key()
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"error": (
                "No API key configured. Click Settings (top right) to paste your "
                "Groq key, or copy .env.example to .env, add your key from "
                "https://console.groq.com/keys, then restart the app.")},
        )

    chunks: list[dict] = _doc["chunks_data"]
    index: TfIdfIndex = _doc["index"]
    hits = index.search(question, TOP_K)
    if not hits:
        # No shared vocabulary at all — answer honestly without spending tokens.
        return {
            "answer": "I couldn't find that information in this document.",
            "sources": [],
            "chunks_used": 0,
            "model": GROQ_MODEL,
        }
    context, pages = _build_context(hits, chunks)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in _doc["history"][-HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn["q"]})
        messages.append({"role": "assistant", "content": turn["a"]})
    messages.append({
        "role": "user",
        "content": (
            f"Document excerpts relevant to my question:\n\n{context}\n\n"
            f"My question: {question}\n\n"
            "Answer using ONLY the excerpts above, citing pages like [p. N]."),
    })

    try:
        client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=800,
            temperature=0.3,
        )
        answer = (completion.choices[0].message.content or "").strip()
        if not answer:
            return JSONResponse(
                status_code=502,
                content={"error": "The AI returned an empty response. Please try again."})
    except Exception as exc:
        log.exception("Groq request failed")
        status, msg = _friendly_groq_error(exc)
        return JSONResponse(status_code=status, content={"error": msg})

    _doc["history"].append({"q": question, "a": answer})
    return {
        "answer": answer,
        "sources": pages,
        "chunks_used": len(hits),
        "model": GROQ_MODEL,
    }


@app.delete("/api/document")
def api_delete_document():
    """Forget the loaded document and its index/chat history."""
    global _doc
    _doc = None
    return {"ok": True, "message": "Document removed. Upload another PDF to start over."}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8001"))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True)
