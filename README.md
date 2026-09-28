# PDFChat — Chat with any PDF

PDFChat is a polished, ready-to-run **AI PDF assistant**. Upload any text-based PDF, ask questions about it, and get answers grounded **only** in your document — with real page-number sources. Running on Groq's ultra-fast inference.

No database. No frontend framework. No fake responses. Just a clean FastAPI backend with a real local RAG pipeline + a premium document-workspace interface + your own Groq API key.

## Features

- Drag-and-drop PDF upload with staged processing display (Uploading → Extracting text → Creating index → Ready)
- Document sidebar: filename, page count, file size, passage count, status, one-click remove
- Chat workspace: message history, typing animation, suggested questions, page-source chips under every answer
- Real RAG: per-page extraction, sentence-aware chunking, local TF-IDF retrieval — the model never gets the whole PDF blindly
- Grounded answers: if the document doesn't contain the answer, PDFChat says so instead of hallucinating
- Settings panel: paste your Groq key in the UI, verified instantly, kept only in server memory
- Live API status pill, beginner-friendly errors for every failure mode
- Responsive desktop + mobile, zero-dependency retrieval (works offline except the Groq call)

## Requirements

- Python 3.10 or newer
- A free Groq API key (takes ~2 minutes)
- Internet connection (the AI call goes to Groq's API; PDF processing itself is 100% local)

## Installation

```bash
cd PDFChat
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Run the automated test suite (needs no API key, no network for most tests):

```bash
pip install -r requirements-test.txt
pytest tests/ -q
```

## Creating a Groq API key (free)

1. Go to **https://console.groq.com/keys**
2. Sign up / log in (free tier is enough for this project)
3. Click **Create API Key**
4. Copy the key — it starts with `gsk_...`

## API-key configuration (pick either option)

**Option A — Settings panel (easiest, no code):**

1. Run the app and open http://127.0.0.1:8001
2. Click **Settings** (top right)
3. Paste your key, click **Save key** — it's verified against Groq instantly

The key is kept only in the server's memory: never written to disk, never logged, never stored in the browser, never shown again. It clears when the server restarts. Use **Remove** in Settings to forget it at any time.

**Option B — `.env` file (permanent):**

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Then open `.env` and paste your key:

```
GROQ_API_KEY=gsk_paste_your_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

> The default model `openai/gpt-oss-120b` is a current production chat model, verified live against the Groq API. If Groq renames models in the future and you see a "model was not found" error, check https://console.groq.com/docs/models and update `GROQ_MODEL`.

## Running the app

```bash
uvicorn app:app --reload --port 8001
```

Then open **http://127.0.0.1:8001** in your browser.

- Homepage: `GET /`
- Health check: `GET /api/status` (key state + loaded document summary)
- Upload: `POST /api/upload` (multipart `file`)
- Chat: `POST /api/chat` (JSON `{"question": "..."}`)
- Document info: `GET /api/document`
- Remove document: `DELETE /api/document`
- Save key: `POST /api/key` · Remove key: `DELETE /api/key`

## How PDF processing works

```
Your PDF (stays in server memory, never saved to disk)
   │  1. Per-page text extraction (pypdf)
   │  2. Cleaning (whitespace normalization)
   │  3. Chunking (~700 chars, sentence-aware, ~120 overlap,
   │     every chunk tagged with its real page number)
   │  4. Local TF-IDF index built over the chunks
   ▼
Your question
   │  5. TF-IDF cosine retrieval → top 4 most relevant chunks
   │  6. Chunks + question + short chat history sent to Groq
   │     (max ~12,000 chars of context, temperature 0.3 for factuality)
   ▼
Grounded answer + page sources (e.g. p. 4, p. 7)
```

## RAG in simple language

RAG (Retrieval-Augmented Generation) means the AI doesn't guess from memory: for each question, PDFChat **retrieves** the most relevant passages from *your* document, then the AI **generates** its answer from those passages only. Retrieval here is TF-IDF keyword search — a classic, dependency-free technique that runs entirely on your machine (no downloads, no external embedding service). The system prompt additionally forbids the model from using outside knowledge and from inventing page numbers.

Key files:

| File | Purpose |
|---|---|
| `app.py` | FastAPI backend, RAG pipeline, Groq call |
| `static/index.html` | Workspace structure (sidebar + chat) |
| `static/styles.css` | Document-workspace theme |
| `static/app.js` | Upload, chat, sources, settings |
| `tests/test_rag.py` | Retrieval/chunking unit tests |
| `tests/test_api.py` | API validation tests |
| `.env.example` | Template for your config |
| `requirements.txt` | Python dependencies |

## Supported PDFs

- Normal **text-based PDFs** (ones you can select and copy text from): fully supported
- Limits: 25MB, 400 pages, ~600,000 characters of text (configurable in `app.py`)
- **Scanned / image-only PDFs are NOT supported** — there is no OCR. If a PDF has no extractable text, you'll get a clear error explaining this (HTTP 422), not a hallucinated answer.
- Password-protected PDFs: rejected with instructions to remove the password first

## Limitations

- One document at a time (uploading a new PDF replaces the current one; use Remove to start over)
- Keyword-based (TF-IDF) retrieval: excellent for factual lookup, weaker than neural embeddings at paraphrase-heavy questions
- Short chat history (last 3 exchanges) for follow-ups; long multi-turn analysis sessions may lose early context
- Answers capped at ~800 tokens; very long documents are chunked so only the most relevant passages are used
- English stopwords tuned for English documents; other languages still work, slightly less precisely

## Common errors

| Message | What to do |
|---|---|
| `No document loaded` | Upload a PDF first, then ask. |
| `No API key configured` | Click **Settings** and paste your key, or set up `.env`. |
| `That key was rejected by Groq` | Re-copy the key from https://console.groq.com/keys (no extra spaces). |
| `No readable text could be extracted` | Scanned/image-only PDF — use a text-based PDF (no OCR included). |
| `This file could not be read as a PDF` | File is corrupt or not a real PDF — re-export it. |
| `Unsupported file type` | Only `.pdf` files are accepted. |
| `PDF is too large` | Split the PDF or lower `MAX_PDF_MB` limits in `app.py`. |
| `I couldn't find that information in this document` | Not an error — the answer genuinely isn't in the PDF. Rephrase or check the document. |
| `Groq rate limit reached` | Wait ~1 minute and retry. |
| `Model ... was not found` | Groq renamed the model — check https://console.groq.com/docs/models, update `GROQ_MODEL`. |

## Security

- Your `.env` file contains a **secret API key**. Never share it, never commit it, never paste it into screenshots or videos. `.gitignore` excludes `.env` and `*.pdf`.
- The Settings key lives only in server memory: never on disk, never in logs, never in browser storage, never echoed back in any API response.
- Uploaded PDFs are never written to disk and filenames are never executed — used for display only.
- AI output is HTML-escaped before rendering; oversized uploads are rejected before being read into memory.
- If a key ever leaks, delete it at https://console.groq.com/keys and create a new one.

## Customization

- **Change the model:** set `GROQ_MODEL` in `.env` (e.g. `openai/gpt-oss-20b` for faster/cheaper answers).
- **Retrieval depth:** change `TOP_K = 4` in `app.py` (more chunks = more context, slower and pricier).
- **Chunk size:** tune `CHUNK_TARGET` / `CHUNK_OVERLAP` in `app.py`.
- **Answer length:** raise `max_tokens=800` in `api_chat`.
- **Upload caps:** `MAX_PDF_MB` in `.env`; `MAX_PAGES` / `MAX_TOTAL_CHARS` in `app.py`.
- **Tone:** edit `SYSTEM_PROMPT` in `app.py` (keep the grounding rules unless you know what you're doing).
- **Theme:** CSS variables at the top of `static/styles.css`; headings use a serif display stack (`--display`).
