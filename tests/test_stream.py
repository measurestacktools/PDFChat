"""SSE streaming tests for POST /api/chat/stream (mocked Groq, no key needed).

Protocol:
  - token chunks:  data: {"delta": "<token>"}
  - final event:   event: done / data: {answer, sources, chunks_used, model}
                   — SAME json shape as POST /api/chat
  - errors:        event: error / data: {"error": "..."}

Run: pytest tests/test_stream.py
"""
import json

from fastapi.testclient import TestClient

import app as m


FULL_ANSWER = "Cats sit on page one and dogs run on page two."


def _seed_doc():
    chunks = [
        {"text": "Cats sit quietly on page one of this manual.", "page": 1},
        {"text": "Dogs run loudly on page two of this manual.", "page": 2},
    ]
    m._doc = {
        "doc_id": "testdoc",
        "filename": "test.pdf",
        "size_bytes": 100,
        "pages": 2,
        "chunks": len(chunks),
        "chars": sum(len(c["text"]) for c in chunks),
        "chunks_data": chunks,
        "index": m.TfIdfIndex(chunks),
        "history": [],
    }
    m._session_key = "gsk_test_key"


class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content):
        self.choices = [_Choice(content)]


class _Msg:
    def __init__(self, content):
        self.content = content


class _NonStreamChoice:
    def __init__(self, content):
        self.message = _Msg(content)


class _Completions:
    def __init__(self, tokens):
        self._tokens = tokens

    def create(self, **kwargs):
        if kwargs.get("stream"):
            tokens = self._tokens
            assert kwargs.get("model") == m.GROQ_MODEL

            def gen():
                for t in tokens:
                    yield _Chunk(t)

            return gen()
        # non-streaming path used by /api/chat
        class _Resp:
            choices = [_NonStreamChoice(FULL_ANSWER)]

        return _Resp()


class _Chat:
    def __init__(self, tokens):
        self.completions = _Completions(tokens)


class _FakeOpenAI:
    TOKENS = ["Cats sit ", "on page one ", "and dogs run ", "on page two."]

    def __init__(self, *a, **k):
        self.chat = _Chat(list(self.TOKENS))


def _client():
    return TestClient(m.app)


def _collect_sse(resp):
    """Parse an SSE byte stream into (deltas, done_payload, error_payload)."""
    text = resp.text
    deltas: list[str] = []
    done = None
    error = None
    for raw_event in text.split("\n\n"):
        raw_event = raw_event.strip()
        if not raw_event:
            continue
        ev_name = "message"
        data_str = ""
        for ln in raw_event.split("\n"):
            if ln.startswith("event:"):
                ev_name = ln[len("event:"):].strip()
            elif ln.startswith("data:"):
                data_str += ln[len("data:"):].strip()
        if not data_str:
            continue
        payload = json.loads(data_str)
        if ev_name == "done":
            done = payload
        elif ev_name == "error":
            error = payload
        else:
            if "delta" in payload:
                deltas.append(payload["delta"])
    return deltas, done, error


def test_stream_validation_parity(monkeypatch):
    c = _client()
    # No doc -> 400, same as /api/chat
    m._doc = None
    m._session_key = "gsk_test_key"
    assert c.post("/api/chat/stream", json={"question": ""}).status_code == 400
    assert c.post("/api/chat/stream", json={"question": "hi"}).status_code == 400
    assert c.post("/api/chat", json={"question": "hi"}).status_code == 400


def test_stream_done_matches_chat_shape(monkeypatch):
    monkeypatch.setattr(m, "OpenAI", _FakeOpenAI)
    c = _client()

    _seed_doc()
    r_chat = c.post("/api/chat", json={"question": "cats and dogs"})
    assert r_chat.status_code == 200, r_chat.text
    chat_body = r_chat.json()
    assert set(chat_body.keys()) == {"answer", "sources", "chunks_used", "model"}

    _seed_doc()  # reset history so both calls see identical state
    r = c.post("/api/chat/stream", json={"question": "cats and dogs"})
    assert r.status_code == 200, r.text
    assert "text/event-stream" in r.headers["content-type"]

    deltas, done, error = _collect_sse(r)
    assert error is None, error
    assert deltas, "expected at least one token chunk"
    assert "".join(deltas) == FULL_ANSWER
    assert done is not None, "expected a final done event"
    # done-event json matches /api/chat shape exactly
    assert set(done.keys()) == {"answer", "sources", "chunks_used", "model"}
    assert done["answer"] == FULL_ANSWER == chat_body["answer"]
    assert done["sources"] == chat_body["sources"]
    assert done["chunks_used"] == chat_body["chunks_used"]
    assert done["model"] == chat_body["model"]
    # streaming persists history just like /api/chat
    assert m._doc["history"] and m._doc["history"][-1]["a"] == FULL_ANSWER
