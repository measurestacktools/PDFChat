"""Unit tests for the local RAG pipeline: chunking, cleaning, TF-IDF retrieval.

Run:  pytest tests/test_rag.py
These tests need no API key and no network.
"""
from app import TfIdfIndex, _clean_text, _split_scope, _tokenize, chunk_page


def test_tokenize_lowercases_and_drops_stopwords():
    toks = _tokenize("The QUICK brown fox jumps 42 times!")
    assert "quick" in toks and "brown" in toks and "42" in toks
    assert "the" not in toks and "and" not in toks


def test_clean_text_collapses_whitespace():
    assert _clean_text("hello   \n\n\n  world") == "hello\n\nworld"


def test_chunk_page_tracks_real_pages():
    text = ("First sentence about cats. Second sentence about cats too. "
            "Third sentence still about cats and dogs.")
    chunks = chunk_page(text, page=7, target=60, overlap=10)
    assert chunks, "expected at least one chunk"
    assert all(c["page"] == 7 for c in chunks)
    # All original words survive across chunks (overlap may duplicate, never drop).
    joined = " ".join(c["text"] for c in chunks)
    for word in ["First", "Second", "Third", "cats", "dogs"]:
        assert word in joined


def test_chunk_page_skips_tiny_text():
    assert chunk_page("  hi  ", page=1) == []


def test_tfidf_ranks_relevant_chunk_first():
    chunks = [
        {"text": "Cats are independent pets that groom themselves daily.", "page": 1},
        {"text": "The quarterly revenue grew by twelve percent in March.", "page": 2},
        {"text": "Dogs need regular walks and open spaces to stay healthy.", "page": 3},
    ]
    index = TfIdfIndex(chunks)
    hits = index.search("How much did revenue grow in March?")
    assert hits, "expected retrieval hits"
    best_idx, best_score = hits[0]
    assert best_idx == 1, f"expected chunk 1 first, got {best_idx}"
    assert best_score > 0


def test_tfidf_no_shared_vocabulary_returns_empty():
    chunks = [{"text": "Cats groom themselves daily with great care.", "page": 1}]
    index = TfIdfIndex(chunks)
    assert index.search("Xylophone quantum zebra petroleum") == []


def test_tfidf_scores_are_bounded_cosine():
    chunks = [
        {"text": "Solar panels convert sunlight into electricity efficiently.", "page": 1},
        {"text": "Solar panel installation costs dropped sharply this year.", "page": 2},
    ]
    index = TfIdfIndex(chunks)
    for _, score in index.search("solar panels electricity"):
        assert 0.0 < score <= 1.0


def test_split_scope_detects_general_prefix():
    from app import GENERAL_PREFIX
    is_general, clean = _split_scope(GENERAL_PREFIX + "\nParis is the capital of France.")
    assert is_general is True
    assert clean == "Paris is the capital of France."
    # No page sources may ever ride along with a general answer.


def test_split_scope_keeps_document_answers():
    is_general, clean = _split_scope("The launch was March 14, 2026 [p. 1].")
    assert is_general is False
    assert clean == "The launch was March 14, 2026 [p. 1]."


def test_split_scope_neutralizes_legacy_refusal():
    is_general, clean = _split_scope("I couldn't find that information in this document.")
    assert is_general is True
    assert "couldn't find" in clean
