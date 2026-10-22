"""Unit tests for src/keywords.py (TDD: RED -> GREEN -> REFACTOR)."""

import pytest
from src.keywords import (
    DEFAULT_KEYWORD_COUNT,
    Keyword,
    extract_keywords,
)


@pytest.fixture
def sample_sentences():
    return [
        "Information retrieval systems process search queries against indexed documents.",
        "TF-IDF weighting combines term frequency with inverse document frequency.",
        "Cosine similarity measures the angle between term vector representations.",
        "PageRank computes recursive importance scores over graph vertices.",
        "Extractive summarization produces compact summaries without generating new text.",
    ]


def test_extract_keywords_basic(sample_sentences):
    keywords = extract_keywords(sample_sentences, count=5)
    assert len(keywords) == 5
    terms = [kw.term for kw in keywords]
    assert all(isinstance(kw, Keyword) for kw in keywords)
    assert all(kw.score > 0 for kw in keywords)
    # Check that high frequency domain terms are present
    assert any(term in terms for term in ["frequency", "similarity", "pagerank", "retrieval", "summarization", "term"])


def test_extract_keywords_count(sample_sentences):
    keywords = extract_keywords(sample_sentences, count=3)
    assert len(keywords) == 3


def test_extract_keywords_stop_words(sample_sentences):
    keywords = extract_keywords(sample_sentences, count=10)
    terms = [kw.term.lower() for kw in keywords]
    stop_words = {"the", "is", "and", "with", "in", "without", "against", "over"}
    assert not any(sw in terms for sw in stop_words)


def test_extract_keywords_empty():
    assert extract_keywords([]) == []
    assert extract_keywords(["   ", ""]) == []


def test_extract_keywords_single_sentence():
    sentences = ["Information retrieval indexing algorithms evaluate inverted document tables."]
    keywords = extract_keywords(sentences, count=5)
    assert len(keywords) > 0
    assert all(kw.score > 0 for kw in keywords)
