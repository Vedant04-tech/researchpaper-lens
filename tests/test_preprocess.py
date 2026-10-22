"""Unit tests for src/preprocess.py (TDD: RED -> GREEN -> REFACTOR)."""

import pytest
from src.preprocess import (
    clean_text,
    split_sentences,
    split_references,
    preprocess_document,
    PreprocessedDocument,
)


def test_clean_text_whitespace():
    raw = "Information   retrieval   systems\n\n\n\nare   useful.\t\tVery   useful."
    cleaned = clean_text(raw)
    assert "   " not in cleaned
    assert "\t" not in cleaned
    assert "\n\n\n" not in cleaned
    assert "Information retrieval systems\n\nare useful. Very useful." == cleaned


def test_clean_text_hyphenation():
    raw = "We study the infor-\nmation retrieval and docu-\nment indexing process."
    cleaned = clean_text(raw)
    assert "information" in cleaned
    assert "document" in cleaned


def test_split_sentences_basic():
    text = "First sentence is here. Second sentence follows! Third one is a question?"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence is here."
    assert sentences[1] == "Second sentence follows!"
    assert sentences[2] == "Third one is a question?"


def test_split_sentences_protected_abbreviations():
    text = (
        "Prof. Smith et al. introduced a method (e.g. TextRank, i.e. PageRank for text). "
        "See Fig. 2 and Eq. 4 for the formula with pi = 3.14. "
        "The results vs. baseline are statistically significant."
    )
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("Prof. Smith et al. introduced")
    assert "e.g. TextRank" in sentences[0]
    assert "i.e. PageRank" in sentences[0]
    assert sentences[1].startswith("See Fig. 2")
    assert "pi = 3.14" in sentences[1]
    assert sentences[2].startswith("The results vs. baseline")


def test_split_sentences_empty_or_whitespace():
    assert split_sentences("") == []
    assert split_sentences("   \n\t   ") == []


def test_split_references_present():
    body = "This is the main content of the paper with key findings and analysis."
    refs = "[1] Page, L. et al. The PageRank citation ranking.\n[2] Mihalcea, R. TextRank."
    full_text = f"{body}\n\nReferences\n\n{refs}"

    content_body, references = split_references(full_text)
    assert content_body.strip() == body
    assert references is not None
    assert "[1] Page" in references


def test_split_references_absent():
    text = "A paper without references section."
    content_body, references = split_references(text)
    assert content_body == text
    assert references is None


def test_preprocess_document():
    raw = (
        "Information Retrieval is essential.\n\n"
        "Modern search engines use TF-IDF and graph algorithms.\n\n"
        "References\n\n"
        "[1] Salton, G. Introduction to Modern Information Retrieval."
    )
    doc = preprocess_document(raw)
    assert isinstance(doc, PreprocessedDocument)
    assert doc.canonical_text == raw
    assert len(doc.sentences) == 2
    assert doc.word_count > 0
    assert doc.references_text is not None
    assert "[1] Salton" in doc.references_text
