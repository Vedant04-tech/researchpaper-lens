"""Unit tests for src/retrieval.py (TDD: RED -> GREEN -> REFACTOR)."""

import pytest
from src.retrieval import (
    DEFAULT_PASSAGE_SENTENCE_COUNT,
    DEFAULT_TOP_K,
    Passage,
    PassageRetrievalIndex,
    RetrievalResult,
    create_passages,
)


@pytest.fixture
def sample_sentences():
    return [
        "Sentence zero introduces the research problem in information retrieval.",
        "Sentence one outlines the dataset and text processing pipeline.",
        "Sentence two details tokenization and stop word removal techniques.",
        "Sentence three introduces graph-based ranking and the PageRank algorithm.",
        "Sentence four discusses vertex centrality and damping factor parameters.",
        "Sentence five evaluates sentence extraction and compression ratio performance.",
        "Sentence six presents precision and mean reciprocal rank evaluation metrics.",
        "Sentence seven concludes the study with future research directions.",
    ]


def test_create_passages_grouping(sample_sentences):
    passages = create_passages(sample_sentences, sentence_count=3)
    assert len(passages) == 3
    assert passages[0].passage_id == 0
    assert passages[0].sentence_indices == [0, 1, 2]
    assert passages[1].passage_id == 1
    assert passages[1].sentence_indices == [3, 4, 5]
    assert passages[2].passage_id == 2
    assert passages[2].sentence_indices == [6, 7]


def test_create_passages_short_document():
    sentences = ["Only one sentence here."]
    passages = create_passages(sentences, sentence_count=3)
    assert len(passages) == 1
    assert passages[0].sentence_indices == [0]


def test_create_passages_empty():
    assert create_passages([]) == []


def test_retrieve_passages_basic(sample_sentences):
    index = PassageRetrievalIndex(sample_sentences, passage_size=3)
    results = index.search("PageRank graph centrality", top_k=3)
    assert len(results) > 0
    # Passage 1 contains "PageRank", "graph", "centrality"
    assert results[0].passage_id == 1
    assert results[0].score > 0.0
    assert "PageRank" in results[0].text


def test_retrieve_passages_top_k(sample_sentences):
    index = PassageRetrievalIndex(sample_sentences, passage_size=3)
    results = index.search("sentence", top_k=2)
    assert len(results) <= 2


def test_retrieve_passages_oov_query(sample_sentences):
    index = PassageRetrievalIndex(sample_sentences, passage_size=3)
    # Complete out-of-vocabulary query
    results = index.search("xylophone zeppelin aardvark", top_k=3)
    assert len(results) == 0


def test_retrieve_passages_blank_query(sample_sentences):
    index = PassageRetrievalIndex(sample_sentences, passage_size=3)
    assert index.search("", top_k=5) == []
    assert index.search("   \t\n  ", top_k=5) == []


def test_retrieve_passages_empty_index():
    index = PassageRetrievalIndex([])
    assert index.search("test query") == []
