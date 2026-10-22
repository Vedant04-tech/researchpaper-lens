"""Unit tests for src/baselines.py (TDD: RED -> GREEN -> REFACTOR)."""

import pytest
from src.baselines import (
    lead_k_summarize,
    word_frequency_summarize,
    keyword_match_retrieve,
    BaselineSummaryResult,
)
from src.retrieval import create_passages


@pytest.fixture
def sample_sentences():
    return [
        "First sentence introduces the primary concepts in information retrieval.",
        "Second sentence discusses modern search engines and indexing systems.",
        "Third sentence details query parsing and term weighting models.",
        "Fourth sentence reviews graph algorithms and PageRank node importance.",
        "Fifth sentence summarizes findings and potential future research directions.",
    ]


def test_lead_k_summarize(sample_sentences):
    # 5 sentences * 0.40 = 2 sentences
    result = lead_k_summarize(sample_sentences, ratio=0.40)
    assert isinstance(result, BaselineSummaryResult)
    assert len(result.selected_indices) == 2
    assert result.selected_indices == [0, 1]
    assert result.summary == f"{sample_sentences[0]} {sample_sentences[1]}"


def test_lead_k_empty():
    result = lead_k_summarize([])
    assert result.summary == ""
    assert result.selected_indices == []


def test_word_frequency_summarize(sample_sentences):
    result = word_frequency_summarize(sample_sentences, ratio=0.40)
    assert isinstance(result, BaselineSummaryResult)
    assert len(result.selected_indices) == 2
    # Verify order is preserved
    assert result.selected_indices == sorted(result.selected_indices)
    for idx in result.selected_indices:
        assert sample_sentences[idx] in result.summary


def test_keyword_match_retrieve(sample_sentences):
    passages = create_passages(sample_sentences, sentence_count=2)
    # Passages: [0, 1] -> 0, [2, 3] -> 1, [4] -> 2
    results = keyword_match_retrieve(passages, query="graph algorithms PageRank", top_k=2)
    assert len(results) > 0
    # Passage 1 contains "graph algorithms and PageRank"
    assert results[0].passage_id == 1
    assert results[0].score > 0.0


def test_keyword_match_retrieve_oov(sample_sentences):
    passages = create_passages(sample_sentences, sentence_count=2)
    results = keyword_match_retrieve(passages, query="xylophone zeppelin", top_k=2)
    assert len(results) == 0


def test_keyword_match_retrieve_blank(sample_sentences):
    passages = create_passages(sample_sentences, sentence_count=2)
    assert keyword_match_retrieve(passages, query="", top_k=5) == []
    assert keyword_match_retrieve(passages, query="   ", top_k=5) == []
