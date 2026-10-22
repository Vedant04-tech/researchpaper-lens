"""Unit tests for src/textrank.py (TDD: RED -> GREEN -> REFACTOR)."""

import pytest
from src.textrank import (
    DEFAULT_SUMMARY_RATIO,
    MIN_SENTENCE_WORDS,
    PAGERANK_DAMPING,
    PAGERANK_MAX_ITER,
    PAGERANK_TOLERANCE,
    TextRankResult,
    textrank_summarize,
)


@pytest.fixture
def sample_sentences():
    return [
        "Information retrieval is the science of searching for relevant documents in large collections.",
        "Automated indexing systems dramatically reduce human search effort and information overload.",
        "Search engines index billions of web pages using inverted indexes and term frequencies.",
        "Extractive summarization identifies the most salient sentences directly from the original document text.",
        "Graph ranking algorithms like PageRank evaluate vertex importance recursively from graph topology.",
        "TextRank constructs a graph of sentences connected by pairwise lexical similarity edges.",
        "Sentences with higher centrality scores are selected to form a cohesive summary representation.",
        "Finally the selected sentences are presented in their original document order for readability.",
    ]


def test_textrank_single_sentence():
    sentences = ["Information retrieval systems process queries against indexed documents."]
    result = textrank_summarize(sentences)
    assert isinstance(result, TextRankResult)
    assert result.summary == sentences[0]
    assert len(result.selected_indices) == 1
    assert result.selected_indices[0] == 0


def test_textrank_empty_input():
    result = textrank_summarize([])
    assert result.summary == ""
    assert result.selected_indices == []
    assert len(result.scores) == 0


def test_textrank_normal_document(sample_sentences):
    result = textrank_summarize(sample_sentences, ratio=0.25)
    # 8 sentences * 0.25 = 2 sentences
    assert len(result.selected_indices) == 2
    # Verify selected sentences exist in source
    for idx in result.selected_indices:
        assert sample_sentences[idx] in result.summary


def test_textrank_preserves_order(sample_sentences):
    result = textrank_summarize(sample_sentences, ratio=0.5)
    # Selected indices must be strictly sorted in ascending order
    assert result.selected_indices == sorted(result.selected_indices)
    assert len(set(result.selected_indices)) == len(result.selected_indices)


def test_textrank_min_sentence_words_filter():
    sentences = [
        "Hi.",  # 1 word
        "Short title.",  # 2 words
        "Still short heading here.",  # 4 words
        "This is an eligible sentence that meets the five word minimum requirement.",  # 11 words
        "Another long sentence discussing natural language processing algorithms in depth.",  # 10 words
    ]
    result = textrank_summarize(sentences, ratio=0.5)
    # The short ones (< 5 words) should be excluded from selection if longer ones exist
    for idx in result.selected_indices:
        assert idx in [3, 4]


def test_textrank_all_short_sentences_fallback():
    sentences = [
        "Introduction here.",
        "Method section.",
        "Experimental results.",
        "Conclusion summary.",
    ]
    # All are < 5 words; must not crash and should fall back deterministically
    result = textrank_summarize(sentences, ratio=0.5)
    assert len(result.selected_indices) > 0
    assert result.summary != ""


def test_textrank_zero_overlap_sentences():
    # Disjoint vocabulary with zero similarity between sentences
    sentences = [
        "Apples bananas cherries oranges grapefruits kiwis lemons.",
        "Quantum physics relativistic mechanics electrodynamics thermodynamics optics.",
        "Renaissance sculpture architecture fresco baroque neoclassicism.",
    ]
    result = textrank_summarize(sentences, ratio=0.30)
    # Should handle zero-edge graph deterministically without crashing (ceil(0.30*3) = 1)
    assert len(result.selected_indices) == 1
    assert result.summary in sentences



def test_textrank_duplicate_sentences():
    sentences = [
        "Graph based ranking algorithms compute importance recursively across vertices.",
        "Graph based ranking algorithms compute importance recursively across vertices.",
        "Unrelated document talking about organic chemistry synthesis and enzymes.",
    ]
    result = textrank_summarize(sentences, ratio=0.66)
    assert len(result.selected_indices) == 2


def test_textrank_ratio_clamping(sample_sentences):
    # Ratio < 0.10 clamped to 0.10
    res_low = textrank_summarize(sample_sentences, ratio=0.01)
    assert len(res_low.selected_indices) >= 1

    # Ratio > 0.50 clamped to 0.50
    res_high = textrank_summarize(sample_sentences, ratio=0.99)
    # 8 * 0.5 = 4 sentences
    assert len(res_high.selected_indices) <= 4
