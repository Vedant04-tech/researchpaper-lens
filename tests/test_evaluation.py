"""Unit tests for src/evaluation.py (TDD: RED -> GREEN -> REFACTOR)."""

import pytest
from src.evaluation import (
    calculate_precision_at_k,
    calculate_mrr,
    calculate_compression_ratio,
    calculate_rouge_n,
    RougeScore,
)


def test_precision_at_k():
    retrieved = [10, 20, 30, 40, 50]
    relevant = {20, 40}
    p5 = calculate_precision_at_k(retrieved, relevant, k=5)
    assert p5 == 2 / 5

    p2 = calculate_precision_at_k(retrieved, relevant, k=2)
    # top 2 are [10, 20], 20 is relevant -> 1/2 = 0.5
    assert p2 == 0.5


def test_precision_at_k_empty():
    assert calculate_precision_at_k([], {1, 2}, k=5) == 0.0
    assert calculate_precision_at_k([1, 2], set(), k=5) == 0.0
    assert calculate_precision_at_k([1, 2], {1}, k=0) == 0.0


def test_mean_reciprocal_rank():
    # Query 1: first relevant at rank 2 -> RR = 1/2
    # Query 2: first relevant at rank 1 -> RR = 1/1
    # Query 3: no relevant in top k -> RR = 0
    query_evals = [
        ([1, 2, 3], {2}),
        ([4, 5, 6], {4, 6}),
        ([7, 8, 9], {10}),
    ]
    mrr = calculate_mrr(query_evals)
    expected = (0.5 + 1.0 + 0.0) / 3
    assert pytest.approx(mrr, rel=1e-3) == expected


def test_mean_reciprocal_rank_empty():
    assert calculate_mrr([]) == 0.0


def test_compression_ratio():
    summary = "This is a short summary text."  # 6 words
    source = "This is a much longer original source document that contains many more words than summary."  # 15 words
    ratio = calculate_compression_ratio(summary, source)
    assert ratio == 6 / 15
    assert calculate_compression_ratio("", source) == 0.0
    assert calculate_compression_ratio(summary, "") == 0.0


def test_rouge_n_unigram():
    candidate = "information retrieval is effective and fast"
    reference = "information retrieval is very fast"
    score = calculate_rouge_n(candidate, reference, n=1)
    assert isinstance(score, RougeScore)
    # Overlapping unigrams: "information", "retrieval", "is", "fast" (4 terms)
    # Reference has 5 words, candidate has 6 words
    assert pytest.approx(score.recall, rel=1e-3) == 4 / 5
    assert pytest.approx(score.precision, rel=1e-3) == 4 / 6
    assert score.f1 > 0.0



def test_rouge_n_bigram():
    candidate = "graph ranking algorithm"
    reference = "graph ranking model"
    score = calculate_rouge_n(candidate, reference, n=2)
    # Bigrams candidate: ("graph", "ranking"), ("ranking", "algorithm")
    # Bigrams reference: ("graph", "ranking"), ("ranking", "model")
    # Overlap: 1 ("graph", "ranking")
    assert score.recall == 1 / 2
    assert score.precision == 1 / 2
    assert score.f1 == 0.5
