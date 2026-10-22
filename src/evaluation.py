"""ResearchPaper Lens — IR & Summarization Evaluation Metrics Module.

Implements academic Information Retrieval and text summarization evaluation metrics:
1. Precision@k (P@k)
2. Mean Reciprocal Rank (MRR)
3. Document compression ratio
4. Local ROUGE-n (ROUGE-1, ROUGE-2) recall, precision, and F1 calculations
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class RougeScore:
    """Represents ROUGE precision, recall, and F1 score."""

    precision: float
    recall: float
    f1: float


def calculate_precision_at_k(
    retrieved: Sequence[Any], relevant: set[Any], k: int = 5
) -> float:
    """Calculates Precision@k: proportion of top-k retrieved items that are relevant.

    Args:
        retrieved: Ordered list of retrieved passage/document identifiers.
        relevant: Set of ground-truth relevant identifiers.
        k: Cut-off rank.

    Returns:
        Precision score in [0.0, 1.0].
    """
    if k <= 0 or not retrieved or not relevant:
        return 0.0

    top_k = retrieved[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def calculate_mrr(
    queries_retrieved_and_relevant: Sequence[tuple[Sequence[Any], set[Any]]],
) -> float:
    """Calculates Mean Reciprocal Rank (MRR) across a collection of queries.

    Args:
        queries_retrieved_and_relevant: Sequence of tuples, each containing
            an ordered sequence of retrieved IDs and a set of relevant IDs.

    Returns:
        MRR score in [0.0, 1.0].
    """
    if not queries_retrieved_and_relevant:
        return 0.0

    reciprocal_ranks: list[float] = []

    for retrieved, relevant in queries_retrieved_and_relevant:
        rr = 0.0
        for rank, item in enumerate(retrieved, start=1):
            if item in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_compression_ratio(summary_text: str, source_text: str) -> float:
    """Computes the length compression ratio between summary and original source.

    Args:
        summary_text: Summary text string.
        source_text: Full original document text string.

    Returns:
        Ratio (summary words / source words).
    """
    summary_words = len(summary_text.strip().split())
    source_words = len(source_text.strip().split())

    if source_words == 0 or summary_words == 0:
        return 0.0

    return summary_words / source_words


def _get_ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    """Extracts contiguous n-grams from a word list."""
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def calculate_rouge_n(candidate: str, reference: str, n: int = 1) -> RougeScore:
    """Computes local ROUGE-n precision, recall, and F1 between candidate and reference texts.

    Args:
        candidate: Generated summary text.
        reference: Ground-truth reference summary text.
        n: N-gram length (1 for ROUGE-1, 2 for ROUGE-2).

    Returns:
        RougeScore containing recall, precision, and F1.
    """
    cand_words = re.findall(r"\b\w+\b", candidate.lower())
    ref_words = re.findall(r"\b\w+\b", reference.lower())

    cand_ngrams = _get_ngrams(cand_words, n)
    ref_ngrams = _get_ngrams(ref_words, n)

    if not ref_ngrams or not cand_ngrams:
        return RougeScore(precision=0.0, recall=0.0, f1=0.0)

    cand_counter = Counter(cand_ngrams)
    ref_counter = Counter(ref_ngrams)

    # Compute overlapping n-grams (clipped to reference count)
    overlap = sum(min(cand_counter[ng], ref_counter[ng]) for ng in cand_counter if ng in ref_counter)

    recall = overlap / len(ref_ngrams)
    precision = overlap / len(cand_ngrams)

    if precision + recall > 0.0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return RougeScore(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
    )
