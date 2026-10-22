"""ResearchPaper Lens — Baseline Algorithms for Academic Comparison.

Implements standard academic IR and NLP baselines required by the curriculum:
1. Lead-k sentence extraction summarizer
2. Word-frequency sentence ranking summarizer
3. Exact keyword-matching passage retrieval
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.retrieval import Passage, RetrievalResult
from src.textrank import (
    DEFAULT_SUMMARY_RATIO,
    MAX_SUMMARY_RATIO,
    MIN_SENTENCE_WORDS,
    MIN_SUMMARY_RATIO,
    _filter_eligible_sentences,
)


@dataclass
class BaselineSummaryResult:
    """Result from a baseline summarization method."""

    summary: str
    selected_indices: list[int] = field(default_factory=list)
    method: str = "baseline"
    ratio: float = DEFAULT_SUMMARY_RATIO


def lead_k_summarize(
    sentences: Sequence[str],
    ratio: float = DEFAULT_SUMMARY_RATIO,
    min_words: int = MIN_SENTENCE_WORDS,
) -> BaselineSummaryResult:
    """Selects the first k eligible sentences in the document (Lead-k baseline).

    Args:
        sentences: Ordered sequence of document sentences.
        ratio: Target compression ratio in [0.10, 0.50].
        min_words: Minimum words per eligible sentence.

    Returns:
        BaselineSummaryResult with extracted leading sentences.
    """
    if not sentences:
        return BaselineSummaryResult(summary="", selected_indices=[], method="lead-k", ratio=ratio)

    clamped_ratio = max(MIN_SUMMARY_RATIO, min(MAX_SUMMARY_RATIO, float(ratio)))
    eligible = _filter_eligible_sentences(sentences, min_words=min_words)
    num_candidates = len(eligible)

    if num_candidates == 0:
        return BaselineSummaryResult(summary="", selected_indices=[], method="lead-k", ratio=clamped_ratio)

    k = min(num_candidates, max(1, math.ceil(clamped_ratio * num_candidates)))
    selected_indices = eligible[:k]
    summary_text = " ".join([sentences[idx] for idx in selected_indices])

    return BaselineSummaryResult(
        summary=summary_text,
        selected_indices=selected_indices,
        method="lead-k",
        ratio=clamped_ratio,
    )


def word_frequency_summarize(
    sentences: Sequence[str],
    ratio: float = DEFAULT_SUMMARY_RATIO,
    min_words: int = MIN_SENTENCE_WORDS,
) -> BaselineSummaryResult:
    """Scores sentences by normalized word frequencies of non-stop-words.

    Args:
        sentences: Ordered sequence of document sentences.
        ratio: Target compression ratio in [0.10, 0.50].
        min_words: Minimum words per eligible sentence.

    Returns:
        BaselineSummaryResult with top frequency-scored sentences in source order.
    """
    if not sentences:
        return BaselineSummaryResult(
            summary="", selected_indices=[], method="word-frequency", ratio=ratio
        )

    clamped_ratio = max(MIN_SUMMARY_RATIO, min(MAX_SUMMARY_RATIO, float(ratio)))
    eligible = _filter_eligible_sentences(sentences, min_words=min_words)
    num_candidates = len(eligible)

    if num_candidates == 0:
        return BaselineSummaryResult(
            summary="", selected_indices=[], method="word-frequency", ratio=clamped_ratio
        )

    k = min(num_candidates, max(1, math.ceil(clamped_ratio * num_candidates)))

    # Compute global term frequencies across all sentences
    word_counter: Counter[str] = Counter()
    for s in sentences:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", s.lower())
        for w in words:
            if w not in ENGLISH_STOP_WORDS:
                word_counter[w] += 1

    max_freq = max(word_counter.values()) if word_counter else 1

    # Score each eligible sentence
    scored_candidates: list[tuple[int, float]] = []
    for orig_idx in eligible:
        s_words = re.findall(r"\b[a-zA-Z]{3,}\b", sentences[orig_idx].lower())
        meaningful_words = [w for w in s_words if w not in ENGLISH_STOP_WORDS]
        if not meaningful_words:
            scored_candidates.append((orig_idx, 0.0))
            continue

        raw_score = sum(word_counter[w] / max_freq for w in meaningful_words)
        normalized_score = raw_score / len(s_words)
        scored_candidates.append((orig_idx, normalized_score))

    # Rank by score descending, tie-break by original index
    scored_candidates.sort(key=lambda item: (-item[1], item[0]))
    top_indices = sorted([item[0] for item in scored_candidates[:k]])

    summary_text = " ".join([sentences[idx] for idx in top_indices])

    return BaselineSummaryResult(
        summary=summary_text,
        selected_indices=top_indices,
        method="word-frequency",
        ratio=clamped_ratio,
    )


def keyword_match_retrieve(
    passages: Sequence[Passage],
    query: str,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """Retrieves passages using exact case-insensitive query token matching.

    Args:
        passages: List of document Passage objects.
        query: Query string.
        top_k: Maximum ranked passages to return.

    Returns:
        List of RetrievalResult objects ordered by overlap ratio descending.
    """
    if not query or not query.strip() or not passages:
        return []

    tokens = [t.lower() for t in re.findall(r"\b[a-zA-Z0-9]+\b", query)]
    if not tokens:
        return []

    scored: list[tuple[int, float]] = []
    total_tokens = len(tokens)

    for p in passages:
        p_text_lower = p.text.lower()
        # Count matched tokens
        matches = sum(1 for t in tokens if re.search(rf"\b{re.escape(t)}\b", p_text_lower))
        score = matches / total_tokens
        if score > 0.0:
            scored.append((p.passage_id, score))

    scored.sort(key=lambda item: (-item[1], item[0]))

    results: list[RetrievalResult] = []
    for passage_id, score in scored[:top_k]:
        passage = next(p for p in passages if p.passage_id == passage_id)
        results.append(
            RetrievalResult(
                passage_id=passage.passage_id,
                text=passage.text,
                score=round(score, 4),
                sentence_indices=passage.sentence_indices,
            )
        )

    return results
