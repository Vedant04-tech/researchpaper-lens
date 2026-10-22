"""ResearchPaper Lens — TextRank Extractive Summarizer.

Implements TextRank-style graph-based extractive summarization:
- Sentence-level TF-IDF vectorization and pairwise cosine similarity
- Undirected weighted sentence graph
- NetworkX PageRank with explicit damping, iteration, and tolerance settings
- Bounded summary ratio [0.10, 0.50]
- Minimum sentence length filter (>= 5 words) with deterministic fallback
- Chronological source order preservation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Architectural defaults
DEFAULT_SUMMARY_RATIO: float = 0.25
MIN_SUMMARY_RATIO: float = 0.10
MAX_SUMMARY_RATIO: float = 0.50
MIN_SENTENCE_WORDS: int = 5
PAGERANK_DAMPING: float = 0.85
PAGERANK_MAX_ITER: int = 100
PAGERANK_TOLERANCE: float = 1e-6


@dataclass
class TextRankResult:
    """Result from TextRank extractive summarization."""

    summary: str
    selected_indices: list[int] = field(default_factory=list)
    scores: dict[int, float] = field(default_factory=dict)
    ratio: float = DEFAULT_SUMMARY_RATIO


def _filter_eligible_sentences(
    sentences: Sequence[str], min_words: int = MIN_SENTENCE_WORDS
) -> list[int]:
    """Returns indices of sentences meeting the minimum word length requirement.

    If no sentences meet the threshold, falls back to all non-empty sentences.
    """
    eligible = [
        idx
        for idx, s in enumerate(sentences)
        if len(s.strip().split()) >= min_words
    ]
    if not eligible:
        # Fallback: all non-empty sentences
        eligible = [idx for idx, s in enumerate(sentences) if s.strip()]
    return eligible


def textrank_summarize(
    sentences: Sequence[str],
    ratio: float = DEFAULT_SUMMARY_RATIO,
    min_words: int = MIN_SENTENCE_WORDS,
    damping: float = PAGERANK_DAMPING,
    max_iter: int = PAGERANK_MAX_ITER,
    tol: float = PAGERANK_TOLERANCE,
) -> TextRankResult:
    """Extractively summarizes a list of sentences using TextRank graph ranking.

    Args:
        sentences: Ordered sequence of sentences from the preprocessed document.
        ratio: Target compression ratio, clamped to [0.10, 0.50].
        min_words: Minimum whitespace-delimited words for graph eligibility.
        damping: PageRank damping parameter (default 0.85).
        max_iter: Maximum PageRank iterations (default 100).
        tol: Convergence tolerance (default 1e-6).

    Returns:
        TextRankResult with summary text, selected indices, and score dict.
    """
    if not sentences:
        return TextRankResult(summary="", selected_indices=[], scores={}, ratio=ratio)

    if len(sentences) == 1:
        return TextRankResult(
            summary=sentences[0],
            selected_indices=[0],
            scores={0: 1.0},
            ratio=ratio,
        )

    # Clamp ratio to approved bounds [0.10, 0.50]
    clamped_ratio = max(MIN_SUMMARY_RATIO, min(MAX_SUMMARY_RATIO, float(ratio)))

    # Filter eligible candidate sentences
    eligible_indices = _filter_eligible_sentences(sentences, min_words=min_words)
    candidate_texts = [sentences[idx] for idx in eligible_indices]
    num_candidates = len(candidate_texts)

    if num_candidates == 0:
        return TextRankResult(summary="", selected_indices=[], scores={}, ratio=clamped_ratio)

    if num_candidates == 1:
        idx = eligible_indices[0]
        return TextRankResult(
            summary=sentences[idx],
            selected_indices=[idx],
            scores={idx: 1.0},
            ratio=clamped_ratio,
        )

    # Compute target number of sentences: k = min(n, max(1, ceil(ratio * n)))
    k = min(num_candidates, max(1, math.ceil(clamped_ratio * num_candidates)))

    # Vectorize candidate sentences with TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(candidate_texts)
        sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    except ValueError:
        # Happens if all candidate texts contain only stop words
        sim_matrix = np.zeros((num_candidates, num_candidates))

    # Construct weighted undirected sentence graph
    graph = nx.Graph()
    for i in range(num_candidates):
        graph.add_node(i)

    for i in range(num_candidates):
        for j in range(i + 1, num_candidates):
            weight = float(sim_matrix[i, j])
            if weight > 0.0:
                graph.add_edge(i, j, weight=weight)

    # Compute PageRank scores
    try:
        if graph.number_of_edges() == 0:
            # Deterministic fallback when sentences have no lexical overlap
            candidate_scores = {i: 1.0 / num_candidates for i in range(num_candidates)}
        else:
            candidate_scores = nx.pagerank(
                graph,
                alpha=damping,
                max_iter=max_iter,
                tol=tol,
                weight="weight",
            )
    except (nx.PowerIterationFailedConvergence, Exception):
        # Fallback to degree or uniform centrality
        candidate_scores = {i: 1.0 / num_candidates for i in range(num_candidates)}

    # Map candidate indices back to original sentence indices
    scores_by_original_idx: dict[int, float] = {
        eligible_indices[c_idx]: score for c_idx, score in candidate_scores.items()
    }

    # Rank eligible sentences: sort by score descending, breaking ties by earlier position
    ranked_candidates = sorted(
        candidate_scores.keys(),
        key=lambda c_idx: (-candidate_scores[c_idx], c_idx),
    )

    top_candidate_indices = ranked_candidates[:k]

    # Restore original document order
    selected_original_indices = sorted(
        [eligible_indices[c_idx] for c_idx in top_candidate_indices]
    )

    selected_sentences = [sentences[idx] for idx in selected_original_indices]
    summary_text = " ".join(selected_sentences)

    return TextRankResult(
        summary=summary_text,
        selected_indices=selected_original_indices,
        scores=scores_by_original_idx,
        ratio=clamped_ratio,
    )
