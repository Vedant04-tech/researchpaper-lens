"""ResearchPaper Lens — Passage Indexing & TF-IDF Cosine Retrieval.

Implements deterministic adjacent-sentence passage grouping and lexical
retrieval using scikit-learn TF-IDF and cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_PASSAGE_SENTENCE_COUNT: int = 3
DEFAULT_TOP_K: int = 5
MIN_TOP_K: int = 1
MAX_TOP_K: int = 10


@dataclass
class Passage:
    """Represents a grouped block of adjacent sentences."""

    passage_id: int
    text: str
    sentence_indices: list[int] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Represents a ranked passage match."""

    passage_id: int
    text: str
    score: float
    sentence_indices: list[int] = field(default_factory=list)


def create_passages(
    sentences: Sequence[str], sentence_count: int = DEFAULT_PASSAGE_SENTENCE_COUNT
) -> list[Passage]:
    """Partitions an ordered list of sentences into deterministic adjacent passages.

    Args:
        sentences: Ordered sequence of sentences from preprocessed document.
        sentence_count: Number of adjacent sentences per passage (default 3).

    Returns:
        List of Passage objects.
    """
    if not sentences:
        return []

    passages: list[Passage] = []
    total = len(sentences)
    passage_id = 0

    for i in range(0, total, sentence_count):
        chunk = sentences[i : i + sentence_count]
        indices = list(range(i, min(i + sentence_count, total)))
        text = " ".join(chunk).strip()
        passages.append(
            Passage(
                passage_id=passage_id,
                text=text,
                sentence_indices=indices,
            )
        )
        passage_id += 1

    return passages


class PassageRetrievalIndex:
    """In-memory TF-IDF index for querying document passages."""

    def __init__(
        self,
        sentences: Sequence[str],
        passage_size: int = DEFAULT_PASSAGE_SENTENCE_COUNT,
    ):
        self.sentences = list(sentences)
        self.passage_size = passage_size
        self.passages = create_passages(self.sentences, sentence_count=passage_size)
        self.vectorizer: TfidfVectorizer | None = None
        self.passage_matrix = None

        if self.passages:
            self._build_index()

    def _build_index(self) -> None:
        """Fits TfidfVectorizer across passage texts."""
        texts = [p.text for p in self.passages]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        try:
            self.passage_matrix = self.vectorizer.fit_transform(texts)
        except ValueError:
            # All passages empty or consist only of stop words
            self.passage_matrix = None

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Searches indexed passages using TF-IDF and cosine similarity.

        Args:
            query: User search query string.
            top_k: Number of ranked passages to return (clamped to [1, 10]).

        Returns:
            List of RetrievalResult objects ranked in descending score order.
        """
        if not query or not query.strip():
            return []

        if not self.passages or self.vectorizer is None or self.passage_matrix is None:
            return []

        # Clamp top_k to [1, 10]
        clamped_k = max(MIN_TOP_K, min(MAX_TOP_K, int(top_k)))

        try:
            query_vec = self.vectorizer.transform([query.strip()])
        except Exception:
            return []

        # Check for empty query vector (e.g. out-of-vocabulary or stop words only)
        if query_vec.nnz == 0:
            return []

        # Compute cosine similarity
        sims = cosine_similarity(query_vec, self.passage_matrix).flatten()

        # Filter strictly positive matches and sort descendingly
        scored_passages: list[tuple[int, float]] = []
        for p_idx, score in enumerate(sims):
            val = float(score)
            if val > 0.0:
                scored_passages.append((p_idx, val))

        # Tie-breaker: higher score first, earlier passage_id on tie
        scored_passages.sort(key=lambda item: (-item[1], item[0]))

        results: list[RetrievalResult] = []
        for p_idx, score in scored_passages[:clamped_k]:
            passage = self.passages[p_idx]
            results.append(
                RetrievalResult(
                    passage_id=passage.passage_id,
                    text=passage.text,
                    score=round(score, 4),
                    sentence_indices=passage.sentence_indices,
                )
            )

        return results
