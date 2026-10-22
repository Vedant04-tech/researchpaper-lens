"""ResearchPaper Lens — Keyword Extraction Module.

Extracts salient keywords using sentence-level TF-IDF term weight aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_KEYWORD_COUNT: int = 10


@dataclass
class Keyword:
    """Represents an extracted keyword term with its aggregated TF-IDF weight."""

    term: str
    score: float


def extract_keywords(
    sentences: Sequence[str], count: int = DEFAULT_KEYWORD_COUNT
) -> list[Keyword]:
    """Extracts top keywords by fitting TF-IDF across sentences and aggregating weights.

    Args:
        sentences: Individual sentences from the document acting as the corpus.
        count: Maximum number of keywords to return (default 10).

    Returns:
        List of Keyword objects sorted descendingly by aggregate importance.
    """
    valid_sentences = [s.strip() for s in sentences if s and s.strip()]
    if not valid_sentences or count <= 0:
        return []

    # Fit TF-IDF across sentences, requiring alphabetic tokens with at least 3 chars
    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",
    )

    try:
        tfidf_matrix = vectorizer.fit_transform(valid_sentences)
    except ValueError:
        # Happens when sentences have no eligible tokens
        return []

    feature_names = vectorizer.get_feature_names_out()
    if len(feature_names) == 0:
        return []

    # Aggregate term weights across all sentences (sum)
    aggregated_weights = np.asarray(tfidf_matrix.sum(axis=0)).flatten()

    # Sort descending by score
    sorted_indices = np.argsort(-aggregated_weights)

    keywords: list[Keyword] = []
    for idx in sorted_indices[:count]:
        score = float(aggregated_weights[idx])
        if score > 0.0:
            keywords.append(
                Keyword(
                    term=str(feature_names[idx]),
                    score=round(score, 4),
                )
            )

    return keywords
