"""ResearchPaper Lens — Text Preprocessing & Sentence Splitting Module.

Adheres strictly to the architectural contracts defined in architecture.md:
- Preserves canonical source text separately from normalized text
- Deterministic regex-based English sentence splitter with protected abbreviations
- No runtime downloads (NLTK/spaCy-free)
- Clean whitespace normalization and line-break un-hyphenation
- Automatic References/Bibliography detection and extraction
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Common abbreviations in academic / technical papers
PROTECTED_ABBREVIATIONS: list[str] = [
    # Titles
    r"\bDr\.",
    r"\bMr\.",
    r"\bMrs\.",
    r"\bMs\.",
    r"\bProf\.",
    r"\bSr\.",
    r"\bJr\.",
    # Academic and technical terms
    r"\be\.g\.",
    r"\bi\.e\.",
    r"\bet al\.",
    r"\bal\.",
    r"\bFig\.",
    r"\bFigs\.",
    r"\bEq\.",
    r"\bEqs\.",
    r"\bRef\.",
    r"\bRefs\.",
    r"\bVol\.",
    r"\bNo\.",
    r"\bpp\.",
    r"\bp\.",
    r"\bvs\.",
    r"\bcf\.",
    r"\bapprox\.",
    r"\bsec\.",
    r"\bch\.",
    r"\bdept\.",
    r"\buniv\.",
]

_PLACEHOLDER_DOT = "\ue001"


@dataclass
class PreprocessedDocument:
    """Preprocessed document data structure."""

    canonical_text: str
    cleaned_text: str
    sentences: list[str] = field(default_factory=list)
    references_text: str | None = None
    word_count: int = 0
    sentence_count: int = 0


def clean_text(raw_text: str) -> str:
    """Normalizes whitespace and repairs line-break hyphenation while preserving readability."""
    if not raw_text:
        return ""

    # Normalize line breaks
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Repair hyphenated line wraps (e.g., "infor-\nmation" -> "information")
    text = re.sub(r"(\b[A-Za-z]+)-\n([A-Za-z]+\b)", r"\1\2", text)

    # Convert tabs to single spaces
    text = text.replace("\t", " ")

    # Collapse multiple horizontal whitespace characters to a single space
    text = re.sub(r"[^\S\n]+", " ", text)

    # Collapse lines with only whitespace
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Collapse 3 or more consecutive newlines into 2 newlines (paragraph boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_references(text: str) -> tuple[str, str | None]:
    """Detects and separates a trailing References or Bibliography section."""
    if not text:
        return "", None

    # Matches standalone header like "\nReferences\n" or "\nBibliography\n"
    match = re.search(
        r"\n+(?:(?:\d+\.?\s*)?(?:references|bibliography))\s*\n+",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        body = text[: match.start()].strip()
        references = text[match.end() :].strip()
        return body, references if references else None

    return text, None


def split_sentences(text: str) -> list[str]:
    """Splits English text into clean sentences using deterministic regex with protected abbreviations."""
    if not text or not text.strip():
        return []

    # Clean text representation for sentence parsing
    working = clean_text(text)
    if not working:
        return []

    # Protect decimals (e.g. 3.14 -> 3<DOT>14)
    working = re.sub(r"(\d+)\.(\d+)", rf"\1{_PLACEHOLDER_DOT}\2", working)

    # Protect common academic abbreviations
    for abbr_pattern in PROTECTED_ABBREVIATIONS:
        matches = re.finditer(abbr_pattern, working, flags=re.IGNORECASE)
        for m in reversed(list(matches)):
            original = m.group(0)
            replacement = original.replace(".", _PLACEHOLDER_DOT)
            working = working[: m.start()] + replacement + working[m.end() :]

    # Split on sentence boundaries: punctuation (. ! ?) followed by whitespace and an uppercase letter/quote/number
    # Also handle end of paragraph newlines
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"\'\(\[])|\n\n+", working)

    sentences: list[str] = []
    for s in raw_sentences:
        # Restore protected dots
        restored = s.replace(_PLACEHOLDER_DOT, ".").strip()
        # Remove stray internal newlines inside a single sentence
        restored = re.sub(r"\s*\n\s*", " ", restored)
        if restored:
            sentences.append(restored)

    return sentences


def preprocess_document(raw_text: str) -> PreprocessedDocument:
    """End-to-end preprocessing pipeline for an ingested document.

    Preserves the raw canonical text, produces cleaned content body,
    separates References/Bibliography if present, and segments into sentences.
    """
    cleaned = clean_text(raw_text)
    body, references = split_references(cleaned)
    sentences = split_sentences(body)
    word_count = len(body.split())

    return PreprocessedDocument(
        canonical_text=raw_text,
        cleaned_text=body,
        sentences=sentences,
        references_text=references,
        word_count=word_count,
        sentence_count=len(sentences),
    )
