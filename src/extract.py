"""ResearchPaper Lens — PDF and TXT Ingestion & Extraction Module.

Adheres strictly to the architectural contracts defined in architecture.md:
- 10 MiB upload cap
- 100 maximum PDF page cap
- < 100 characters/page low-text warning (heuristic for scanned PDFs)
- pypdf for local, permissive extraction
- Canonical text preservation
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Union

import pypdf
from pypdf.errors import PdfReadError

# Architectural defaults
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MiB
MAX_PDF_PAGES: int = 100
LOW_TEXT_CHARS_PER_PAGE_THRESHOLD: int = 100


class ExtractionError(Exception):
    """Base exception for document extraction failures."""


class FileTooLargeError(ExtractionError):
    """Raised when document size exceeds the 10 MiB limit."""


class PageLimitExceededError(ExtractionError):
    """Raised when PDF exceeds the 100-page limit."""


class UnsupportedFormatError(ExtractionError):
    """Raised when file extension/format is not supported (only PDF and TXT)."""


class EmptyDocumentError(ExtractionError):
    """Raised when an uploaded document contains no readable text."""


@dataclass
class ExtractionResult:
    """Structured result returned from extraction."""

    text: str
    page_count: int
    char_count: int
    warnings: list[str] = field(default_factory=list)
    filename: str = ""


def _resolve_source(
    source: Union[str, Path, bytes, BinaryIO], filename: str | None = None
) -> tuple[bytes, str]:
    """Resolves input source into raw bytes and a resolved filename."""
    resolved_name = filename or ""

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ExtractionError(f"File not found: {path}")
        if not resolved_name:
            resolved_name = path.name
        file_size = os.path.getsize(path)
        if file_size > MAX_FILE_SIZE_BYTES:
            raise FileTooLargeError(
                f"File size ({file_size} bytes) exceeds limit of {MAX_FILE_SIZE_BYTES} bytes (10 MiB)."
            )
        data = path.read_bytes()
        return data, resolved_name

    if isinstance(source, bytes):
        if len(source) > MAX_FILE_SIZE_BYTES:
            raise FileTooLargeError(
                f"File size ({len(source)} bytes) exceeds limit of {MAX_FILE_SIZE_BYTES} bytes (10 MiB)."
            )
        return source, resolved_name

    # File-like object (e.g. io.BytesIO or Streamlit UploadedFile)
    if hasattr(source, "name") and not resolved_name:
        resolved_name = getattr(source, "name", "")

    # Check size attribute if present
    if hasattr(source, "size"):
        size_val = getattr(source, "size")
        if isinstance(size_val, int) and size_val > MAX_FILE_SIZE_BYTES:
            raise FileTooLargeError(
                f"File size ({size_val} bytes) exceeds limit of {MAX_FILE_SIZE_BYTES} bytes (10 MiB)."
            )

    pos = source.tell() if hasattr(source, "tell") else 0
    if hasattr(source, "seek"):
        source.seek(0, io.SEEK_END)
        size_val = source.tell()
        source.seek(pos)
        if size_val > MAX_FILE_SIZE_BYTES:
            raise FileTooLargeError(
                f"File size ({size_val} bytes) exceeds limit of {MAX_FILE_SIZE_BYTES} bytes (10 MiB)."
            )

    data = source.read()
    if hasattr(source, "seek"):
        source.seek(pos)
    return data, resolved_name


def extract_text(
    source: Union[str, Path, bytes, BinaryIO], filename: str | None = None
) -> ExtractionResult:
    """Extracts canonical text and metadata from a PDF or TXT document.

    Args:
        source: File path, raw bytes, or binary stream.
        filename: Optional filename hint (especially if source is bytes/stream).

    Returns:
        ExtractionResult containing text, page_count, char_count, and warnings.

    Raises:
        FileTooLargeError: If file exceeds 10 MiB.
        UnsupportedFormatError: If format is not PDF or TXT.
        PageLimitExceededError: If PDF exceeds 100 pages.
        EmptyDocumentError: If document contains no readable text.
        ExtractionError: On corrupt data or extraction failure.
    """
    data, resolved_name = _resolve_source(source, filename)

    lower_name = resolved_name.lower()
    if lower_name.endswith(".txt"):
        return _extract_txt(data, resolved_name)
    elif lower_name.endswith(".pdf") or data.startswith(b"%PDF"):
        return _extract_pdf(data, resolved_name)
    else:
        raise UnsupportedFormatError(
            f"Unsupported file format for '{resolved_name}'. Only .pdf and .txt files are supported."
        )


def _extract_txt(data: bytes, filename: str) -> ExtractionResult:
    """Decodes plain text with UTF-8 and Latin-1 fallback."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("latin-1")
        except Exception as e:
            raise ExtractionError(f"Failed to decode text file: {e}") from e

    if not text.strip():
        raise EmptyDocumentError("The text file is empty or contains only whitespace.")

    return ExtractionResult(
        text=text,
        page_count=1,
        char_count=len(text),
        warnings=[],
        filename=filename,
    )


def _extract_pdf(data: bytes, filename: str) -> ExtractionResult:
    """Extracts text from PDF page by page using pypdf."""
    stream = io.BytesIO(data)
    try:
        reader = pypdf.PdfReader(stream)
        num_pages = len(reader.pages)
    except (PdfReadError, Exception) as e:
        raise ExtractionError(f"Corrupt or invalid PDF file: {e}") from e

    if num_pages == 0:
        raise EmptyDocumentError("The PDF contains 0 pages.")

    if num_pages > MAX_PDF_PAGES:
        raise PageLimitExceededError(
            f"PDF exceeds the maximum limit of {MAX_PDF_PAGES} pages (document contains {num_pages} pages)."
        )

    warnings: list[str] = []
    page_texts: list[str] = []

    for page_idx, page in enumerate(reader.pages):
        try:
            extracted = page.extract_text() or ""
            page_texts.append(extracted)
        except Exception as e:
            warnings.append(f"Failed to extract text from page {page_idx + 1}: {e}")

    full_text = "\n\n".join(page_texts)
    char_count = len(full_text.strip())

    avg_chars_per_page = char_count / num_pages if num_pages > 0 else 0
    if avg_chars_per_page < LOW_TEXT_CHARS_PER_PAGE_THRESHOLD:
        warnings.append(
            f"Low-text document: Average of {avg_chars_per_page:.1f} characters per page "
            f"(threshold: {LOW_TEXT_CHARS_PER_PAGE_THRESHOLD}). This document may be scanned or image-based; "
            "OCR is not performed in this MVP."
        )

    return ExtractionResult(
        text=full_text,
        page_count=num_pages,
        char_count=len(full_text),
        warnings=warnings,
        filename=filename,
    )
