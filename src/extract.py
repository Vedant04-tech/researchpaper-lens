"""ResearchPaper Lens — PDF and TXT Ingestion & Extraction Module.

Adheres strictly to the architectural contracts defined in architecture.md:
- 10 MiB upload cap
- 100 maximum PDF page cap
- < 100 characters/page low-text warning (heuristic for scanned PDFs)
- pypdf for local, permissive extraction
- Canonical text preservation
"""

from __future__ import annotations

import hashlib
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Union

import pypdf
from pypdf.errors import PdfReadError

# Architectural defaults
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MiB
MAX_PDF_PAGES: int = 100
LOW_TEXT_CHARS_PER_PAGE_THRESHOLD: int = 100
MAX_CHARS_PER_PAGE: int = 100_000
MAX_TOTAL_CHARS: int = 5_000_000


class ExtractionError(Exception):
    """Base exception for document extraction failures."""


class FileTooLargeError(ExtractionError):
    """Raised when document size exceeds the 10 MiB limit."""


class PageLimitExceededError(ExtractionError):
    """Raised when PDF exceeds the 100-page limit."""


class UnsupportedFormatError(ExtractionError):
    """Raised when file extension/format is not supported (only PDF and TXT)."""


class EmptyDocumentError(ExtractionError):
    """Raised when an uploaded document contains no readable text or is 0 bytes."""


class CorruptDocumentError(ExtractionError):
    """Raised when input bitstream is malformed or unreadable."""


class DecompressionBombError(ExtractionError):
    """Raised when extracted document text exceeds safety limits (possible decompression bomb)."""


def sanitize_filename(filename: str | None) -> str:
    """Sanitizes user-supplied filenames for display and file system safety.

    - Strips directory traversal patterns (e.g. ../, ..\\, /, \\)
    - Strips control characters (ASCII < 32 and 127)
    - Truncates to a maximum length of 255 characters while preserving extension
    - Defaults to 'unnamed_document' if empty or purely invalid
    """
    if not filename:
        return "unnamed_document"

    # Remove directory paths / traversal components
    clean = re.sub(r"[/\\]+", "_", filename)
    clean = clean.replace("..", "_")

    # Strip control characters and non-printable characters
    clean = "".join(ch for ch in clean if ord(ch) >= 32 and ord(ch) != 127)
    clean = clean.strip(". ")

    if not clean:
        return "unnamed_document"

    # Truncate to 255 characters max, preserving file extension if present
    if len(clean) > 255:
        p = Path(clean)
        ext = p.suffix[:10]  # keep reasonable extension
        stem = p.stem[: 255 - len(ext)]
        clean = stem + ext

    return clean or "unnamed_document"


def compute_hashes(data: bytes) -> dict[str, str]:
    """Computes SHA-256, SHA-1, and MD5 cryptographic digests on raw source bytes."""
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "sha1": hashlib.sha1(data).hexdigest(),
        "md5": hashlib.md5(data).hexdigest(),
    }


@dataclass
class ExtractionResult:
    """Structured result returned from extraction."""

    text: str
    page_count: int
    char_count: int
    warnings: list[str] = field(default_factory=list)
    filename: str = ""
    hashes: dict[str, str] = field(default_factory=dict)
    raw_size_bytes: int = 0


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

    if len(data) == 0:
        raise EmptyDocumentError("The document is empty (0 bytes).")

    return data, sanitize_filename(resolved_name)


def extract_text(
    source: Union[str, Path, bytes, BinaryIO], filename: str | None = None
) -> ExtractionResult:
    """Extracts canonical text and metadata from a PDF or TXT document.

    Args:
        source: File path, raw bytes, or binary stream.
        filename: Optional filename hint (especially if source is bytes/stream).

    Returns:
        ExtractionResult containing text, page_count, char_count, hashes, and warnings.

    Raises:
        FileTooLargeError: If file exceeds 10 MiB.
        UnsupportedFormatError: If format is not PDF or TXT.
        PageLimitExceededError: If PDF exceeds 100 pages.
        EmptyDocumentError: If document contains no readable text or is 0 bytes.
        DecompressionBombError: If extracted text exceeds safe limits.
        CorruptDocumentError: On corrupt bitstreams.
        ExtractionError: On unexpected extraction failure.
    """
    data, resolved_name = _resolve_source(source, filename)

    if len(data) == 0:
        raise EmptyDocumentError("The document is empty (0 bytes).")

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

    if len(text) > MAX_TOTAL_CHARS:
        raise DecompressionBombError(
            f"Text file content ({len(text)} characters) exceeds maximum safety limit of {MAX_TOTAL_CHARS}."
        )

    return ExtractionResult(
        text=text,
        page_count=1,
        char_count=len(text),
        warnings=[],
        filename=filename,
        hashes=compute_hashes(data),
        raw_size_bytes=len(data),
    )


def _extract_pdf(data: bytes, filename: str) -> ExtractionResult:
    """Extracts text from PDF page by page using pypdf."""
    stream = io.BytesIO(data)
    try:
        reader = pypdf.PdfReader(stream)
        num_pages = len(reader.pages)
    except (PdfReadError, Exception) as e:
        raise CorruptDocumentError(f"Corrupt or invalid PDF file: {e}") from e

    if num_pages == 0:
        raise EmptyDocumentError("The PDF contains 0 pages.")

    if num_pages > MAX_PDF_PAGES:
        raise PageLimitExceededError(
            f"PDF exceeds the maximum limit of {MAX_PDF_PAGES} pages (document contains {num_pages} pages)."
        )

    warnings: list[str] = []
    page_texts: list[str] = []
    running_chars = 0

    for page_idx, page in enumerate(reader.pages):
        try:
            extracted = page.extract_text() or ""
            if len(extracted) > MAX_CHARS_PER_PAGE:
                raise DecompressionBombError(
                    f"Page {page_idx + 1} extracted text ({len(extracted)} chars) exceeds limit of {MAX_CHARS_PER_PAGE}."
                )
            running_chars += len(extracted)
            if running_chars > MAX_TOTAL_CHARS:
                raise DecompressionBombError(
                    f"Total extracted text exceeds maximum safety limit of {MAX_TOTAL_CHARS} characters."
                )
            page_texts.append(extracted)
        except DecompressionBombError:
            raise
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
        hashes=compute_hashes(data),
        raw_size_bytes=len(data),
    )
