"""Unit tests for src/extract.py (TDD: RED -> GREEN -> REFACTOR)."""

import io
import pytest
from pypdf import PdfWriter

from src.extract import (
    MAX_FILE_SIZE_BYTES,
    MAX_PDF_PAGES,
    LOW_TEXT_CHARS_PER_PAGE_THRESHOLD,
    ExtractionResult,
    FileTooLargeError,
    PageLimitExceededError,
    UnsupportedFormatError,
    EmptyDocumentError,
    ExtractionError,
    extract_text,
)


def _create_dummy_pdf(num_pages: int = 1, text_per_page: str = "This is a test research paper page.") -> bytes:
    """Helper to generate an in-memory PDF with pypdf."""
    writer = PdfWriter()
    for _ in range(num_pages):
        # Add a blank page with annotation or simple text page
        page = writer.add_blank_page(width=612, height=792)
        # In pypdf, we can use compress or write empty pages; for text extraction test
        # we can use writer with custom metadata or dummy stream
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_extract_txt_valid(tmp_path):
    txt_file = tmp_path / "paper.txt"
    content = "Information retrieval is the science of searching for documents and metadata."
    txt_file.write_text(content, encoding="utf-8")

    result = extract_text(txt_file)
    assert isinstance(result, ExtractionResult)
    assert result.text == content
    assert result.page_count == 1
    assert result.char_count == len(content)
    assert len(result.warnings) == 0


def test_extract_txt_latin1_fallback(tmp_path):
    txt_file = tmp_path / "paper_latin1.txt"
    content_bytes = "Research by Müller & Schrödinger on IR.".encode("latin-1")
    txt_file.write_bytes(content_bytes)

    result = extract_text(txt_file)
    assert "Müller & Schrödinger" in result.text
    assert result.page_count == 1


def test_extract_txt_empty(tmp_path):
    txt_file = tmp_path / "empty.txt"
    txt_file.write_text("   \n\t  ", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        extract_text(txt_file)


def test_extract_file_too_large(tmp_path):
    txt_file = tmp_path / "huge.txt"
    # Create file slightly above 10 MiB
    oversized = MAX_FILE_SIZE_BYTES + 1
    stream = io.BytesIO(b"x" * oversized)

    with pytest.raises(FileTooLargeError):
        extract_text(stream, filename="huge.txt")


def test_extract_unsupported_format(tmp_path):
    doc_file = tmp_path / "paper.docx"
    doc_file.write_bytes(b"dummy")

    with pytest.raises(UnsupportedFormatError):
        extract_text(doc_file)


def test_extract_corrupt_pdf():
    corrupt_stream = io.BytesIO(b"%PDF-1.4 corrupt data not a real pdf")

    with pytest.raises(ExtractionError):
        extract_text(corrupt_stream, filename="corrupt.pdf")


def test_extract_pdf_page_limit():
    writer = PdfWriter()
    for _ in range(MAX_PDF_PAGES + 1):
        writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)

    with pytest.raises(PageLimitExceededError):
        extract_text(buf.getvalue(), filename="too_long.pdf")


def test_extract_pdf_low_text_warning():
    # A PDF with 1 page and empty/near-empty text (< 100 chars)
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)

    result = extract_text(buf.getvalue(), filename="scanned.pdf")
    assert any("low-text" in w.lower() or "scanned" in w.lower() for w in result.warnings)


def test_extract_nonexistent_file():
    with pytest.raises(ExtractionError, match="File not found"):
        extract_text("non_existent_file.pdf")


def test_extract_pdf_magic_bytes_without_filename():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    writer.write(buf)

    result = extract_text(buf.getvalue())
    assert result.page_count == 1
    assert any("low-text" in w.lower() for w in result.warnings)

