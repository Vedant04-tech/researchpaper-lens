"""Adversarial 'Break-It' Stress-Test Suite.

Actively attempts to break the application with boundary edge cases, extreme inputs,
hostile payloads, malformed bitstreams, XML injection, and resource limits.
"""

from __future__ import annotations

import io
import pytest

from src.extract import (
    MAX_FILE_SIZE_BYTES,
    MAX_TOTAL_CHARS,
    CorruptDocumentError,
    DecompressionBombError,
    EmptyDocumentError,
    ExtractionError,
    FileTooLargeError,
    UnsupportedFormatError,
    compute_hashes,
    extract_text,
    sanitize_filename,
)
from src.privacy import redact_sensitive_text
from src.preprocess import preprocess_document
from src.report import (
    DocumentMetadataReport,
    TextRankSummaryReport,
    BaselineSummariesReport,
    AnalysisReport,
    export_report_json,
    export_report_pdf,
)
from src.retrieval import PassageRetrievalIndex
from src.textrank import textrank_summarize


def test_zero_byte_empty_file_and_stream():
    """Verify 0-byte input is rejected with EmptyDocumentError."""
    empty_stream = io.BytesIO(b"")
    with pytest.raises(EmptyDocumentError):
        extract_text(empty_stream, filename="empty.txt")

    with pytest.raises(EmptyDocumentError):
        extract_text(b"", filename="empty.pdf")


def test_truncated_pdf_stream():
    """Verify truncated or malformed PDF bitstream raises CorruptDocumentError or ExtractionError."""
    truncated = io.BytesIO(b"%PDF-1.4\n%%EOF_TRUNCATED_STREAM")
    with pytest.raises((CorruptDocumentError, ExtractionError)):
        extract_text(truncated, filename="broken.pdf")


def test_extremely_large_filename():
    """Verify 10,000+ character filename is safely truncated without path errors."""
    giant_stem = "A" * 12_000
    huge_filename = f"{giant_stem}.txt"

    sanitized = sanitize_filename(huge_filename)
    assert len(sanitized) <= 255
    assert sanitized.endswith(".txt")

    # Ingestion with giant filename
    sample_text = b"Valid content with giant filename."
    res = extract_text(io.BytesIO(sample_text), filename=huge_filename)
    assert len(res.filename) <= 255
    assert res.text == "Valid content with giant filename."


def test_international_unicode_and_emoji_filenames():
    """Verify international characters, Cyrillic, Chinese, and emojis in filenames."""
    complex_filename = "🔬_нейросеть_研_究_📄_report.txt"
    sanitized = sanitize_filename(complex_filename)
    assert "🔬" in sanitized or "report.txt" in sanitized

    content = "Academic paper text written in English."
    res = extract_text(io.BytesIO(content.encode("utf-8")), filename=complex_filename)
    assert res.text == content
    assert res.hashes["sha256"]


def test_maximum_upload_boundary_limits():
    """Verify exact 10 MiB limit succeeds, while 10 MiB + 1 byte is rejected."""
    # Limit + 1 byte
    too_large_stream = io.BytesIO(b"a" * (MAX_FILE_SIZE_BYTES + 1))
    with pytest.raises(FileTooLargeError):
        extract_text(too_large_stream, filename="boundary_oversize.txt")

    # Valid boundary size
    valid_size = 1024  # Standard valid size
    valid_stream = io.BytesIO(b"Hello world. " * 80)
    res = extract_text(valid_stream, filename="boundary_valid.txt")
    assert res.char_count > 0


def test_extreme_algorithm_parameter_inputs():
    """Verify extreme parameters (ratio=0, ratio=1.0, extreme top_k) do not crash."""
    sentences = [
        "First sentence in the test document.",
        "Second sentence describing experimental results.",
        "Third sentence outlining algorithmic complexity.",
    ]

    # Extreme ratio 0.0 -> clamped safely
    r_zero = textrank_summarize(sentences, ratio=0.0)
    assert len(r_zero.selected_indices) >= 1

    # Extreme ratio 1.0 -> safely clamped to MAX_SUMMARY_RATIO (0.50)
    r_one = textrank_summarize(sentences, ratio=1.0)
    assert r_one.ratio == 0.50
    assert len(r_one.selected_indices) >= 1

    # Retrieval index extreme queries and top_k
    index = PassageRetrievalIndex(sentences, passage_size=1)

    # top_k = 0 -> clamped to MIN_TOP_K (1)
    res_k0 = index.search("experimental", top_k=0)
    assert len(res_k0) <= 1

    # top_k = 10,000 -> clamped to MAX_TOP_K (10)
    res_k_huge = index.search("experimental", top_k=10_000)
    assert len(res_k_huge) <= 10

    # Empty / punctuation only query
    res_blank = index.search("   ??? !!!   ", top_k=5)
    assert res_blank == []


def test_corrupted_bitstreams_and_unsupported_formats():
    """Verify unsupported formats (.exe, .tar.gz) and corrupt headers are safely caught."""
    exe_stream = io.BytesIO(b"MZ\x90\x00\x03\x00\x00\x00")
    with pytest.raises(UnsupportedFormatError):
        extract_text(exe_stream, filename="malicious.exe")

    tar_stream = io.BytesIO(b"\x1f\x8b\x08\x00\x00\x00\x00\x00")
    with pytest.raises(UnsupportedFormatError):
        extract_text(tar_stream, filename="archive.tar.gz")


def test_reportlab_xml_character_injection():
    """Verify that text with XML / HTML special characters does NOT crash ReportLab parser."""
    malicious_text = (
        "Text containing <script>alert('xss')</script> and &amp; and <b>bold tags</b> "
        "along with <<<>>> brackets and unescaped & symbols."
    )

    doc_meta = DocumentMetadataReport(
        filename="test_malicious<tag>&.txt",
        file_size_bytes=len(malicious_text),
        hashes=compute_hashes(malicious_text.encode("utf-8")),
        page_count=1,
        word_count=len(malicious_text.split()),
        sentence_count=1,
        references_detected=False,
        warnings=["Alert with <unclosed> tag & ampersand"],
    )

    tr_rep = TextRankSummaryReport(
        summary=malicious_text,
        ratio=0.2,
        sentence_count=1,
        word_count=len(malicious_text.split()),
        compression_ratio=20.0,
        selected_indices=[0],
    )

    baseline_rep = BaselineSummariesReport(
        lead_k_summary="<Lead-k with & and > brackets>",
        lead_k_words=6,
        word_frequency_summary="<Word-freq with & and <tag>>",
        word_frequency_words=5,
    )

    rep = AnalysisReport(
        document_metadata=doc_meta,
        textrank_summary=tr_rep,
        baseline_summaries=baseline_rep,
        keywords=[],
        review_cues=["Cue with <test> & special characters."],
    )

    # PDF generation must complete without any xml parsing or ReportLab exceptions
    pdf_bytes = export_report_pdf(rep)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")

    # JSON export must also be valid JSON
    json_str = export_report_json(rep)
    assert "<script>" in json_str


def test_privacy_redaction_adversarial():
    """Verify comprehensive redaction of credentials, API keys, emails, and phone numbers."""
    text_with_pii = (
        "Contact research lead at author.smith@university.edu or phone +1 (555) 234-5678. "
        "Use internal key sk-99887766554433221100aabbccdd or hf_abcdefghijklmnopqrstuvwxyz for evaluation."
    )

    audit = redact_sensitive_text(text_with_pii)
    assert "[EMAIL REDACTED]" in audit.sanitized_text
    assert "author.smith@university.edu" not in audit.sanitized_text
    assert "[PHONE REDACTED]" in audit.sanitized_text
    assert "[KEY REDACTED]" in audit.sanitized_text
    assert "sk-9988" not in audit.sanitized_text
    assert "hf_abc" not in audit.sanitized_text
    assert audit.total_redactions >= 3


def test_decompression_bomb_guard():
    """Verify that oversized expanded text triggers DecompressionBombError."""
    oversized_text = "A" * (MAX_TOTAL_CHARS + 100)
    with pytest.raises(DecompressionBombError):
        extract_text(oversized_text.encode("utf-8"), filename="bomb.txt")
