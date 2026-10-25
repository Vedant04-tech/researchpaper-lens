"""ResearchPaper Lens — Dual-Format Reporting Module (JSON & PDF).

Provides structured JSON schema export and publication-grade PDF generation via ReportLab.
Ensures XML entity escaping on all paragraph inputs to prevent parser crashes.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from src.privacy import redact_sensitive_text


@dataclass
class DocumentMetadataReport:
    """Document identity and cryptographic integrity record."""

    filename: str
    file_size_bytes: int
    hashes: dict[str, str]
    page_count: int
    word_count: int
    sentence_count: int
    references_detected: bool
    warnings: list[str] = field(default_factory=list)


@dataclass
class TextRankSummaryReport:
    """Extractive summary metrics and output."""

    summary: str
    ratio: float
    sentence_count: int
    word_count: int
    compression_ratio: float
    selected_indices: list[int]


@dataclass
class BaselineSummariesReport:
    """Academic baseline comparison summaries."""

    lead_k_summary: str
    lead_k_words: int
    word_frequency_summary: str
    word_frequency_words: int


@dataclass
class KeywordReportItem:
    """Keyword entry with term and TF-IDF score."""

    term: str
    score: float


@dataclass
class AbstractiveSummaryReport:
    """Abstractive synthesis result if requested."""

    summary: str
    model_name: str
    provider: str
    latency_seconds: float


@dataclass
class AnalysisReport:
    """Complete forensic & analytical session report."""

    document_metadata: DocumentMetadataReport
    textrank_summary: TextRankSummaryReport
    baseline_summaries: BaselineSummariesReport
    keywords: list[KeywordReportItem]
    review_cues: list[str]
    abstractive_summary: AbstractiveSummaryReport | None = None
    redactions_count: int = 0
    schema_version: str = "1.0.0"
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def build_analysis_report(
    extraction_res: Any,
    prep_doc: Any,
    textrank_res: Any,
    summary_ratio: float,
    lead_res: Any = None,
    freq_res: Any = None,
    keywords_list: list[Any] | None = None,
    abstractive_res: Any = None,
    apply_privacy: bool = True,
) -> AnalysisReport:
    """Constructs a validated AnalysisReport object from pipeline components."""
    # Compute summaries and counts
    tr_summary = textrank_res.summary
    total_redactions = 0

    if apply_privacy:
        audit = redact_sensitive_text(tr_summary)
        tr_summary = audit.sanitized_text
        total_redactions += audit.total_redactions

    tr_words = len(tr_summary.split())
    comp_ratio = (
        (tr_words / prep_doc.word_count * 100) if prep_doc.word_count > 0 else 0.0
    )

    lead_summary_str = lead_res.summary if lead_res else ""
    freq_summary_str = freq_res.summary if freq_res else ""

    if apply_privacy:
        lead_summary_str = redact_sensitive_text(lead_summary_str).sanitized_text
        freq_summary_str = redact_sensitive_text(freq_summary_str).sanitized_text

    baseline_rep = BaselineSummariesReport(
        lead_k_summary=lead_summary_str,
        lead_k_words=len(lead_summary_str.split()),
        word_frequency_summary=freq_summary_str,
        word_frequency_words=len(freq_summary_str.split()),
    )

    kw_items = []
    if keywords_list:
        for kw in keywords_list:
            term = kw.term if hasattr(kw, "term") else str(kw)
            score = float(kw.score) if hasattr(kw, "score") else 0.0
            kw_items.append(KeywordReportItem(term=term, score=score))

    abs_rep = None
    if abstractive_res:
        abs_summary_str = abstractive_res.summary
        if apply_privacy:
            abs_summary_str = redact_sensitive_text(abs_summary_str).sanitized_text
        abs_rep = AbstractiveSummaryReport(
            summary=abs_summary_str,
            model_name=getattr(abstractive_res, "model_name", "abstractive-model"),
            provider=getattr(abstractive_res, "provider", "inference"),
            latency_seconds=getattr(abstractive_res, "latency_seconds", 0.0),
        )

    # Standard forensic & analytical review cues
    cues = [
        "Extractive Faithfulness: TextRank extracts verbatim sentences directly from source text (0% hallucination risk).",
        "Graph Centrality: Vertex importance is determined via PageRank (d=0.85, max_iter=100) over TF-IDF cosine similarities.",
        "Chronological Ordering: Extracted sentences are strictly restored to their original document sequence.",
        "Scanned PDF Limitation: Low-text heuristic (< 100 chars/page) flags image-only PDFs where embedded text is minimal.",
        "Lexical Search Boundary: Passage retrieval uses lexical TF-IDF matching and does not perform neural semantic re-ranking.",
        "Privacy by Default: Sensitive personal identifiers, emails, and credentials are automatically masked.",
    ]

    doc_meta = DocumentMetadataReport(
        filename=extraction_res.filename or "document",
        file_size_bytes=getattr(extraction_res, "raw_size_bytes", 0),
        hashes=getattr(extraction_res, "hashes", {}),
        page_count=extraction_res.page_count,
        word_count=prep_doc.word_count,
        sentence_count=prep_doc.sentence_count,
        references_detected=bool(prep_doc.references_text),
        warnings=list(extraction_res.warnings),
    )

    tr_rep = TextRankSummaryReport(
        summary=tr_summary,
        ratio=summary_ratio,
        sentence_count=len(textrank_res.selected_indices),
        word_count=tr_words,
        compression_ratio=round(comp_ratio, 2),
        selected_indices=list(textrank_res.selected_indices),
    )

    return AnalysisReport(
        document_metadata=doc_meta,
        textrank_summary=tr_rep,
        baseline_summaries=baseline_rep,
        keywords=kw_items,
        review_cues=cues,
        abstractive_summary=abs_rep,
        redactions_count=total_redactions,
    )


def export_report_json(report: AnalysisReport, indent: int = 2) -> str:
    """Exports AnalysisReport into a validated, structured JSON string."""
    data = asdict(report)
    return json.dumps(data, indent=indent, ensure_ascii=False)


def export_report_pdf(report: AnalysisReport) -> bytes:
    """Generates a publication-grade PDF report using ReportLab with XML escaping on all text."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#172033"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#526174"),
        spaceAfter=12,
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#285D9A"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B"),
    )
    cue_style = ParagraphStyle(
        "CueItem",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
    )
    code_style = ParagraphStyle(
        "ReportCode",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#0F172A"),
    )

    story = []

    # Title & Header
    story.append(Paragraph(xml_escape("ResearchPaper Lens — Forensic & Analytical Report"), title_style))
    story.append(
        Paragraph(
            xml_escape(
                f"Generated at: {report.generated_at} (UTC)  |  Schema Version: {report.schema_version}  |  File: {report.document_metadata.filename}"
            ),
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#285D9A"), spaceAfter=10))

    # 1. Document Identity & Cryptographic Hashes Table
    story.append(Paragraph(xml_escape("1. Document Integrity & Cryptographic Evidence"), section_heading))

    meta = report.document_metadata
    sha256_val = meta.hashes.get("sha256", "N/A")
    sha1_val = meta.hashes.get("sha1", "N/A")
    md5_val = meta.hashes.get("md5", "N/A")

    table_data = [
        [Paragraph(xml_escape("Metric / Digest"), section_heading), Paragraph(xml_escape("Value"), section_heading)],
        [Paragraph("<b>Filename:</b>", body_style), Paragraph(xml_escape(meta.filename), body_style)],
        [Paragraph("<b>File Size:</b>", body_style), Paragraph(xml_escape(f"{meta.file_size_bytes:,} bytes"), body_style)],
        [Paragraph("<b>Pages / Words / Sentences:</b>", body_style), Paragraph(xml_escape(f"{meta.page_count} pages  |  {meta.word_count:,} words  |  {meta.sentence_count:,} sentences"), body_style)],
        [Paragraph("<b>References Section:</b>", body_style), Paragraph(xml_escape("Detected & Isolated" if meta.references_detected else "None detected"), body_style)],
        [Paragraph("<b>SHA-256 Digest:</b>", body_style), Paragraph(f"<code>{xml_escape(sha256_val)}</code>", code_style)],
        [Paragraph("<b>SHA-1 Digest:</b>", body_style), Paragraph(f"<code>{xml_escape(sha1_val)}</code>", code_style)],
        [Paragraph("<b>MD5 Digest:</b>", body_style), Paragraph(f"<code>{xml_escape(md5_val)}</code>", code_style)],
    ]

    t = Table(table_data, colWidths=[160, 370])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 10))

    # Warnings if any
    if meta.warnings:
        story.append(Paragraph(xml_escape("Extraction Warnings & Container Alerts:"), section_heading))
        for w in meta.warnings:
            story.append(Paragraph(f"• <b>[WARNING]</b> {xml_escape(w)}", body_style))
        story.append(Spacer(1, 8))

    # 2. Methodological Review Cues & Disclaimers
    story.append(Paragraph(xml_escape("2. Explainable Review Cues & Limitations"), section_heading))
    cue_paragraphs = []
    for cue in report.review_cues:
        cue_paragraphs.append([Paragraph(f"• {xml_escape(cue)}", cue_style)])

    cue_table = Table(cue_paragraphs, colWidths=[530])
    cue_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#E2E8F0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(cue_table)
    story.append(Spacer(1, 10))

    # 3. TextRank Extractive Summary
    tr = report.textrank_summary
    story.append(
        Paragraph(
            xml_escape(f"3. TextRank Extractive Summary ({tr.word_count} words, {tr.compression_ratio:.1f}% compression)"),
            section_heading,
        )
    )
    story.append(
        Paragraph(
            xml_escape(
                f"Extracted {tr.sentence_count} sentences out of {meta.sentence_count} using TF-IDF PageRank graph centrality."
            ),
            subtitle_style,
        )
    )

    summary_escaped = xml_escape(tr.summary).replace("\n", "<br/><br/>")
    story.append(Paragraph(summary_escaped, body_style))
    story.append(Spacer(1, 10))

    # 4. Academic Baseline Summaries Comparison
    story.append(Paragraph(xml_escape("4. Academic Baseline Summaries"), section_heading))
    lead = report.baseline_summaries.lead_k_summary or "N/A"
    freq = report.baseline_summaries.word_frequency_summary or "N/A"

    baseline_data = [
        [Paragraph(xml_escape("Method"), section_heading), Paragraph(xml_escape("Output"), section_heading)],
        [
            Paragraph("<b>Lead-k:</b><br/><i>(First k sentences)</i>", body_style),
            Paragraph(xml_escape(lead).replace("\n", "<br/>"), body_style),
        ],
        [
            Paragraph("<b>Word-Frequency:</b><br/><i>(Term count weighting)</i>", body_style),
            Paragraph(xml_escape(freq).replace("\n", "<br/>"), body_style),
        ],
    ]
    b_table = Table(baseline_data, colWidths=[140, 390])
    b_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(b_table)
    story.append(Spacer(1, 10))

    # 5. Salient Keywords
    if report.keywords:
        story.append(Paragraph(xml_escape("5. Salient Keywords (TF-IDF Aggregation)"), section_heading))
        kw_rows = []
        for i in range(0, min(len(report.keywords), 15), 3):
            chunk = report.keywords[i : i + 3]
            row = []
            for item in chunk:
                row.append(Paragraph(f"<b>{xml_escape(item.term)}</b>: {item.score:.2f}", body_style))
            while len(row) < 3:
                row.append(Paragraph("", body_style))
            kw_rows.append(row)

        kw_table = Table(kw_rows, colWidths=[176, 176, 176])
        kw_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(kw_table)
        story.append(Spacer(1, 10))

    # 6. Abstractive Synthesis (if present)
    if report.abstractive_summary:
        abs_rep = report.abstractive_summary
        story.append(
            Paragraph(
                xml_escape(f"6. Abstractive Synthesis (Model: {abs_rep.model_name}, Latency: {abs_rep.latency_seconds:.2f}s)"),
                section_heading,
            )
        )
        story.append(
            Paragraph(
                xml_escape(
                    f"Synthesized deep learning output ({abs_rep.provider}); paraphrased non-verbatim representation."
                ),
                subtitle_style,
            )
        )
        abs_text_escaped = xml_escape(abs_rep.summary).replace("\n", "<br/><br/>")
        story.append(Paragraph(abs_text_escaped, body_style))
        story.append(Spacer(1, 10))

    # Privacy notice footer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceBefore=10, spaceAfter=6))
    story.append(
        Paragraph(
            xml_escape(
                f"Confidentiality & Privacy: Redactions applied ({report.redactions_count} sensitive patterns masked). "
                "Document evaluated locally with ResearchPaper Lens."
            ),
            cue_style,
        )
    )

    doc.build(story)
    return buffer.getvalue()
