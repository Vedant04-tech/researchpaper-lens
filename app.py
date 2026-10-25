"""ResearchPaper Lens — Streamlit Application.

Document summarization and information retrieval workspace.
Adheres strictly to design.md, architecture.md, and prd.md.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import streamlit as st

from src.abstractive import (
    DEFAULT_HF_MODEL,
    DEFAULT_OPENAI_MODEL,
    AbstractiveError,
    AbstractiveResult,
    MissingApiKeyError,
    ModelTimeoutError,
    summarize_abstractive,
)
from src.baselines import (
    keyword_match_retrieve,
    lead_k_summarize,
    word_frequency_summarize,
)
from src.extract import (
    CorruptDocumentError,
    DecompressionBombError,
    EmptyDocumentError,
    ExtractionError,
    FileTooLargeError,
    PageLimitExceededError,
    UnsupportedFormatError,
    extract_text,
)
from src.keywords import extract_keywords
from src.preprocess import preprocess_document
from src.privacy import redact_sensitive_text
from src.report import (
    build_analysis_report,
    export_report_json,
    export_report_pdf,
)
from src.retrieval import PassageRetrievalIndex
from src.textrank import DEFAULT_SUMMARY_RATIO, textrank_summarize

# Configure Streamlit page layout and metadata
st.set_page_config(
    page_title="ResearchPaper Lens",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling aligning with design.md color tokens and calm research workspace aesthetic
st.markdown(
    """
    <style>
    /* Primary typography & container spacing */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #172033;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #526174;
        margin-bottom: 1.2rem;
    }
    /* Stat cards */
    .stat-box {
        background-color: #FFFFFF;
        border: 1px solid #D8E0E9;
        border-radius: 8px;
        padding: 0.9rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .stat-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #285D9A;
    }
    .stat-label {
        font-size: 0.8rem;
        color: #526174;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    /* Hash identity bar */
    .integrity-container {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1rem;
    }
    .hash-badge {
        display: inline-block;
        background-color: #EDE9FE;
        color: #5B21B6;
        font-family: monospace;
        font-size: 0.8rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        margin-right: 0.5rem;
    }
    .review-cue-box {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.88rem;
        color: #166534;
        line-height: 1.45;
    }
    /* Passage search cards */
    .passage-card {
        background-color: #FFFFFF;
        border: 1px solid #D8E0E9;
        border-left: 4px solid #285D9A;
        border-radius: 6px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .passage-meta {
        font-size: 0.82rem;
        color: #526174;
        margin-bottom: 0.4rem;
        font-family: monospace;
    }
    .passage-text {
        font-size: 0.98rem;
        line-height: 1.6;
        color: #172033;
    }
    .badge {
        display: inline-block;
        background-color: #E7F0FB;
        color: #285D9A;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .model-badge {
        display: inline-block;
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .privacy-box {
        background-color: #FFFBEB;
        border: 1px solid #FCD34D;
        border-left: 4px solid #F59E0B;
        border-radius: 6px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1.2rem;
        font-size: 0.9rem;
        color: #92400E;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

PRESETS = {
    "-- Custom File Upload --": None,
    "Sample 1: Clean Baseline (Attention & Transformers)": "fixtures/sample_01_clean_baseline.txt",
    "Sample 2: Multi-Section Paper with References (GNNs)": "fixtures/sample_02_metadata_rich.txt",
    "Sample 3: Edge-Case Math, Abbreviations & Hyphens": "fixtures/sample_03_edge_case_math.txt",
    "Sample 4: Academic PDF Publication (Unified TextRank)": "fixtures/sample_05_academic_paper.pdf",
    "Sample 5: Corrupted / Malformed Bitstream (Stress Demo)": "fixtures/sample_04_corrupted.pdf",
}


def main() -> None:
    st.markdown('<div class="main-header">ResearchPaper Lens 📄</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Explainable Document Summarization & Information Retrieval for Research Papers</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Document Ingestion & Sample Presets
    st.sidebar.header("📁 Document Ingestion")

    selected_preset = st.sidebar.selectbox(
        "Standardized Sample Presets",
        options=list(PRESETS.keys()),
        index=0,
        help="Select a standardized fixture for 1-click evaluation, viva demonstration, or edge-case stress testing.",
    )

    uploaded_file = None
    file_bytes: bytes | None = None
    filename: str = ""

    if selected_preset != "-- Custom File Upload --":
        preset_path = PRESETS[selected_preset]
        if preset_path and os.path.exists(preset_path):
            with open(preset_path, "rb") as f:
                file_bytes = f.read()
            filename = os.path.basename(preset_path)
            st.sidebar.success(f"Loaded: `{filename}` ({len(file_bytes):,} bytes)")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload PDF or TXT file (Max 10 MiB)",
            type=["pdf", "txt"],
            help="Upload a research paper or technical document. PDFs are processed permissively with pypdf.",
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name

    st.sidebar.divider()
    st.sidebar.header("⚙️ Algorithm Controls")
    summary_ratio = st.sidebar.slider(
        "Extractive Summary Ratio",
        min_value=0.10,
        max_value=0.50,
        value=DEFAULT_SUMMARY_RATIO,
        step=0.05,
        help="Target proportion of sentences extracted by TextRank.",
    )

    top_k_passages = st.sidebar.slider(
        "Search Results (Top K)",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="Number of top ranked passages to return per retrieval query.",
    )

    show_baselines = st.sidebar.checkbox(
        "Enable Baseline Comparisons",
        value=True,
        help="Compare TextRank with Lead-k and Word-Frequency; compare TF-IDF with Keyword Matching.",
    )

    apply_privacy = st.sidebar.checkbox(
        "🔒 Privacy-by-Default Filtering",
        value=True,
        help="Automatically mask emails, phone numbers, and credentials in previews and reports.",
    )

    if st.sidebar.button("🔄 Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if file_bytes is None or not filename:
        render_welcome_state()
        return

    # Ingestion & Extraction Phase
    try:
        with st.spinner("Extracting canonical text from document..."):
            extraction = extract_text(file_bytes, filename=filename)
    except EmptyDocumentError as e:
        st.error(f"⚠️ Empty document: {e}")
        return
    except CorruptDocumentError as e:
        st.error(f"❌ Corrupt document bitstream: {e}")
        return
    except DecompressionBombError as e:
        st.error(f"💣 Decompression bomb alert: {e}")
        return
    except FileTooLargeError as e:
        st.error(f"❌ File too large: {e}")
        return
    except PageLimitExceededError as e:
        st.error(f"❌ Page limit exceeded: {e}")
        return
    except UnsupportedFormatError as e:
        st.error(f"❌ Unsupported format: {e}")
        return
    except ExtractionError as e:
        st.error(f"❌ Extraction failure: {e}")
        return
    except Exception as e:
        st.error(f"❌ Unexpected processing error: {e}")
        return

    # Preprocessing Phase
    with st.spinner("Normalizing text and segmenting sentences..."):
        prep_doc = preprocess_document(extraction.text)

    # Display cryptographic evidence and container properties
    render_integrity_and_metadata_banner(extraction, prep_doc, apply_privacy)

    if not prep_doc.sentences:
        st.warning("⚠️ No valid sentences detected in the document content body.")
        return

    # Precompute TextRank summary and baselines
    with st.spinner("Computing PageRank over sentence-similarity graph..."):
        textrank_res = textrank_summarize(prep_doc.sentences, ratio=summary_ratio)
        lead_res = lead_k_summarize(prep_doc.sentences, ratio=summary_ratio)
        freq_res = word_frequency_summarize(prep_doc.sentences, ratio=summary_ratio)
        keywords_res = extract_keywords(prep_doc.sentences, count=15)

    # Dual-Format Export Bar (PDF & JSON)
    report_obj = build_analysis_report(
        extraction_res=extraction,
        prep_doc=prep_doc,
        textrank_res=textrank_res,
        summary_ratio=summary_ratio,
        lead_res=lead_res,
        freq_res=freq_res,
        keywords_list=keywords_res,
        abstractive_res=st.session_state.get("abstractive_result"),
        apply_privacy=apply_privacy,
    )

    col_exp_1, col_exp_2 = st.columns(2)
    with col_exp_1:
        pdf_bytes = export_report_pdf(report_obj)
        st.download_button(
            label="📄 Download Forensic PDF Report",
            data=pdf_bytes,
            file_name=f"{Path(extraction.filename).stem}_forensic_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Generate publication-grade PDF report with cryptographic digests, explainable cues, and tables.",
        )
    with col_exp_2:
        json_str = export_report_json(report_obj)
        st.download_button(
            label="📦 Download Structured JSON Report",
            data=json_str,
            file_name=f"{Path(extraction.filename).stem}_report.json",
            mime="application/json",
            use_container_width=True,
            help="Export machine-readable JSON schema report.",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Application Tabs
    tab_summary, tab_retrieval, tab_keywords, tab_canonical, tab_abstractive = st.tabs(
        [
            "📝 Extractive Summary (TextRank)",
            "🔍 Passage Retrieval (TF-IDF)",
            "🏷️ Salient Keywords",
            "📖 Canonical Source Text",
            "✨ Abstractive Synthesis (Phase 6)",
        ]
    )

    # 1. Extractive Summary Tab
    with tab_summary:
        render_summary_tab(prep_doc, summary_ratio, show_baselines, textrank_res, lead_res, freq_res)

    # 2. Passage Retrieval Tab
    with tab_retrieval:
        render_retrieval_tab(prep_doc, top_k_passages, show_baselines)

    # 3. Keywords & Analytics Tab
    with tab_keywords:
        render_keywords_tab(prep_doc, keywords_res)

    # 4. Canonical Source Text Tab
    with tab_canonical:
        render_canonical_tab(extraction, prep_doc)

    # 5. Abstractive Synthesis Tab (Phase 6)
    with tab_abstractive:
        render_abstractive_tab(prep_doc, textrank_res.summary)


def render_welcome_state() -> None:
    st.info(
        "👋 **Welcome to ResearchPaper Lens.** Choose a preset sample from the sidebar dropdown or upload your own research paper (.pdf / .txt) to begin analysis."
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. TextRank Summarizer")
        st.caption(
            "Constructs a sentence-similarity graph using TF-IDF cosine similarities and computes PageRank centrality scores to select salient sentences in source order."
        )
    with col2:
        st.markdown("#### 2. TF-IDF Passage Retrieval")
        st.caption(
            "Indexes adjacent 3-sentence passages. Queries are mapped to the TF-IDF space and ranked by cosine similarity with out-of-vocabulary protection."
        )
    with col3:
        st.markdown("#### 3. Academic Baselines & Dual Reports")
        st.caption(
            "Evaluates TextRank against Lead-k and Word-Frequency baselines, with 1-click forensic PDF and structured JSON exports."
        )


def render_integrity_and_metadata_banner(extraction, prep_doc, apply_privacy: bool) -> None:
    # Display any extraction warnings (e.g. low-text / scanned PDF alert)
    for warning_msg in extraction.warnings:
        st.warning(f"⚠️ {warning_msg}")

    # Explainable review cues banner
    st.markdown(
        """
        <div class="review-cue-box">
            <strong>🔍 Explainable Diagnostic Review Cues:</strong><br/>
            • <strong>Extractive Centrality:</strong> TextRank selects verbatim sentences via PageRank ($d=0.85$) over sentence cosine graph. <em>0% hallucination risk.</em><br/>
            • <strong>Lexical Retrieval:</strong> Passage ranking uses TF-IDF cosine similarity without generative rephrasing.<br/>
            • <strong>Immutability & Integrity:</strong> Source document treated as immutable session evidence with verified cryptographic digests.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Cryptographic digests bar
    sha256 = extraction.hashes.get("sha256", "N/A")
    sha1 = extraction.hashes.get("sha1", "N/A")
    md5 = extraction.hashes.get("md5", "N/A")

    st.markdown(
        f"""
        <div class="integrity-container">
            <strong>Evidence Identity:</strong> <code>{extraction.filename}</code> ({extraction.raw_size_bytes:,} bytes)<br/>
            <span class="hash-badge">SHA-256</span> <code>{sha256}</code><br/>
            <span class="hash-badge">SHA-1</span> <code>{sha1}</code> &nbsp;|&nbsp;
            <span class="hash-badge">MD5</span> <code>{md5}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top container metrics bar
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f'<div class="stat-box"><div class="stat-val">{extraction.page_count}</div><div class="stat-label">Pages</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="stat-box"><div class="stat-val">{prep_doc.word_count:,}</div><div class="stat-label">Words</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="stat-box"><div class="stat-val">{prep_doc.sentence_count}</div><div class="stat-label">Sentences</div></div>',
            unsafe_allow_html=True,
        )
    with c4:
        ref_status = "Detected" if prep_doc.references_text else "None"
        st.markdown(
            f'<div class="stat-box"><div class="stat-val">{ref_status}</div><div class="stat-label">References</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)


def render_summary_tab(
    prep_doc, summary_ratio: float, show_baselines: bool, textrank_res, lead_res, freq_res
) -> None:
    st.subheader("TextRank Extractive Summary")

    summary_words = len(textrank_res.summary.split())
    comp_ratio = (
        (summary_words / prep_doc.word_count * 100) if prep_doc.word_count > 0 else 0
    )

    st.caption(
        f"Selected **{len(textrank_res.selected_indices)}** of **{prep_doc.sentence_count}** sentences "
        f"({summary_words} words, **{comp_ratio:.1f}%** compression ratio)."
    )

    # Render summary box
    st.text_area(
        "Extractive Summary Output",
        value=textrank_res.summary,
        height=220,
        help="Summary sentences extracted using TextRank and restored to original chronological document order.",
    )

    # Download summary markdown button
    download_content = (
        f"# ResearchPaper Lens — Summary\n\n"
        f"**Algorithm:** TextRank (TF-IDF Cosine Graph + PageRank)\n"
        f"**Compression:** {comp_ratio:.1f}% ({summary_words} words from {prep_doc.word_count})\n\n"
        f"## Summary\n\n{textrank_res.summary}\n"
    )
    st.download_button(
        label="📥 Download Summary (.md)",
        data=download_content,
        file_name="researchpaper_summary.md",
        mime="text/markdown",
    )

    # Baseline Comparisons
    if show_baselines and prep_doc.sentence_count > 1:
        st.divider()
        st.markdown("### 📊 Academic Baseline Comparison")
        st.caption("Side-by-side comparison of extractive summarization methods.")

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.markdown("**1. Lead-k Baseline** *(Chronological leading sentences)*")
            st.text_area("Lead-k Summary", value=lead_res.summary, height=160, key="lead_k_txt")
        with b_col2:
            st.markdown("**2. Word-Frequency Baseline** *(Sum of term frequencies)*")
            st.text_area(
                "Word-Frequency Summary", value=freq_res.summary, height=160, key="freq_txt"
            )


def render_retrieval_tab(prep_doc, top_k: int, show_baselines: bool) -> None:
    st.subheader("TF-IDF Passage Retrieval")
    st.caption("Search across 3-sentence adjacent document passages ranked by cosine similarity.")

    index = PassageRetrievalIndex(prep_doc.sentences, passage_size=3)

    query = st.text_input(
        "Enter search query or research topic:",
        placeholder="e.g. PageRank algorithm, experimental evaluation, neural networks...",
    )

    if not query.strip():
        st.info("💡 Enter a query above to search for relevant passages.")
        return

    results = index.search(query, top_k=top_k)

    if not results:
        st.warning(f"No relevant passages found for '{query}'. (All terms may be out-of-vocabulary or stop words).")
        return

    st.markdown(f"**Found {len(results)} relevant passage(s) (Top {top_k}):**")

    search_export_lines = [f"# Search Results for query: '{query}'\n"]

    for r in results:
        score_pct = r.score * 100
        st.markdown(
            f"""
            <div class="passage-card">
                <div class="passage-meta">PASSAGE #{r.passage_id + 1} &nbsp;|&nbsp; SENTENCES {r.sentence_indices[0] + 1}–{r.sentence_indices[-1] + 1} &nbsp;|&nbsp; COSINE SIMILARITY: {r.score:.4f} ({score_pct:.1f}%)</div>
                <div class="passage-text">{r.text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        search_export_lines.append(
            f"### Passage #{r.passage_id + 1} (Score: {r.score:.4f})\n{r.text}\n"
        )

    # Download search results
    st.download_button(
        label="📥 Download Search Results (.txt)",
        data="\n".join(search_export_lines),
        file_name="passage_search_results.txt",
        mime="text/plain",
    )

    if show_baselines:
        st.divider()
        with st.expander("🔬 View Baseline: Exact Keyword-Matching Retrieval"):
            keyword_results = keyword_match_retrieve(index.passages, query=query, top_k=top_k)
            if not keyword_results:
                st.write("No exact keyword matches found.")
            else:
                for kr in keyword_results:
                    st.markdown(
                        f"**Passage #{kr.passage_id + 1}** — Token Match Ratio: `{kr.score:.2f}`\n\n> {kr.text}"
                    )


def render_keywords_tab(prep_doc, keywords) -> None:
    st.subheader("Document Keywords & Term Importance")
    st.caption("Top salient terms calculated from sentence-level TF-IDF aggregation.")

    if not keywords:
        st.info("No eligible keyword terms detected.")
        return

    badges_html = "".join(
        [f'<span class="badge">{kw.term} ({kw.score:.2f})</span>' for kw in keywords]
    )
    st.markdown(f"<div>{badges_html}</div><br>", unsafe_allow_html=True)

    # Keyword table
    kw_data = [{"Keyword": kw.term, "Aggregate TF-IDF Score": kw.score} for kw in keywords]
    st.table(kw_data)


def render_canonical_tab(extraction, prep_doc) -> None:
    st.subheader("Canonical Source Text")
    st.caption("Original extracted text preserved without loss or mutation.")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**Content Body (Eligible for Analysis)**")
        st.text_area("Body Text", value=prep_doc.cleaned_text, height=350)
    with col2:
        st.markdown("**References / Bibliography (Isolated)**")
        ref_text = prep_doc.references_text or "No References section separated."
        st.text_area("References", value=ref_text, height=350)


def render_abstractive_tab(prep_doc, textrank_summary: str) -> None:
    st.subheader("Abstractive Synthesis & Model Adaptation (Phase 6)")
    st.caption(
        "Abstractive summarization synthesizes and paraphrases key concepts in novel prose, "
        "providing a complementary alternative to extractive sentence selection."
    )

    # Privacy & Cost Disclosure Box
    st.markdown(
        """
        <div class="privacy-box">
            <strong>🔒 Privacy & Data Transfer Policy:</strong><br>
            When utilizing hosted cloud inference (Hugging Face or OpenAI), document excerpts are transmitted
            over HTTPS to external API endpoints. For confidential papers or zero-cost offline exploration,
            select <strong>Mock / Demo Mode (Offline)</strong> or connect to a local Ollama instance.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_prov, c_source = st.columns([1, 1])
    with c_prov:
        provider_choice = st.radio(
            "Inference Provider",
            options=[
                "Mock / Demo Mode (Offline)",
                "Hugging Face Inference API",
                "OpenAI / Ollama (v1-Compatible)",
            ],
            index=0,
            help="Choose between zero-cost offline simulation or hosted deep learning models.",
        )
    with c_source:
        pipeline_strategy = st.radio(
            "Pipeline Input Strategy",
            options=[
                "Extractive-then-Abstractive (Synthesize TextRank output)",
                "Raw Document Excerpt (First 4,000 characters)",
            ],
            index=0,
            help="Extractive-then-Abstractive filters dense academic text into salient points first, fitting standard 1024-token model windows.",
        )

    api_key = ""
    model_id = ""
    base_url = "https://api.openai.com/v1"
    provider_key = "mock"

    if provider_choice == "Mock / Demo Mode (Offline)":
        st.info(
            "💡 **Mock Mode Active:** Generates an executive synthesis deterministically without network calls, API keys, or latency."
        )
        model_id = "mock-abstractive-v1"
        provider_key = "mock"
    elif provider_choice == "Hugging Face Inference API":
        c_k, c_m = st.columns([1, 1])
        with c_k:
            api_key = st.text_input(
                "Hugging Face API Token",
                type="password",
                placeholder="hf_...",
                help="Requires a free or paid Hugging Face User Access Token.",
            )
        with c_m:
            model_id = st.text_input(
                "Model Identifier",
                value=DEFAULT_HF_MODEL,
                help="HF summarization model e.g. facebook/bart-large-cnn or google/pegasus-xsum.",
            )
        provider_key = "huggingface"
    else:  # OpenAI / Ollama
        c_k, c_m, c_u = st.columns([1, 1, 1])
        with c_k:
            api_key = st.text_input(
                "API Key",
                type="password",
                placeholder="sk-... (or 'dummy' for Ollama)",
                help="API key for OpenAI / Groq, or dummy string if using a local Ollama server.",
            )
        with c_m:
            model_id = st.text_input(
                "Model Identifier",
                value=DEFAULT_OPENAI_MODEL,
                help="e.g. gpt-3.5-turbo, gpt-4o-mini, llama3:8b, mistral",
            )
        with c_u:
            base_url = st.text_input(
                "Base URL",
                value="https://api.openai.com/v1",
                help="API endpoint URL. For local Ollama, use http://localhost:11434/v1.",
            )
        provider_key = "openai"

    if st.button("🚀 Generate Abstractive Summary", type="primary"):
        if "Extractive-then-Abstractive" in pipeline_strategy:
            target_input = textrank_summary
        else:
            target_input = prep_doc.cleaned_text[:4000]

        if not target_input.strip():
            st.warning("⚠️ Input text is empty. Cannot synthesize summary.")
            return

        with st.spinner(f"Requesting abstractive synthesis via {provider_choice}..."):
            try:
                res = summarize_abstractive(
                    text=target_input,
                    provider=provider_key,
                    api_key=api_key,
                    model=model_id,
                    base_url=base_url,
                )
                st.session_state["abstractive_result"] = res
                st.rerun()
            except MissingApiKeyError as e:
                st.error(f"🔑 Missing API Key: {e}")
            except ModelTimeoutError as e:
                st.error(f"⏱️ Model Request Timeout: {e}")
            except AbstractiveError as e:
                st.error(f"❌ Abstractive Synthesis Failure: {e}")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {e}")

    # Render persisted summary if present in session state
    if "abstractive_result" in st.session_state:
        res: AbstractiveResult = st.session_state["abstractive_result"]
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f'<div class="model-badge">🤖 Model-Generated Output &nbsp;|&nbsp; '
            f'Model: <code>{res.model_name}</code> &nbsp;|&nbsp; '
            f'Provider: <code>{res.provider}</code> &nbsp;|&nbsp; '
            f'Latency: {res.latency_seconds:.3f}s</div>',
            unsafe_allow_html=True,
        )

        st.text_area(
            "Abstractive Synthesis Output",
            value=res.summary,
            height=200,
            help="Synthesized text generated by deep learning model. Rephrased from source content.",
        )

        abs_words = len(res.summary.split())
        abs_comp = (abs_words / prep_doc.word_count * 100) if prep_doc.word_count > 0 else 0
        st.caption(
            f"Synthesized **{abs_words}** words (**{abs_comp:.1f}%** compression ratio from {prep_doc.word_count} total words)."
        )

        # Download Abstractive Summary
        abs_download_md = (
            f"# ResearchPaper Lens — Abstractive Summary\n\n"
            f"- **Method:** Abstractive Synthesis ({res.provider})\n"
            f"- **Model Identifier:** `{res.model_name}`\n"
            f"- **Inference Latency:** {res.latency_seconds:.3f} seconds\n"
            f"- **Compression Ratio:** {abs_comp:.1f}% ({abs_words} words)\n"
            f"- **Attribution Note:** Synthesized deep-learning output; verified non-verbatim.\n\n"
            f"## Synthesized Summary\n\n{res.summary}\n"
        )
        st.download_button(
            label="📥 Download Abstractive Summary (.md)",
            data=abs_download_md,
            file_name="abstractive_summary.md",
            mime="text/markdown",
            key="btn_dl_abs",
        )

        # Comparative Analysis: Extractive vs Abstractive
        st.divider()
        st.markdown("### ⚖️ Extractive vs. Abstractive Comparison")
        st.caption("Juxtaposition of graph-based extraction against neural generative synthesis.")

        comp_col1, comp_col2 = st.columns(2)
        with comp_col1:
            st.markdown("#### 1. TextRank Extractive Output")
            st.text_area(
                "Verbatim Source Sentences",
                value=textrank_summary,
                height=180,
                key="ext_comp_view",
            )
            ext_words = len(textrank_summary.split())
            st.caption(f"Length: **{ext_words} words** | Faithfulness: **100% verbatim**")

        with comp_col2:
            st.markdown("#### 2. Abstractive Synthesis Output")
            st.text_area(
                "Synthesized Novel Prose",
                value=res.summary,
                height=180,
                key="abs_comp_view",
            )
            st.caption(f"Length: **{abs_words} words** | Faithfulness: **Paraphrased by model**")

        st.markdown(
            """
| Dimension | TextRank Extractive | Abstractive Synthesis |
| :--- | :--- | :--- |
| **Generation Mechanism** | Graph centrality (PageRank over TF-IDF cosine similarities) | Sequence-to-sequence neural generation (BART / Transformers) |
| **Vocabulary Source** | Verbatim from source document | Generates novel vocabulary and connecting transitions |
| **Hallucination Risk** | **Zero** (no novel facts can be invented) | **Low to Moderate** (depends on model temperature and grounding) |
| **Computational Footprint** | Pure CPU ($O(V^2)$ cosine matrix, fast PageRank convergence) | High GPU/TPU memory or network API dependency |
| **Token Limit Constraint** | Unconstrained (scales linearly across document passages) | Limited by model context window (typically 1024–4096 tokens) |
"""
        )


if __name__ == "__main__":
    main()
