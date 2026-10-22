"""ResearchPaper Lens — Streamlit Application.

Document summarization and information retrieval workspace.
Adheres strictly to design.md, architecture.md, and prd.md.
"""

from __future__ import annotations

import io
from typing import Optional

import streamlit as st

from src.baselines import (
    keyword_match_retrieve,
    lead_k_summarize,
    word_frequency_summarize,
)
from src.extract import (
    EmptyDocumentError,
    ExtractionError,
    FileTooLargeError,
    PageLimitExceededError,
    UnsupportedFormatError,
    extract_text,
)
from src.keywords import extract_keywords
from src.preprocess import preprocess_document
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
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
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
        margin-bottom: 1.5rem;
    }
    /* Stat cards */
    .stat-box {
        background-color: #FFFFFF;
        border: 1px solid #D8E0E9;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .stat-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #285D9A;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #526174;
        text-transform: uppercase;
        letter-spacing: 0.04em;
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
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    st.markdown('<div class="main-header">ResearchPaper Lens</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Explainable document summarization and Information Retrieval for academic papers.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Document Upload & Algorithm Controls
    st.sidebar.header("📁 Document Ingestion")
    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF or TXT file (Max 10 MiB)",
        type=["pdf", "txt"],
        help="Upload a research paper or technical document. PDFs are processed with pypdf.",
    )

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

    if uploaded_file is None:
        render_welcome_state()
        return

    # Ingestion & Extraction Phase
    try:
        with st.spinner("Extracting canonical text from document..."):
            extraction = extract_text(uploaded_file, filename=uploaded_file.name)
    except FileTooLargeError as e:
        st.error(f"❌ File too large: {e}")
        return
    except PageLimitExceededError as e:
        st.error(f"❌ Page limit exceeded: {e}")
        return
    except EmptyDocumentError as e:
        st.error(f"⚠️ Empty document: {e}")
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

    # Display document metadata banner and warnings
    render_metadata_banner(extraction, prep_doc)

    if not prep_doc.sentences:
        st.warning("⚠️ No valid sentences detected in the document content body.")
        return

    # Main Application Tabs
    tab_summary, tab_retrieval, tab_keywords, tab_canonical = st.tabs(
        [
            "📝 Extractive Summary (TextRank)",
            "🔍 Passage Retrieval (TF-IDF)",
            "🏷️ Salient Keywords",
            "📖 Canonical Source Text",
        ]
    )

    # 1. Extractive Summary Tab
    with tab_summary:
        render_summary_tab(prep_doc, summary_ratio, show_baselines)

    # 2. Passage Retrieval Tab
    with tab_retrieval:
        render_retrieval_tab(prep_doc, top_k_passages, show_baselines)

    # 3. Keywords & Analytics Tab
    with tab_keywords:
        render_keywords_tab(prep_doc)

    # 4. Canonical Source Text Tab
    with tab_canonical:
        render_canonical_tab(extraction, prep_doc)


def render_welcome_state() -> None:
    st.info(
        "👋 **Welcome to ResearchPaper Lens.** To begin analysis, upload a research paper (.pdf) or text notes (.txt) using the sidebar."
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
        st.markdown("#### 3. Academic Baselines")
        st.caption(
            "Evaluates TextRank against Lead-k and Word-Frequency baselines, and compares TF-IDF with exact keyword-match retrieval."
        )


def render_metadata_banner(extraction, prep_doc) -> None:
    # Display any extraction warnings (e.g. low-text / scanned PDF alert)
    for warning_msg in extraction.warnings:
        st.warning(f"⚠️ {warning_msg}")

    # Top metrics bar
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


def render_summary_tab(prep_doc, summary_ratio: float, show_baselines: bool) -> None:
    st.subheader("TextRank Extractive Summary")

    with st.spinner("Computing PageRank over sentence-similarity graph..."):
        textrank_res = textrank_summarize(prep_doc.sentences, ratio=summary_ratio)

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

    # Download summary button
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

        lead_res = lead_k_summarize(prep_doc.sentences, ratio=summary_ratio)
        freq_res = word_frequency_summarize(prep_doc.sentences, ratio=summary_ratio)

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


def render_keywords_tab(prep_doc) -> None:
    st.subheader("Document Keywords & Term Importance")
    st.caption("Top 15 salient terms calculated from sentence-level TF-IDF aggregation.")

    keywords = extract_keywords(prep_doc.sentences, count=15)

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


if __name__ == "__main__":
    main()
