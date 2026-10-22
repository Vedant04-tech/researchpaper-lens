# ResearchPaper Lens 📄

> **Explainable Document Summarization & Information Retrieval for Research Papers**  
> Built as an SPPU Information Retrieval mini-project and a transparent, portfolio-grade NLP artifact.

---

## 🌟 Key Capabilities

1. **Document Ingestion & Validation** (`src/extract.py`)
   - Permissive text extraction using `pypdf` (BSD-3-Clause) and UTF-8/Latin-1 plain text fallback.
   - Enforces strict architectural contracts: 10 MiB upload cap, 100-page PDF limit, and a `< 100 chars/page` heuristic warning for scanned image-only PDFs (no false claims of OCR).
2. **Text Preprocessing & Sentence Splitting** (`src/preprocess.py`)
   - Normalizes excessive whitespace and repairs hyphenated line wraps.
   - Deterministic regex-based sentence segmentation with protected academic abbreviations (`e.g.`, `i.e.`, `et al.`, `Dr.`, `Fig.`, `Eq.`, decimals like `3.14`). No runtime downloads (NLTK/spaCy-free).
   - Automatically isolates trailing **References / Bibliography** sections so citations do not distort summary centrality.
3. **TextRank Extractive Summarization** (`src/textrank.py`)
   - Constructs an undirected sentence-similarity graph using TF-IDF vectors and cosine similarities.
   - Calculates vertex centrality via `networkx` PageRank ($d=0.85$, max iterations 100, tolerance $10^{-6}$).
   - Filters fragments under 5 words with deterministic fallbacks.
   - Restores selected sentences to their original chronological document order.
4. **TF-IDF Passage Retrieval** (`src/retrieval.py`)
   - Partitions documents into adjacent 3-sentence passages.
   - Indexes passages in an in-memory TF-IDF space and scores queries via cosine similarity with out-of-vocabulary protection.
5. **Salient Keyword Extraction** (`src/keywords.py`)
   - Treats document sentences as individual documents in a corpus, fitting TF-IDF and aggregating term weights to surface top technical terms.
6. **Academic Baselines** (`src/baselines.py`)
   - **Lead-k summarizer:** Extracts leading $k$ sentences chronologically.
   - **Word-Frequency summarizer:** Scores sentences by normalized non-stop-word term frequencies.
   - **Exact Keyword-Matching retrieval:** Evaluates query token presence across passages.
7. **Evaluation Metrics** (`src/evaluation.py`)
   - Information Retrieval: **Precision@k** and **Mean Reciprocal Rank (MRR)**.
   - Summarization: **Compression ratio** and local **ROUGE-1 / ROUGE-2** precision, recall, and F1.
8. **Interactive Streamlit Workspace** (`app.py`)
   - Calm, editorial research workspace adhering to `design.md`.
   - Real-time compression metrics, side-by-side baseline comparisons, instant passage search, and one-click markdown/text exports.

---

## 📁 Repository Structure

```text
ResearchPaper-Lens/
├── app.py                       # Streamlit UI & interactive workspace
├── requirements.txt             # Pinned core runtime dependencies
├── requirements-dev.txt         # Development & test tooling (pytest)
├── README.md                    # Project documentation
├── CLAUDE.md                    # Agent entrypoint and coding instructions
├── prd.md                       # Product requirements document
├── architecture.md              # Technical contracts, defaults, and data flows
├── rules.md                     # Engineering guardrails & TDD principles
├── phases.md                    # Phase delivery roadmap & exit gates
├── design.md                    # Editorial UI design system & tokens
├── memory.md                    # Durable decision log & status ledger
├── src/
│   ├── __init__.py
│   ├── extract.py               # Ingestion, validation, and PDF/TXT extraction
│   ├── preprocess.py            # Sentence splitting, abbreviation protection, references isolation
│   ├── textrank.py              # TF-IDF cosine graph & PageRank summarizer
│   ├── retrieval.py             # Passage grouping & TF-IDF cosine search
│   ├── keywords.py              # Sentence-level TF-IDF keyword extraction
│   ├── baselines.py             # Lead-k, word-frequency, and keyword-match baselines
│   └── evaluation.py            # Precision@k, MRR, compression ratio, ROUGE-n
├── tests/
│   ├── test_smoke.py            # Smoke tests verifying dependency imports
│   ├── test_extract.py          # 10 unit tests covering extraction contracts
│   ├── test_preprocess.py       # 8 unit tests covering sentence segmentation & cleanup
│   ├── test_textrank.py         # 9 unit tests covering graph ranking & order preservation
│   ├── test_retrieval.py        # 8 unit tests covering passage grouping & TF-IDF retrieval
│   ├── test_keywords.py         # 5 unit tests covering keyword extraction
│   ├── test_baselines.py        # 6 unit tests covering academic baselines
│   ├── test_evaluation.py       # 7 unit tests covering IR & summarization metrics
│   └── fixtures/                # Synthetic test document fixtures
└── reports/
    ├── report-outline.md        # Academic report structure for college viva
    ├── algorithms.md            # Mathematical model, formulas, and complexity
    ├── test-cases.md            # Verified 20-test-case validation ledger
    └── evaluation.md            # Dataset manifest and comparison notes
```

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- Python 3.11.x (tested on 3.11.9)
- Git

### 2. Environment Activation & Dependencies
```bash
# Clone the repository
git clone <repo-url>
cd "ResearchPaper Lens"

# Create and activate virtual environment (Windows PowerShell)
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install runtime and development dependencies
pip install -r requirements-dev.txt
```

### 3. Run the Web Application
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

### 4. Run the Test Suite
```bash
pytest
```
Expected output: **54 passed unit tests**.

---

## 📊 Mathematical Foundations

- **TF-IDF Term Weighting:**
  $$\text{tfidf}(t, d) = \text{tf}(t, d) \times \left(\log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1\right)$$
- **Cosine Similarity:**
  $$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
- **PageRank Vertex Centrality:**
  $$R_i = \frac{1 - d}{n} + d \sum_{j \in \text{In}(i)} \frac{w_{ji}}{\sum_{k \in \text{Out}(j)} w_{jk}} R_j$$
  *(Damping factor $d = 0.85$, max iterations $= 100$, tolerance $= 10^{-6}$)*

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.
