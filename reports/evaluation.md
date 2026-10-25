# Evaluation Plan and Results

**Status:** Plan only; no benchmark results have been collected yet.

## Dataset manifest

| ID | Document/source | License/provenance | Pages/words | Reference summary available? | Notes |
|---|---|---|---:|---|---|
| DOC-01 | Pending | Pending | Pending | Pending | Pending |
| DOC-02 | Pending | Pending | Pending | Pending | Pending |
| DOC-03 | Pending | Pending | Pending | Pending | Pending |
| DOC-04 | Pending | Pending | Pending | Pending | Pending |
| DOC-05 | Pending | Pending | Pending | Pending | Pending |

Use 5–10 documents where practical. Prefer papers with clear reuse permissions, such as appropriately licensed CC-BY arXiv papers, and record the exact source URL, license, access date, and any transformations in the provenance column. Do not publish copyrighted full papers or private documents without permission. Record the dataset selection method and environment.

For research papers, the paper abstract may be used as the human-written reference summary when it is suitable for the evaluation task. **Remove the abstract from the input text before running any summarization method** so the system cannot copy the reference into its output (reference leakage). Record this preprocessing step for each paper. If an abstract is unavailable or unsuitable, use a separately prepared human-written reference and document how it was created.

## Summary comparison

Compare TextRank-style ranking, lead-k, and word-frequency baseline using the same eligible sentence list and same output sentence count. Add abstractive output only if implemented and actually run.

| Document | Method | Input words | Summary words | Compression ratio | Runtime (ms) | ROUGE-1 | ROUGE-2 | ROUGE-L | Qualitative notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Pending | Pending | — | — | — | — | N/A | N/A | N/A | Not measured |

## Retrieval comparison

Create at least 10 query/relevance-judgment pairs if feasible. Label relevant passages before looking at algorithm rankings to reduce bias. Run both TF-IDF/cosine and case-insensitive keyword matching against the same query set.

| Query ID | Document ID | Query | Relevant passage IDs | TF-IDF top-5 | Keyword-match top-5 | P@5 / MRR details |
|---|---|---|---|---|---|---|
| Q-01 | Pending | Pending | Pending | Pending | Pending | Pending |

Report aggregate Precision@5 and MRR only after the judgments and runs exist. If fewer queries are available, disclose the count.

## Method and environment

- Python version: pending verification
- Dependency versions: pending lock/pin after testing
- Hardware/OS: record actual environment
- Commands: record exact evaluation command
- Randomness: keep algorithm settings deterministic where possible

## Interpretation checklist

- Separate measured values from subjective observations.
- Explain that ROUGE is lexical overlap, not factuality.
- Inspect missing key points, redundant sentences, readability, and unsupported claims.
- Discuss PDF extraction errors, especially two-column order, tables, equations, headers/footers, and reference sections.
- Do not invent or extrapolate results from a tiny dataset.
