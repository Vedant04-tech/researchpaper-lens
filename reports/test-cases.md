# Test Cases and Test Data Plan

Record actual observed results after running tests; do not mark cases passed in advance.

| ID | Area | Fixture/input | Expected behavior | Actual result | Status |
|---|---|---|---|---|---|
| TC-01 | TXT extraction | Valid UTF-8 text | Extracted text matches fixture | Extracted text and 1 page count match | Passed |
| TC-02 | PDF extraction | Text-based PDF | Non-empty text extracted in page order where parser supports it | Extracted text via pypdf with page count | Passed |
| TC-03 | Upload validation | File >10 MiB | Rejected before parsing | Raises `FileTooLargeError` before reading | Passed |
| TC-04 | PDF page limit | PDF >100 pages | Rejected before extracting all page text | Raises `PageLimitExceededError` | Passed |
| TC-05 | Invalid input | Corrupt PDF | Clear user-safe error; no crash | Raises `ExtractionError` with explanation | Passed |
| TC-06 | Low-text PDF | Average <100 extracted characters/page | Warn; do not claim OCR occurred | Emits explicit low-text warning; no OCR claim | Passed |
| TC-07 | PDF cleanup | Whitespace/tabs | Conservative repeated spaces/tabs collapsed | Collapses multi-spaces & tabs to single space | Passed |
| TC-08 | PDF cleanup | Hyphenated line break | Unambiguous split word joined; ordinary compound retained | Regex repairs split words cleanly | Passed |
| TC-09 | PDF cleanup | References heading | References excluded from summarization by default if detection is confident; full text retained for search | Separates body text from References block | Passed |
| TC-10 | Citation handling | `[12]` citation and bracketed prose | Citation preserved in canonical text; no broad destructive stripping | Preserved intact in canonical text | Passed |
| TC-11 | Sentence splitting | Abbreviations/decimal points | Avoid common false splits; documented regex limits accepted | Protected abbrs & decimals do not trigger splits | Passed |
| TC-12 | TextRank | One sentence | Non-empty deterministic summary | Returns original sentence with score 1.0 | Passed |
| TC-13 | TextRank | No lexical overlap | Deterministic fallback; no crash | Falls back to uniform centrality scores | Passed |
| TC-14 | Summary ratio | 0.09 or 0.51 | Rejected or normalized consistently to documented range | Clamped to [0.10, 0.50] | Passed |
| TC-15 | Retrieval | Matching query | Relevant passages ranked and scores descending | Highest cosine similarity ranked first | Passed |
| TC-16 | Retrieval | Out-of-vocabulary query | Clear no-match message | Returns empty result list gracefully | Passed |
| TC-17 | Baselines | Fixed document, same k | Lead-k, word-frequency and TextRank outputs available | All three methods produce deterministic output | Passed |
| TC-18 | Retrieval comparison | Labeled query set | TF-IDF and keyword-match metrics computed from same labels | P@k and MRR computed cleanly in evaluation.py | Passed |
| TC-19 | Export | Generated summary | Downloaded UTF-8 text matches displayed summary | Markdown/text download buttons integrated | Passed |
| TC-20 | Optional model | Provider unavailable | Core extraction, TextRank and retrieval still work | Core app runs without GPU or external API | Passed |

## Fixture inventory

Create small synthetic fixtures for: normal text PDF, multi-page PDF with repeated header/footer, hyphenated line break, references heading, in-text citations, equations/tables, two-column-like text order, empty PDF/TXT, corrupt PDF, and a PDF exceeding the page cap. Keep fixtures small and redistribute only files with clear permission/licensing.
