# Algorithms and Mathematical Model

This file is the working source for the report's algorithm section. Align it with the implemented code before submission.

## 1. TF-IDF

For term \(t\) in document \(d\):

\[
\operatorname{tfidf}(t,d)=\operatorname{tf}(t,d)\times\operatorname{idf}(t)
\]

A common smoothed inverse-document-frequency form is:

\[
\operatorname{idf}(t)=\log\left(\frac{1+N}{1+\operatorname{df}(t)}\right)+1
\]

where \(N\) is the number of documents in the fitted corpus and \(\operatorname{df}(t)\) is the number of corpus documents containing \(t\). The project uses scikit-learn's `TfidfVectorizer`; report the exact configuration used by the implementation. For keyword extraction, individual sentences are the corpus documents, not the whole upload treated as one document.

## 2. Cosine similarity

For non-zero vectors \(a\) and \(b\):

\[
\operatorname{cos}(a,b)=\frac{a\cdot b}{\|a\|_2\|b\|_2}
\]

The value is used to score sentence-to-sentence similarity for the graph and query-to-passage similarity for retrieval. Zero vectors must be handled explicitly.

## 3. PageRank on a sentence-similarity graph

Each eligible sentence is a node. Edge weights are non-negative cosine similarities. The implementation uses PageRank with damping \(d=0.85\), maximum iterations 100, and tolerance \(10^{-6}\):

\[
R_i=\frac{1-d}{n}+d\sum_{j\in In(i)}\frac{w_{ji}}{\sum_{k\in Out(j)}w_{jk}}R_j
\]

The implemented graph is based on TF-IDF sentence cosine similarity plus PageRank. Describe it as **TextRank-style graph-based extractive summarization (LexRank-like sentence-similarity graph)** rather than claiming it is a strict reproduction of original TextRank's word-co-occurrence graph.

## 4. Sentence selection

For \(n\) eligible sentences and ratio \(r\in[0.10,0.50]\):

\[
k=\min(n,\max(1,\lceil rn\rceil))
\]

Select the top \(k\) scores, then restore their original source order. If the graph has no useful edges, use a deterministic documented fallback.

## 5. Baselines

- **Lead-k:** select the first \(k\) eligible sentences.
- **Word-frequency baseline:** score sentences by normalized frequency of non-stop-word terms, select top \(k\), then restore source order.
- **Keyword-match retrieval:** rank passages by case-insensitive query-token overlap; define tie-breaking consistently.

## 6. Retrieval evaluation

For each query, label relevant passages before running the comparison.

\[
P@k=\frac{\text{relevant passages in top }k}{k}
\]

For a query with a first relevant result at rank \(r_q\), reciprocal rank is \(RR_q=1/r_q\); if no relevant result is returned, it is zero. Mean Reciprocal Rank is:

\[
MRR=\frac{1}{|Q|}\sum_{q\in Q}RR_q
\]

State the actual number of queries and relevance judgments. These metrics are not meaningful without documented labels.

## 7. Summary evaluation

Compression ratio:

\[
\text{Compression Ratio}=\frac{\text{summary word count}}{\text{source word count}}
\]

ROUGE-1/2/L may be reported only when human-written reference summaries are available. ROUGE overlap does not establish factuality; pair it with qualitative checks.

## 8. Complexity notes

- Dense pairwise similarity across \(n\) sentences requires \(O(n^2)\) pair comparisons, multiplied by vector-operation cost.
- PageRank iterations require approximately \(O(I(n+e))\) for \(I\) iterations and \(e\) graph edges.
- Dense query scoring across \(p\) passages and \(m\) features is approximately \(O(pm)\); sparse vector representations may reduce practical work.
- Report theoretical estimates separately from measured runtime.
