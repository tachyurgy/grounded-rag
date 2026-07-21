# Retrieval strategies

Naive top-k dense retrieval is a strong baseline, but several strategies raise
answer quality when it falls short.

**Hybrid search** runs dense embedding search and sparse keyword search (BM25)
together and fuses the rankings, commonly with Reciprocal Rank Fusion. Dense
search captures meaning; sparse search nails exact terms like error codes, part
numbers, and proper nouns. Fusing them is more robust than either alone.

**Reranking** retrieves a generous candidate set — say the top 50 — then scores
each candidate against the query with a cross-encoder that reads the query and
passage together. Cross-encoders are far more accurate than the bi-encoder used
for first-stage retrieval but too slow to run over the whole corpus, so they are
applied only to the shortlist. Rerank, then keep the top few for the prompt.

**Query transformation** rewrites the user's question before retrieval.
Multi-query expansion generates several paraphrases and unions their results.
HyDE (Hypothetical Document Embeddings) asks the model to draft a hypothetical
answer and embeds that, since a full answer often sits closer to the target
passage than a terse question does.

**Maximal Marginal Relevance (MMR)** diversifies results by penalizing
candidates that are too similar to ones already selected, which prevents a
context window full of near-duplicate chunks and widens coverage.

Choose based on the failure you observe. If retrieval misses exact terms, add
hybrid search. If the right passage is retrieved but ranked low, add a reranker.
If questions are vague, transform the query. Measure recall@k before and after so
each addition is justified by evidence, not intuition.
