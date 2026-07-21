# Cost and latency

A RAG request spends time and money in a few predictable places: embedding the query,
searching the index, and generating the answer. Generation usually dominates both,
because token cost and latency scale with how much context you stuff into the prompt.

Retrieve less, but retrieve well. Passing the top three well-ranked passages often beats
passing the top ten, since extra passages add tokens, cost, and distraction while rarely
adding the missing fact. A reranker lets you retrieve broadly then trim to a tight, high-
precision context, which cuts generation cost more than it adds in reranking cost.

Cache aggressively. Identical or near-identical questions can return a cached answer, and
document embeddings should be computed once at indexing time and reused, never recomputed
per query. Prompt caching on the model side can also make the fixed instruction and
context cheaper to resend.

Right-size the model. A small fast model handles routing, query rewriting, and simple
lookups; reserve the large expensive model for answers that need real synthesis. Streaming
the response does not reduce cost but it improves perceived latency, since the user reads
the first tokens while the rest generate.

Measure the whole budget, not just quality. Track tokens, latency, and dollar cost per
request alongside accuracy, because a reranker or an agent that buys one point of accuracy
for triple the latency may not be worth shipping. The cheapest RAG system is the one that
retrieves precisely enough to answer in one short, well-grounded pass.
