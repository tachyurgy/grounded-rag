# Evaluating RAG systems

Evaluate retrieval and generation separately, because they fail for different
reasons and demand different fixes.

**Retrieval metrics** treat the pipeline as search. Recall@k asks whether the
relevant chunk appears in the top k results; it is the ceiling on answer quality,
since a passage never retrieved can never be used. Precision@k and Mean
Reciprocal Rank measure how highly the relevant chunk ranks. Building a small
labelled set of question-to-chunk pairs, even a few dozen, turns retrieval tuning
into a measurable loop instead of guesswork.

**Generation metrics** judge the answer given the retrieved context.
Faithfulness measures the fraction of the answer's claims that are supported by
the context — the direct anti-hallucination signal. Answer relevance measures
whether the answer addresses the question. Context precision and context recall
assess whether the retrieved passages were the right ones to hand the generator.

Two practical approaches produce these scores. Reference-based evaluation compares
answers to gold answers with exact match or semantic similarity, which is precise
but needs labelled data. LLM-as-judge uses a strong model to grade faithfulness
and relevance against a rubric, which scales cheaply but must itself be validated
against human ratings on a sample, since judges have biases such as favouring
longer or more confident answers.

Whatever the method, keep a fixed regression set and re-run it on every change so
you can tell an improvement from a regression. Track cost and latency alongside
quality; a reranker or an agent that adds a point of accuracy for triple the
latency may not be worth shipping. The goal is a small, trusted evaluation loop
that makes each change accountable to evidence.
