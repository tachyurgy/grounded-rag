# Grounding and hallucination

A hallucination is a confident claim that is not supported by the model's
sources. In a RAG system the fix is not a better model alone but a discipline of
grounding: every claim in the answer must be traceable to a retrieved passage.

Three techniques enforce grounding. First, **instruct and constrain**: tell the
model to answer only from the numbered sources, to cite each claim inline with
the source number, and to say "the sources do not contain this" when they do not.
An explicit permission to abstain sharply reduces fabrication.

Second, **verify citations after generation**. Parse the citation markers the
model emitted and check two things: that every cited index refers to a source
that was actually retrieved (models sometimes invent citation numbers), and that
each sentence's content genuinely overlaps the passage it cites. A cheap lexical
overlap check catches the common case where a sentence cites a real source that
does not in fact support it. Answers that fail can be flagged, regenerated, or
refused.

Third, **measure faithfulness** as an evaluation metric. Faithfulness asks what
fraction of the answer's claims are entailed by the retrieved context; answer
relevance asks whether the answer addresses the question. The two are
independent: an answer can be perfectly faithful yet unhelpful, or fluent and
relevant yet unfaithful.

The strongest guarantee comes from making abstention a first-class outcome. A
system that says "I don't know" when the corpus is silent is more trustworthy
than one that always produces an answer, because users can rely on a claim being
backed by a citation they can open and read.
