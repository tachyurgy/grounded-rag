# Long context versus RAG

As model context windows grow to hundreds of thousands of tokens, a tempting idea is to
skip retrieval and paste the whole corpus into the prompt. For small, static document sets
this can work, but it does not replace RAG for most real systems.

Cost and latency scale with context length. Sending a hundred thousand tokens on every
request is expensive and slow, whereas retrieval sends only the few passages that matter.
For a corpus larger than a window — and most useful corpora are — stuffing everything in is
not even possible, so retrieval is mandatory, not optional.

Attention is not uniform across a long context. Models exhibit a "lost in the middle"
effect, attending well to the start and end of a long prompt but degrading on facts buried
in the middle. Retrieval sidesteps this by surfacing the relevant passage into a short,
high-attention context instead of hoping the model finds it in a wall of text.

Freshness and attribution still favour retrieval. An external index is updated by changing
a document, while a long-context dump must be reassembled and resent each time, and pinpoint
citations are far easier when the answer draws on a handful of identified passages.

Long context and RAG are complements. A larger window lets each retrieved chunk be bigger
and lets you pass more of them, improving recall, while retrieval keeps the prompt focused
and affordable. The durable pattern is retrieve-then-read, now with more generous chunks,
not read-everything.
