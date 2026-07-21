# Chunking strategies

Chunking splits documents into passages small enough to embed and retrieve
precisely, yet large enough to carry a self-contained idea. Chunk size is the
single most impactful indexing knob. Passages that are too small fragment a
thought across boundaries so no single chunk answers the question; passages that
are too large dilute the embedding with unrelated content and waste context
window at query time.

A practical default is 500 to 1000 characters (roughly 128 to 256 tokens) with
a 10 to 20 percent overlap between neighbours. The overlap carries a tail of the
previous chunk into the next one so a sentence that straddles a boundary is not
lost to either side.

Prefer structure-aware splitting over blind fixed-width cuts. Split on paragraph
and section boundaries first, then pack whole paragraphs up to the size budget.
For Markdown, keep headings attached to the text beneath them; for code, split on
function and class boundaries. Recursive character splitting tries a list of
separators in order — paragraph, line, sentence, word — falling back only when a
unit exceeds the budget.

Enrich each chunk with metadata: its source document, a section title, and an
ordinal position. Metadata enables filtered retrieval (restrict to one document
or date range) and lets the answer cite a precise location. A useful pattern is
"small-to-big": embed small precise chunks for matching, but return an expanded
parent window for generation, giving the model surrounding context without
hurting retrieval precision.
