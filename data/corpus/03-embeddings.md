# Embeddings

An embedding maps a piece of text to a dense vector such that semantically
similar texts land near each other in the vector space. Retrieval compares the
query vector against stored chunk vectors and returns the closest ones. Because
matching happens in meaning-space rather than keyword-space, a query about
"how to cut cloud costs" can retrieve a passage about "reducing infrastructure
spend" even with no shared words.

Similarity is measured with cosine similarity, the angle between two vectors. If
vectors are L2-normalized to unit length, cosine similarity equals their dot
product, which makes search a single matrix multiplication — fast and simple.
Normalize on write so every comparison is consistent.

Use the same embedding model for indexing and for queries; mixing models puts
documents and questions in incompatible spaces and retrieval collapses. Higher
dimensionality can capture more nuance but costs more storage and compute; many
modern models support dimensionality reduction (Matryoshka embeddings) so you can
trade a little accuracy for a smaller index, but you must re-normalize after
truncating the vector.

Embeddings have limits. They capture topical similarity well but can miss precise
constraints — a number, a name, an explicit negation — because those details wash
out in a single averaged vector. That is why hybrid retrieval, which combines
dense embeddings with sparse keyword search such as BM25, outperforms either
alone on queries that hinge on an exact term.
