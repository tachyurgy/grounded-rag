# Vector databases

A vector database stores embeddings and answers nearest-neighbour queries over
them. At small scale an exact search — comparing the query against every stored
vector — is fast enough and always correct. As the index grows into the millions,
exact search becomes too slow and systems switch to an Approximate Nearest
Neighbour (ANN) index such as HNSW or IVF, which trades a small amount of recall
for a large speedup.

HNSW (Hierarchical Navigable Small World) builds a layered proximity graph and
walks it greedily from a coarse top layer to a fine bottom layer. It offers
excellent recall and latency at the cost of memory and slower inserts. IVF
partitions vectors into clusters and searches only the clusters nearest the
query, which is memory-light but sensitive to how many clusters you probe.

Popular stores include Chroma and FAISS for embedded or in-process use, and
pgvector, Qdrant, Weaviate, and Milvus for standalone services. pgvector is
attractive when you already run PostgreSQL because it keeps vectors, metadata,
and transactional data in one database with familiar SQL and joins.

Two features matter as much as raw search speed. Metadata filtering restricts
results by attributes such as document id, tenant, or date, and should be applied
during the search rather than after, or you may filter away everything the ANN
index returned. Persistence and incremental upserts let you add and delete
documents without rebuilding the whole index. For any store, keep the embedding
model and vector dimensionality fixed for the life of an index; changing either
requires a full re-embed and re-index.
