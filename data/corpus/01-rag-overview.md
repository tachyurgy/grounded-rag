# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) grounds a language model's output in an
external corpus instead of relying only on the parameters learned during
training. At query time the system retrieves the most relevant passages from a
knowledge base and places them in the model's context window, so the model
answers from evidence it can see rather than from memory it might confabulate.

A RAG pipeline has two phases. The **indexing** phase runs offline: documents
are loaded, split into chunks, embedded into vectors, and written to a vector
store. The **query** phase runs online: the user question is embedded with the
same model, the nearest chunks are retrieved, and those chunks plus the question
are formatted into a prompt for generation.

RAG is the standard answer to three problems. It reduces hallucination because
the model is asked to answer from supplied passages. It keeps answers current
because updating the corpus is cheaper than retraining. And it enables
attribution, because every retrieved passage can be cited, letting a user verify
a claim against its source.

The quality of a RAG system is bounded by retrieval. If the right passage is
never retrieved, no amount of prompting will recover the correct answer, so
evaluation should measure retrieval (did we fetch the relevant chunk?) separately
from generation (did we use it faithfully?). Common failure modes are poor
chunking that splits an answer across boundaries, an embedding model that does
not capture the query's intent, and a context window packed with near-duplicate
passages that crowd out the one that matters.
