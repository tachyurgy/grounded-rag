# Query routing

Not every question should hit the same retriever, and not every question needs
retrieval at all. A router inspects the incoming query and dispatches it to the right
path, which improves both accuracy and cost.

The simplest router is a classifier. A small model or a few keyword rules decide
whether a query is a factual lookup that needs the knowledge base, a chit-chat turn
that needs no retrieval, or an operation that needs a tool such as a calculator or a
live API. Skipping retrieval when it is not needed removes irrelevant context that
would otherwise dilute the answer.

Multi-index routing picks among several corpora. If documents are partitioned by
topic, product, or tenant, the router selects the index whose description best matches
the query, so retrieval searches only relevant material. This scales better than one
giant index and enforces access boundaries between tenants.

Routing pairs naturally with query decomposition. A complex question that spans two
subjects can be split into sub-questions, each routed and retrieved independently, with
the answers composed at the end. Decomposition rescues multi-hop questions that a single
retrieval would answer only partially.

Keep the router cheap and observable. It runs on every request, so a heavyweight router
erases the savings it was meant to create; log its decisions so a misroute is easy to
spot. A good router is invisible when it works and the first thing you check when an
answer is confidently about the wrong document.
