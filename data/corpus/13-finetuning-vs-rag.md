# Fine-tuning versus RAG

Fine-tuning and retrieval solve different problems, and the common mistake is reaching
for one when the job needs the other. A useful rule: RAG changes what the model knows,
fine-tuning changes how the model behaves.

Use RAG when the need is knowledge — facts that change, that are private to your
organization, or that are too numerous to bake into weights. Retrieval keeps that
knowledge in an external store you can update cheaply, and it gives you citations, so the
answer is attributable. Retraining to add a single new document would be absurd; adding it
to the index takes seconds.

Use fine-tuning when the need is form — a consistent tone, a strict output schema, a
domain's vocabulary and phrasing, or a task pattern the base model performs unreliably.
Fine-tuning teaches the model to respond a certain way, but it does not reliably teach new
facts, and facts learned in weights cannot be cited or easily corrected when they go stale.

The two compose well. A model fine-tuned to follow your citation format and house style,
answering over a RAG pipeline that supplies current grounded facts, gets the behaviour
from tuning and the knowledge from retrieval. Start with RAG and prompting, because they
are faster and cheaper to iterate, and add fine-tuning only when you have evidence that
behaviour, not knowledge, is the bottleneck. Reaching for fine-tuning to fix a knowledge
gap is the classic expensive detour.
