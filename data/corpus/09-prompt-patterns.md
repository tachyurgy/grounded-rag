# Prompt patterns for RAG

The generation prompt is where retrieved context becomes a grounded answer, and
small wording choices change faithfulness more than model size does.

Structure the prompt in a fixed order: a system instruction that sets the rules, the
numbered context passages, then the user question last. Number each passage so the
model can cite it, and label its source so citations are traceable. Putting the
question after the context keeps the instruction fresh in the model's attention.

State the grounding contract explicitly. Tell the model to answer only from the
provided passages, to attach the source number in brackets to every claim, and to
reply that the passages do not contain the answer when they do not. Granting explicit
permission to abstain is the single most effective anti-fabrication instruction,
because a model with no permission to say "I don't know" will invent something.

Keep formatting demands light. Asking for flowing prose with inline citations yields
answers that are easy to verify sentence by sentence; asking for elaborate headed
sections tempts the model to write uncited transitional lines. Give one short
worked example of the citation style if the model is inconsistent, but avoid long
few-shot blocks that eat the context budget you need for retrieved passages.

Guard the boundaries. Wrap user-supplied text so an instruction hidden inside a
document cannot override the system prompt, and never let retrieved content silently
change the model's role. A prompt that is explicit about sources, citations, and
abstention turns a fluent generator into an auditable one.
