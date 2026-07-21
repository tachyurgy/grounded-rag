# Agents and tools

An agent is a language model given tools and a loop. Instead of answering in one
shot, it decides an action, observes the result, and repeats until it can answer.
Tools turn a text generator into a system that can search a knowledge base, run a
calculation, call an API, or query a database — grounding it in live actions
rather than only stored text.

The classic control loop is ReAct: the model interleaves reasoning ("thought")
with actions. Each turn it emits a structured action — a tool name and its input
— the executor runs the tool and appends the observation to the transcript, and
the loop continues until the model emits a final answer. Modern models expose
native tool-calling that returns a structured function call directly, which is
more reliable than parsing free text.

Robust agents need guardrails. Cap the number of steps so a confused agent cannot
loop forever. Validate tool inputs; an arithmetic tool should evaluate a
whitelisted expression grammar, never a raw eval of untrusted text. Make tool
failures observations the model can recover from — return "error: ..." as the
observation rather than crashing the loop. And keep the tool set small and
well-described, because every extra tool widens the space in which the model can
choose wrongly.

Agents trade latency and cost for capability, so reach for one only when a task
genuinely needs multiple steps or external actions. A single well-retrieved RAG
answer is cheaper and more predictable than an agent for a question that just
needs one lookup. When you do use an agent, retrieval is often its most valuable
tool: a "search the knowledge base" tool lets the agent ground each step in
evidence exactly as a RAG pipeline would.
