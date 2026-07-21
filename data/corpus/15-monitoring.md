# Monitoring RAG in production

An offline evaluation set tells you a system was good on the day you built it; monitoring
tells you whether it is still good on live traffic. Both are needed, because production
queries drift away from whatever you tested.

Log the full trace of every request: the query, the retrieved passage ids and their
scores, the generated answer, and the grounding verdict. This trace is what lets you
answer "why did it say that" after the fact, and it is the raw material for every other
signal below.

Watch retrieval health. Track the similarity score of the top hit, because a falling
average often means incoming questions have drifted outside what the corpus covers. Watch
the rate of "no relevant results" and the rate of abstentions; a rising abstention rate is
usually a coverage gap you can fix by adding documents, not a model regression.

Watch grounding and feedback. The share of answers that pass the grounding check is a
continuous faithfulness signal, and a drop flags a prompt or model problem. Capture
thumbs-up and thumbs-down from users and route the negatives into review, since real
failures cluster around topics your test set never imagined.

Close the loop. Sample traced requests into a growing regression set so yesterday's
production failure becomes tomorrow's test case, and alert on latency and cost per request
alongside quality. A monitored RAG system turns every live question into evidence, which is
the only way to keep an accountable answer accountable over time.
