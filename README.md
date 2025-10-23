# safe-ai-poc
Proofs of concept for Safe AI.

## openai: Open AI on Azure with Guardrails

This is a mini project to test Safe AI using an Open AI on Azure deployment, wrapped
with Guardrails AI.

I discovered some built-in guards against bias running prompts on Open AI on Azure.
This is great news! Typically controls in place would mean having a second system
check this, so I introduced Guardrails as an option.

There are two python classes.

- AureOpenAIClient.py is an attempt at a robust prompt client with retries, optional
use of GPU for ML processing, and optionality of running the program without Guardrails.
- AureOpenAIClientSimple.py was for a POC and blog article. Very simple, happy path,
gets it running and uses some provcative yet PG-rated prompts to see some failures.
