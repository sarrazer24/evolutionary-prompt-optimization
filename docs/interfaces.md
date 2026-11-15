# Shared interfaces and integration contract

Agree on these interfaces before final integration.

## Member 2: LLM

`LLMInterface.load_model() -> LLMInterface`

`LLMInterface.predict_sentiment(prompt: str, text: str) -> str`

Returns one of: `positive`, `negative`, `invalid`.

`LLMInterface.predict_batch(prompt: str, texts: Iterable[str]) -> list[str]`

`LLMInterface.generate_text(system_prompt: str, user_prompt: str, max_new_tokens: int | None = None) -> str`

Candidate prompt is the system message; review text is a separate user message.

## Member 2: prompt operators

`generate_initial_prompts(llm, n, seed_prompts=None) -> list[str]`

`mutate_prompt(llm, prompt) -> str`

`crossover_prompts(llm, parent1, parent2) -> str`

## Member 1: evaluation

Recommended interface:

`evaluate_prompt(prompt, examples, labels, predict_fn) -> dict`

`predict_fn` can be a callable such as:

`lambda text: llm.predict_sentiment(prompt, text)`

The evaluator must not depend on a specific Hugging Face model class.

## Member 3: GA

The GA should depend on callable interfaces for evaluation, mutation and crossover.
It should not contain model-loading code.

## Member 4: experiments

Use the same classifier model and decoding settings for manual, random and GA
prompt comparisons. The final test set must remain untouched until final
evaluation.
