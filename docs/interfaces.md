# Shared interfaces

The authoritative description of the interfaces between members is
[`member2_handoff.md`](member2_handoff.md). It lists exact function signatures, return types,
failure behaviour and a minimal integration example.

Short version:

* **Member 1** owns `src/data/` and `src/evaluation/`. The GA scores a prompt with
  `evaluate_prompt(prompt, texts, labels, llm.predict_raw_batch, model_id=..., inference_config=..., cache=...)`
  and uses `result["metrics"]["f1_macro"]` as fitness.
* **Member 2** owns `src/llm/` and `src/prompts/`: `LLMInterface`, `MockLLM`, baseline prompts,
  `generate_initial_prompts`, `mutate_prompt`, `crossover_prompts`.
* **Member 3** owns `src/genetic_algorithm/` and only calls the functions above.
* The candidate prompt is the SYSTEM message; the review is a separate USER message.
* The held-out test set is never loaded during evolution or prompt selection.