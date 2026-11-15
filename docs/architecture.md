# Architecture overview

The project separates four concerns:

1. Data and evaluation: loads labeled examples and measures predictions.
2. LLM and prompts: loads one local classifier and creates/evolves textual prompts.
3. Genetic Algorithm: selects, crosses and mutates candidate prompts.
4. Experiments: compares baselines and reports performance/cost.

The classifier model is fixed during comparisons. The prompt is the optimized
artifact. Prompt-generation calls must not access held-out test examples or
labels. Only the final selected prompt is evaluated on the held-out test set.
