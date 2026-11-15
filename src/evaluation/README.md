# Member 1: evaluation module

Suggested interface:

`evaluate_prompt(prompt, examples, labels, predict_fn) -> dict`

The evaluator should receive a prediction callable, rather than importing
`src.llm.model_interface`. This keeps evaluation independent of the model.

Primary fitness metric: macro F1. Also report accuracy, precision, recall,
confusion matrix and invalid prediction count. Define how `invalid` outputs
are scored consistently before experiments begin.
