# Member 1: data module

Member 1 owns this directory. Suggested interface:

- `load_dataset(...)`
- `prepare_splits(...)`

Keep the final test split out of all prompt-generation and optimization steps.
Do not commit downloaded dataset files; use Hugging Face Datasets or Kaggle's
dataset cache and document the dataset version and split strategy.
