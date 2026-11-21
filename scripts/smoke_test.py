"""Real-model smoke test. Run from the repository root:

    python -m scripts.smoke_test

Needs a downloaded model (first run) - best on a Kaggle GPU; on a CPU-only PC it works but is slow.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm import LLMInterface  # noqa: E402
from src.prompts.baseline_prompts import get_baseline_prompts  # noqa: E402


def main():
    llm = LLMInterface().load_model()
    print(llm.info())
    prompt = get_baseline_prompts()[0][1]
    reviews = [
        "A wonderfully acted and moving film.",
        "A dull, poorly written movie that wastes its cast.",
    ]
    raw = llm.predict_raw_batch(prompt, reviews)
    labels = llm.predict_batch(prompt, reviews)
    for review, r, label in zip(reviews, raw, labels):
        print(f"Review: {review}\n  raw={r!r}  label={label}\n")


if __name__ == "__main__":
    main()