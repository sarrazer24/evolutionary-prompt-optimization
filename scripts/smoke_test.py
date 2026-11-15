"""Run a small real-model test after enabling Kaggle GPU and Internet."""
from src.llm.model_interface import LLMInterface
from src.prompts.baseline_prompts import get_baseline_prompts


def main():
    llm = LLMInterface().load_model()
    prompts = get_baseline_prompts()
    prompt = prompts[0][1]
    examples = [
        "A wonderfully acted and moving film.",
        "A dull, poorly written movie that wastes its cast.",
    ]
    for review in examples:
        prediction = llm.predict_sentiment(prompt, review)
        print(f"Review: {review}\nPrediction: {prediction}\n")


if __name__ == "__main__":
    main()
