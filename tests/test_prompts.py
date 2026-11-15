from src.llm.model_interface import LLMInterface
from src.prompts.baseline_prompts import get_baseline_prompts
from src.prompts.initial_population import generate_initial_prompts
from src.prompts.operators import mutate_prompt, crossover_prompts


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate_text(self, system_prompt, user_prompt, max_new_tokens=None):
        self.calls.append((system_prompt, user_prompt))
        return self.outputs.pop(0)


def test_sentiment_parser():
    assert LLMInterface.parse_sentiment("positive") == "positive"
    assert LLMInterface.parse_sentiment("The answer is negative.") == "negative"
    assert LLMInterface.parse_sentiment("positive and negative") == "invalid"
    assert LLMInterface.parse_sentiment("uncertain") == "invalid"


def test_baselines_have_both_labels():
    prompts = get_baseline_prompts()
    assert len(prompts) >= 5
    assert all("positive" in p.lower() and "negative" in p.lower() for _, p in prompts)


def test_initial_population_uses_seeds_and_generated_prompts():
    fake = FakeLLM(["1. First generated prompt: positive or negative.\n2. Second generated prompt: positive or negative."])
    result = generate_initial_prompts(fake, 3, seed_prompts=["Seed prompt: positive or negative."])
    assert len(result) == 3
    assert len(set(x.casefold() for x in result)) == 3


def test_mutation_and_crossover_preserve_labels():
    fake = FakeLLM([
        "Carefully classify the review and return exactly positive or negative.",
        "Determine overall polarity and output only positive or negative."
    ])
    mutated = mutate_prompt(fake, "Classify and return positive or negative.")
    child = crossover_prompts(
        fake,
        "Classify and return positive or negative.",
        "Determine sentiment; output positive or negative."
    )
    assert "positive" in mutated.lower() and "negative" in mutated.lower()
    assert "positive" in child.lower() and "negative" in child.lower()
