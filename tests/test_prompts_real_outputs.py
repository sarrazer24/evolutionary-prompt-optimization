"""Regression tests built from REAL Qwen2.5-1.5B-Instruct outputs seen on Kaggle (Step 4b)."""
import random

import pytest

from src.prompts.baseline_prompts import get_baseline_prompt_texts
from src.prompts.initial_population import generate_initial_prompts
from src.prompts.operators import crossover_prompts, mutate_prompt
from src.prompts.validation import OUTPUT_RULE, check_prompt, prepare_candidate

DIRECT = get_baseline_prompt_texts()[0]


class ScriptedLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate_text(self, system_prompt, user_prompt, max_new_tokens=None, **kwargs):
        self.calls.append(user_prompt)
        return self.outputs.pop(0)


# Real generations that forgot the "exactly one word" rule: they must be repaired, not rejected.
FORGOT_RULE = [
    "Please categorize the sentiment of the movie review as positive or negative.",
    "Identify positive sentiment by detecting words and phrases indicating enjoyment, approval, or "
    "recommendation. Identify negative sentiment by recognizing complaints, criticism, or dissatisfaction.",
    '"Classify the sentiment of the text as either positive or negative, considering negation, sarcasm, '
    'and irony appropriately."',
    "classify the sentiment of the text as positive or negative",
    "Decide whether the text is positive or negative. Answer with positive or negative.",
]


@pytest.mark.parametrize("raw", FORGOT_RULE)
def test_missing_output_rule_is_repaired(raw):
    candidate, problem = prepare_candidate(raw)
    assert problem is None and candidate.endswith(OUTPUT_RULE)
    assert check_prompt(candidate) is None


def test_prompt_without_both_labels_is_still_rejected():
    candidate, problem = prepare_candidate("Would you recommend this movie?")
    assert candidate is None and "both labels" in problem


def test_existing_output_rule_is_not_duplicated():
    raw = "Classify the sentiment of the movie review. Answer with exactly one word: positive or negative."
    candidate, _ = prepare_candidate(raw)
    assert candidate.count("exactly one word") == 1


def test_only_without_an_output_rule_no_longer_counts_as_one_word_instruction():
    assert check_prompt("Focus only on the wording of the review and decide positive or negative.") is not None


def test_meta_line_from_crossover_is_removed():
    raw = ("Combine the two system prompts into one new prompt:\n\nBe careful with negation and sarcasm. "
           "Decide whether the overall sentiment is positive or negative. Output exactly one lowercase word.")
    candidate, problem = prepare_candidate(raw)
    assert problem is None and not candidate.lower().startswith("combine")


def test_initial_population_succeeds_with_real_style_outputs():
    llm = ScriptedLLM(FORGOT_RULE + ["Would you recommend this movie?"])
    result = generate_initial_prompts(llm, 5, seed_prompts=[DIRECT], rng=random.Random(0))
    assert len(result) == 5 and result[0] == DIRECT
    assert all(check_prompt(p) is None for p in result)


def test_failure_message_lists_rejection_reasons():
    llm = ScriptedLLM(["Would you recommend this movie?"] * 20)
    with pytest.raises(RuntimeError, match="Rejections"):
        generate_initial_prompts(llm, 3, seed_prompts=[DIRECT], max_attempts_per_prompt=2)


def test_superficial_real_mutation_is_rejected_and_real_edit_accepted():
    superficial = ("Classify the sentiment of the movie review as positive or negative. "
                   "Answer with exactly one word: positive or negative.")        # similarity 0.88 to parent
    real = ("1. Read the movie review.\n2. Determine if the review expresses positive sentiment.\n"
            "3. If yes, classify as positive.\n4. If no, classify as negative.\n"
            "5. Answer with exactly one word: positive or negative.")
    llm = ScriptedLLM([superficial, real])
    child = mutate_prompt(llm, DIRECT, rng=random.Random(0))
    assert child.startswith("1. Read the movie review") and len(llm.calls) == 2