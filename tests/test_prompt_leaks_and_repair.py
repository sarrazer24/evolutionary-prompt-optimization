"""Regression tests from the second Kaggle run (N=1000): echoed instructions and duplicated rules."""
import pytest

from src.prompts.baseline_prompts import get_baseline_prompt_texts
from src.prompts.validation import OUTPUT_RULE, check_prompt, prepare_candidate

# Real crossover output that echoed the instructions instead of being a prompt.
LEAKED = ("Take a distinctive idea from PROMPT A and PROMPT B, merge them smoothly, and combine them into a "
          "new prompt that tells the classifier to judge the sentiment of a movie-review text by using "
          "exactly one word: positive or negative, without any punctuation or additional commentary.")


def test_echoed_instructions_are_rejected():
    candidate, problem = prepare_candidate(LEAKED)
    assert candidate is None and "echoes" in problem


def test_all_baselines_are_free_of_leak_patterns():
    for p in get_baseline_prompt_texts():
        assert check_prompt(p) is None


@pytest.mark.parametrize("raw,expected", [
    # weaker closing sentence is replaced, not duplicated
    ("Decide if the text is positive or negative. Answer with **positive** or **negative**.",
     "Decide if the text is positive or negative. " + OUTPUT_RULE),
    ("Is the review positive or negative? Answer with positive or negative.",
     "Is the review positive or negative? " + OUTPUT_RULE),
    # dangling colon becomes a full stop
    ("Please classify the following text as positive or negative:",
     "Please classify the following text as positive or negative. " + OUTPUT_RULE),
    # nothing to replace: the rule is simply appended
    ("Please categorize the sentiment of the movie review as positive or negative.",
     "Please categorize the sentiment of the movie review as positive or negative. " + OUTPUT_RULE),
])
def test_repair_replaces_weak_output_sentence_instead_of_duplicating(raw, expected):
    candidate, problem = prepare_candidate(raw)
    assert problem is None and candidate == expected
    assert candidate.count("Answer with") <= 1


def test_prompt_that_already_has_the_rule_is_untouched():
    raw = "Given a movie review text, classify it as positive or negative. Answer with exactly one word: positive or negative."
    assert prepare_candidate(raw) == (raw, None)