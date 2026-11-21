"""CPU-only tests for src/prompts (baselines, validation, initial population, operators).

The LLM is replaced by ScriptedLLM, which returns pre-written outputs, so no model is needed.
"""
import random
from itertools import combinations

import pytest

from src.prompts.baseline_prompts import get_baseline_prompt_texts, get_baseline_prompts
from src.prompts.initial_population import generate_initial_prompts
from src.prompts.operators import crossover_prompts, mutate_prompt
from src.prompts.validation import check_prompt, clean_generated_prompt, normalize_prompt, similarity

A = "You are a film critic. Decide whether the review is positive or negative. Answer with only that one word."
B = "Judge the tone of the text carefully and output exactly one label: positive or negative, nothing else."
GOOD_CHILD = ("Act as a strict annotator. Read the whole text, weigh praise against criticism, and "
              "reply with exactly one word: positive or negative.")


class ScriptedLLM:
    """Returns the scripted outputs in order; an Exception instance in the script is raised."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate_text(self, system_prompt, user_prompt, max_new_tokens=None, **kwargs):
        self.calls.append({"system": system_prompt, "user": user_prompt,
                           "max_new_tokens": max_new_tokens, **kwargs})
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


# ------------------------------------------------------------------ baselines
def test_baselines_are_valid_distinct_and_review_free():
    prompts = get_baseline_prompts()
    assert len(prompts) >= 5
    assert len({name for name, _ in prompts}) == len(prompts)
    for name, p in prompts:
        assert check_prompt(p) is None, name
    for (n1, p1), (n2, p2) in combinations(prompts, 2):
        assert similarity(p1, p2) < 0.6, (n1, n2)      # genuinely different, not paraphrases
    assert get_baseline_prompt_texts() == [p for _, p in prompts]


def test_get_baseline_prompts_returns_a_copy():
    get_baseline_prompts().clear()
    assert len(get_baseline_prompts()) >= 5


# ------------------------------------------------------------------ validation / cleaning
@pytest.mark.parametrize("raw", [
    f"```\n{A}\n```",
    f'"{A}"',
    f"Here is the new prompt:\n\n{A}",
    f"Mutated prompt: {A}",
    f"**Prompt:** {A}",
    f"{A}\n\nNote: I added a role.",
    f"  {A.replace(' ', '   ')}\n",
])
def test_clean_generated_prompt_strips_llm_packaging(raw):
    assert clean_generated_prompt(raw) == A


@pytest.mark.parametrize("prompt,fragment", [
    ("positive or negative", "too short"),
    ("x" * 900 + " positive negative only", "too long"),
    ("Decide whether the review is good or bad. Answer with only one word.", "both labels"),
    ("Decide if the review is positive or negative and explain your reasoning at length.", "exactly one"),
    ("Review: [review]. Is it positive or negative? Answer with only one word.", "placeholder"),
    (None, "not a string"),
])
def test_check_prompt_rejects_bad_prompts(prompt, fragment):
    assert fragment in check_prompt(prompt)


def test_check_prompt_accepts_good_prompt():
    assert check_prompt(A) is None and check_prompt(B) is None


# ------------------------------------------------------------------ initial population
def test_initial_population_seeds_first_then_generated_unique():
    llm = ScriptedLLM([GOOD_CHILD, B])
    result = generate_initial_prompts(llm, 3, seed_prompts=[A], rng=random.Random(0))
    assert result == [A, GOOD_CHILD, B]
    assert all(c["sample"] is True for c in llm.calls)      # generation is sampled


def test_initial_population_uses_no_llm_when_seeds_suffice():
    llm = ScriptedLLM([])
    assert generate_initial_prompts(llm, 2, seed_prompts=[A, B, GOOD_CHILD]) == [A, B]
    assert llm.calls == []


def test_initial_population_rejects_invalid_duplicate_and_failed_generations():
    llm = ScriptedLLM(["too short", A, RuntimeError("cuda"), GOOD_CHILD, B])
    result = generate_initial_prompts(llm, 3, seed_prompts=[A], rng=random.Random(1))
    assert result == [A, GOOD_CHILD, B]
    assert len(llm.calls) == 5


def test_initial_population_uses_different_style_hints():
    llm = ScriptedLLM([GOOD_CHILD, B, "x"])
    generate_initial_prompts(llm, 3, seed_prompts=[A], rng=random.Random(0))
    hints = {c["user"].split("Style requirement: ")[1].split(".\n")[0] for c in llm.calls}
    assert len(hints) == len(llm.calls)


def test_initial_population_raises_when_impossible():
    llm = ScriptedLLM(["bad"] * 50)
    with pytest.raises(RuntimeError, match="Requested 3 unique prompts"):
        generate_initial_prompts(llm, 3, seed_prompts=[A], max_attempts_per_prompt=2)


def test_initial_population_validates_arguments():
    with pytest.raises(ValueError):
        generate_initial_prompts(ScriptedLLM([]), 0)
    with pytest.raises(ValueError, match="Invalid seed"):
        generate_initial_prompts(ScriptedLLM([]), 2, seed_prompts=["not a valid prompt"])


# ------------------------------------------------------------------ mutation
def test_mutation_returns_clean_valid_child_and_samples():
    llm = ScriptedLLM([f"Here is the new prompt:\n{GOOD_CHILD}"])
    child = mutate_prompt(llm, A, rng=random.Random(0))
    assert child == GOOD_CHILD and check_prompt(child) is None
    call = llm.calls[0]
    assert call["sample"] is True and A in call["user"]


def test_mutation_retries_after_invalid_superficial_and_failed_outputs():
    near_copy = A.replace("film critic", "movie critic")
    llm = ScriptedLLM(["nonsense", near_copy, RuntimeError("boom"), GOOD_CHILD])
    assert mutate_prompt(llm, A, max_attempts=4, rng=random.Random(0)) == GOOD_CHILD
    assert len(llm.calls) == 4


def test_mutation_falls_back_to_parent_unchanged():
    llm = ScriptedLLM(["bad", "bad", "bad"])
    assert mutate_prompt(llm, A) == A


def test_mutation_uses_requested_strategy_and_is_reproducible():
    llm = ScriptedLLM([GOOD_CHILD])
    mutate_prompt(llm, A, strategy="make it shorter")
    assert "Required change: make it shorter." in llm.calls[0]["user"]
    s1, s2 = ScriptedLLM([GOOD_CHILD]), ScriptedLLM([GOOD_CHILD])
    mutate_prompt(s1, A, rng=random.Random(7))
    mutate_prompt(s2, A, rng=random.Random(7))
    assert s1.calls[0]["user"] == s2.calls[0]["user"]


# ------------------------------------------------------------------ crossover
def test_crossover_returns_valid_child_and_shows_both_parents():
    llm = ScriptedLLM([GOOD_CHILD])
    child = crossover_prompts(llm, A, B)
    assert child == GOOD_CHILD
    user = llm.calls[0]["user"]
    assert A in user and B in user and llm.calls[0]["sample"] is True


def test_crossover_rejects_copy_of_a_parent_then_retries():
    llm = ScriptedLLM([B, GOOD_CHILD])
    assert crossover_prompts(llm, A, B) == GOOD_CHILD
    assert len(llm.calls) == 2


def test_crossover_identical_parents_makes_no_llm_call():
    llm = ScriptedLLM([])
    assert crossover_prompts(llm, A, A) == A and llm.calls == []


def test_crossover_falls_back_to_parent1():
    llm = ScriptedLLM(["bad", "bad", "bad"])
    assert crossover_prompts(llm, A, B) == A


def test_operators_keep_labels_and_single_line():
    llm = ScriptedLLM([GOOD_CHILD, GOOD_CHILD.replace("Act as", "Behave like")])
    for child in (mutate_prompt(llm, A), crossover_prompts(llm, A, B)):
        assert "positive" in child.lower() and "negative" in child.lower()
        assert child == normalize_prompt(child)
        