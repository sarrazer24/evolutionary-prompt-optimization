"""LLM-assisted mutation and crossover for system prompts (Member 2).

Both operators sample from the LLM, validate the result, retry a few times, and if every
attempt fails they return the (first) parent UNCHANGED instead of raising, so a GA run is
never aborted by one bad generation. The GA can detect that with ``child == parent``.
"""
from __future__ import annotations

import logging
import random
from typing import Optional

from .validation import check_prompt, clean_generated_prompt, normalize_prompt, similarity

log = logging.getLogger(__name__)

OPERATOR_SYSTEM = (
    "You are an expert prompt engineer. You rewrite system prompts for a binary sentiment "
    "classifier. You output only the finished prompt text, with no commentary."
)

MUTATION_STRATEGIES = [
    "change the persona or role the classifier plays",
    "add a specific decision criterion, such as how to treat negation, sarcasm or mixed opinions",
    "restructure the prompt into short numbered steps ending with the output rule",
    "make it much shorter and more direct while keeping the output rule",
    "reframe the task in different terms (for example recommend versus not recommend, or favorable versus unfavorable) while still answering positive or negative",
    "add guidance about what evidence to prioritize, such as the overall tone rather than single words",
    "mention that the text may be a short phrase or sentence fragment from a movie review",
    "change the style (formal, plain, imperative, or checklist) and reorder the instructions",
]

_RULES = """Rules:
- The new prompt must still make the classifier decide whether a movie-review text is positive or negative.
- It must still tell the classifier to answer with exactly one word: positive or negative, and nothing else.
- Do not include any example review and do not use placeholders such as [review]."""

MUTATE_TEMPLATE = """Rewrite the system prompt below into a NEW version.

Required change: {strategy}.
Do not just swap synonyms; make the required change clearly visible.

{rules}

CURRENT PROMPT:
{prompt}

Output only the new prompt."""

CROSSOVER_TEMPLATE = """Combine the two system prompts below into ONE new prompt.

Take a distinctive idea from PROMPT A (for example its role, criteria or wording style) and a distinctive idea from PROMPT B, and merge them smoothly. Do not copy one prompt and do not paste them together.

{rules}

PROMPT A:
{a}

PROMPT B:
{b}

Output only the new prompt."""

MUTATION_MAX_SIMILARITY = 0.90   # child at least this similar to its parent = superficial edit
CROSSOVER_MAX_SIMILARITY = 0.95  # child at least this similar to a parent = just a copy


def _attempt(llm, user_prompt: str, max_new_tokens: int) -> Optional[str]:
    """One sampled generation, cleaned and validated; None if it failed for any reason."""
    try:
        raw = llm.generate_text(OPERATOR_SYSTEM, user_prompt, max_new_tokens=max_new_tokens,
                                sample=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("Operator generation failed: %s", exc)
        return None
    candidate = clean_generated_prompt(raw)
    problem = check_prompt(candidate)
    if problem:
        log.info("Rejected operator output (%s)", problem)
        return None
    return candidate


def mutate_prompt(llm, prompt: str, *, rng: Optional[random.Random] = None,
                  strategy: Optional[str] = None, max_attempts: int = 3) -> str:
    """Return one mutated prompt (single-line, valid, clearly different from ``prompt``).

    Returns ``prompt`` unchanged if no valid, sufficiently different child is produced.
    """
    parent = normalize_prompt(prompt)
    rng = rng or random.Random()
    for _ in range(max_attempts):
        chosen = strategy or rng.choice(MUTATION_STRATEGIES)
        user = MUTATE_TEMPLATE.format(strategy=chosen, rules=_RULES, prompt=parent)
        child = _attempt(llm, user, max_new_tokens=220)
        if child is None:
            continue
        if similarity(child, parent) >= MUTATION_MAX_SIMILARITY:
            log.info("Rejected mutation (too similar to parent)")
            continue
        return child
    log.warning("mutate_prompt: no valid mutation after %d attempts; returning parent", max_attempts)
    return prompt


def crossover_prompts(llm, parent1: str, parent2: str, *, max_attempts: int = 3) -> str:
    """Return one child combining ideas from both parents (single-line, valid).

    Returns ``parent1`` unchanged if the parents are identical or no valid child that is
    different from both parents is produced.
    """
    p1, p2 = normalize_prompt(parent1), normalize_prompt(parent2)
    if similarity(p1, p2) >= CROSSOVER_MAX_SIMILARITY:
        return parent1                      # nothing to combine; no LLM call wasted
    user = CROSSOVER_TEMPLATE.format(rules=_RULES, a=p1, b=p2)
    for _ in range(max_attempts):
        child = _attempt(llm, user, max_new_tokens=240)
        if child is None:
            continue
        if max(similarity(child, p1), similarity(child, p2)) >= CROSSOVER_MAX_SIMILARITY:
            log.info("Rejected crossover (copies a parent)")
            continue
        return child
    log.warning("crossover_prompts: no valid child after %d attempts; returning parent1", max_attempts)
    return parent1