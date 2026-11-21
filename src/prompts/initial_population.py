"""Initial population generation (Member 2).

One LLM call per new prompt, each with a DIFFERENT style requirement, so the population is
diverse by construction. Seeds (e.g. the manual baselines) come first. Candidates are
cleaned, repaired, validated and de-duplicated (exact and near-duplicate).
"""
from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Optional

from .validation import check_prompt, normalize_prompt, prepare_candidate, similarity

log = logging.getLogger(__name__)

GENERATOR_SYSTEM = (
    "You are an expert prompt engineer. You write system prompts for a binary sentiment "
    "classifier. You output only the finished prompt text, with no commentary."
)

STYLE_HINTS = [
    "adopt the role of an expert film critic",
    "list a few concrete cues to look for, such as praise or complaint words, tone and intensity",
    "tell the classifier to handle negation, sarcasm and irony carefully",
    "tell the classifier to weigh mixed opinions and pick the dominant sentiment",
    "be extremely concise: one or two short sentences",
    "frame the task as deciding whether the writer would recommend the movie",
    "frame the task as deciding favorable versus unfavorable, then map to positive or negative",
    "mention that the text may be a short phrase or sentence fragment from a movie review",
    "use a formal, rule-based style with numbered rules",
    "address the classifier as a strict annotation tool that must follow the label format exactly",
]

GENERATOR_USER_TEMPLATE = """Write ONE system prompt for a classifier that labels a movie-review text as positive or negative.

Style requirement: {hint}.

Rules:
- The classifier must decide whether the text is positive or negative.
- The prompt must tell the classifier to answer with exactly one word: positive or negative, and nothing else.
- Do not include any example review and do not use placeholders such as [review].

Output only the prompt."""

NEAR_DUPLICATE = 0.90  # word-level similarity at or above this counts as a duplicate


def _is_duplicate(candidate: str, existing: list[str]) -> bool:
    return any(similarity(candidate, e) >= NEAR_DUPLICATE for e in existing)


def generate_initial_prompts(llm, n: int, seed_prompts=None, *,
                             rng: Optional[random.Random] = None,
                             max_attempts_per_prompt: int = 5) -> list[str]:
    """Return exactly ``n`` unique, valid prompts (seeds first, then LLM-generated).

    ``llm`` must provide ``generate_text(system, user, max_new_tokens=..., sample=True)``.
    Raises ValueError for n < 1 or an invalid seed, and RuntimeError if ``n`` unique prompts
    cannot be produced (this is a one-time setup step, so it fails loudly).
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    rng = rng or random.Random()

    population: list[str] = []
    for seed in (seed_prompts or []):
        clean = normalize_prompt(seed)
        problem = check_prompt(clean)
        if problem:
            raise ValueError(f"Invalid seed prompt ({problem}): {clean[:80]!r}")
        if not _is_duplicate(clean, population):
            population.append(clean)
        if len(population) == n:
            return population

    hints = STYLE_HINTS[:]
    rng.shuffle(hints)
    budget = max_attempts_per_prompt * (n - len(population))
    attempts = 0
    rejected: Counter = Counter()
    while len(population) < n and attempts < budget:
        hint = hints[attempts % len(hints)]      # a different style on every attempt
        attempts += 1
        try:
            raw = llm.generate_text(GENERATOR_SYSTEM, GENERATOR_USER_TEMPLATE.format(hint=hint),
                                    max_new_tokens=200, sample=True)
        except Exception as exc:  # noqa: BLE001 - a failed call just costs one attempt
            log.warning("Prompt generation call failed: %s", exc)
            continue
        candidate, problem = prepare_candidate(raw)
        if problem:
            rejected[problem.split(" (")[0]] += 1
            log.info("Rejected generated prompt (%s)", problem)
        elif _is_duplicate(candidate, population):
            rejected["near-duplicate of an existing prompt"] += 1
            log.info("Rejected generated prompt (duplicate)")
        else:
            population.append(candidate)

    if len(population) < n:
        raise RuntimeError(
            f"Requested {n} unique prompts but obtained {len(population)} after {attempts} "
            f"attempts. Rejections: {dict(rejected)}. Retry, add seed prompts, or lower n.")
    return population