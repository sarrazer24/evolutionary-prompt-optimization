"""Random prompt generation baseline for Experiment B (Member 4).

Generates ``n`` unique, valid sentiment-classification system prompts using the
same LLM generator interface as the Genetic Algorithm's initial population,
without evolutionary selection, crossover, mutation, or copying Member 2's
manual baseline prompts.
"""
from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Iterable, Optional

from src.prompts.initial_population import (
    GENERATOR_SYSTEM,
    GENERATOR_USER_TEMPLATE,
    NEAR_DUPLICATE,
    STYLE_HINTS,
    generate_initial_prompts,
)
from src.prompts.validation import normalize_prompt, prepare_candidate, similarity

log = logging.getLogger(__name__)

# Extended style hints used in Stage 2 after Stage 1's core hints so larger
# candidate budgets (e.g. n=50) stay diverse without near-duplicate saturation.
EXTENDED_STYLE_HINTS = [
    "write as a concise NLP annotation guideline with clear decision boundaries",
    "emphasize distinguishing genuine praise from polite or lukewarm criticism",
    "instruct the model to identify the net emotional polarity of the passage",
    "keep the prompt neutral, analytical, and focused on reviewer intent",
    "highlight mixed or bittersweet reviews and ask for the stronger overall lean",
    "frame the task as binary movie-sentiment tagging for a benchmark dataset",
    "remind the model to ignore plot descriptions and focus only on evaluative language",
    "use a single crisp imperative sentence followed by the required label format",
    "frame the decision around whether the reviewer liked or disliked the film overall",
    "instruct the classifier to evaluate adjectives, tone, and concluding remarks together",
]


def _is_near_duplicate(candidate: str, existing: Iterable[str],
                       threshold: float = NEAR_DUPLICATE) -> bool:
    return any(similarity(candidate, item) >= threshold for item in existing)


def generate_random_prompts(
    llm,
    n: int = 50,
    *,
    seed: int = 42,
    excluded_prompts: Optional[Iterable[str]] = None,
    max_attempts_per_prompt: int = 10,
) -> tuple[list[str], dict]:
    """Generate ``n`` unique, valid prompts without evolutionary optimization.

    Unlike GA initialization (which seeds Generation 0 with the 8 manual
    prompts), this baseline uses ``seed_prompts=None`` and filters out any
    prompt matching ``excluded_prompts`` (such as Experiment A's manual prompts)
    so Experiments A and B remain strictly disjoint.

    Returns ``(prompts, stats)``.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")

    rng = random.Random(seed)
    if hasattr(llm, "set_seed"):
        llm.set_seed(seed)

    excluded_set = {normalize_prompt(p) for p in (excluded_prompts or [])}
    candidates: list[str] = []
    rejected: Counter = Counter()
    stage2_calls = 0

    # Stage 1: Call Member 2's initial-population generator (capped at 30 so
    # near-duplicate saturation never discards a huge batch on large n).
    stage1_target = min(n, 30)
    try:
        initial_batch = generate_initial_prompts(
            llm=llm,
            n=stage1_target,
            seed_prompts=None,
            rng=rng,
            max_attempts_per_prompt=max_attempts_per_prompt,
        )
    except RuntimeError as exc:
        log.info("Stage 1 initial generator ended early: %s", exc)
        initial_batch = []

    for p in initial_batch:
        norm = normalize_prompt(p)
        if norm in excluded_set:
            rejected["matches_excluded_manual"] += 1
            continue
        if not _is_near_duplicate(norm, candidates):
            candidates.append(norm)

    # Stage 2: Top up remaining unique prompts using extended style hints.
    all_hints = STYLE_HINTS + EXTENDED_STYLE_HINTS
    rng.shuffle(all_hints)
    remaining_budget = max(60, max_attempts_per_prompt * (n - len(candidates)) * 2)
    attempt = 0

    while len(candidates) < n and attempt < remaining_budget:
        hint = all_hints[attempt % len(all_hints)]
        attempt += 1
        stage2_calls += 1
        temp = 0.8 if attempt <= len(all_hints) * 2 else 0.95
        try:
            raw = llm.generate_text(
                GENERATOR_SYSTEM,
                GENERATOR_USER_TEMPLATE.format(hint=hint),
                max_new_tokens=200,
                sample=True,
                temperature=temp,
            )
        except Exception as exc:  # noqa: BLE001
            rejected["llm_error"] += 1
            log.warning("Random prompt generation call failed: %s", exc)
            continue

        candidate, problem = prepare_candidate(raw)
        if problem:
            rejected[problem.split(" (")[0]] += 1
            continue

        norm = normalize_prompt(candidate)
        if norm in excluded_set:
            rejected["matches_excluded_manual"] += 1
            continue
        if _is_near_duplicate(norm, candidates):
            rejected["near_duplicate"] += 1
            continue

        candidates.append(norm)

    if len(candidates) < n:
        raise RuntimeError(
            f"Generated {len(candidates)}/{n} unique random prompts. "
            f"Rejections: {dict(rejected)}"
        )

    overlap_count = sum(1 for p in candidates if normalize_prompt(p) in excluded_set)
    stats = {
        "n_requested": n,
        "n_generated": len(candidates),
        "seed": seed,
        "manual_seeds_included": False,
        "manual_overlap_count": overlap_count,
        "stage1_count": len(initial_batch),
        "stage2_extra_calls": stage2_calls,
        "stage2_rejections": dict(rejected),
    }
    return candidates, stats
