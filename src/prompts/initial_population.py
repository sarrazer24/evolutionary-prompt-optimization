"""Generate diverse initial candidate instructions using an LLM adapter."""
from __future__ import annotations

GENERATOR_SYSTEM = (
    "You are an expert prompt engineer. Generate clear, distinct system "
    "instructions for binary sentiment classification."
)

GENERATOR_USER_TEMPLATE = """
Create {n} diverse candidate SYSTEM PROMPTS for classifying movie-review
sentiment as either positive or negative.

Each candidate must:
- Instruct the classifier to determine the overall sentiment of a review.
- Require exactly one output label: positive or negative.
- Require no explanation or extra text in the classification output.
- Be a complete system instruction.
- Not include a specific review or its answer.

Avoid near-duplicates. Vary the instruction strategy, not the task.

Return only a numbered list, one prompt per item.
"""


def _extract_numbered_items(text: str) -> list[str]:
    """Conservative parser for numbered or bullet-list model output."""
    items = []
    current = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        import re
        match = re.match(r"^(?:\d+[\).\:]|[-*])\s*(.*)$", stripped)
        if match:
            if current:
                items.append(" ".join(current).strip())
            current = [match.group(1).strip()] if match.group(1).strip() else []
        elif current:
            current.append(stripped)
    if current:
        items.append(" ".join(current).strip())
    return [x for x in items if x]


def generate_initial_prompts(llm, n: int, seed_prompts=None) -> list[str]:
    """Generate prompts; `llm` must expose generate_text(system_prompt,user_prompt)."""
    if n < 1:
        raise ValueError("n must be at least 1.")

    seeds = list(seed_prompts or [])
    request_n = max(0, n - len(seeds))
    generated = []

    if request_n:
        raw = llm.generate_text(
            GENERATOR_SYSTEM,
            GENERATOR_USER_TEMPLATE.format(n=request_n),
            max_new_tokens=min(512, 100 * request_n),
        )
        generated = _extract_numbered_items(raw)

    combined = []
    seen = set()
    for prompt in seeds + generated:
        clean = " ".join(str(prompt).split())
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            combined.append(clean)
        if len(combined) == n:
            break

    if len(combined) < n:
        raise RuntimeError(
            f"Requested {n} unique prompts but obtained {len(combined)}. "
            "Retry generation or provide more seed prompts."
        )
    return combined
