"""LLM-assisted mutation and crossover operators for textual prompts."""

OPERATOR_SYSTEM = (
    "You are an expert prompt engineer. You edit system instructions for "
    "binary sentiment classification. Preserve the task and output contract."
)

TASK_CONTRACT = """
The prompt must classify the overall sentiment of a review as positive or
negative. It must instruct the classifier to return exactly one lowercase
label, either positive or negative, with no explanation. Do not add examples
with labels unless explicitly requested.
"""


def _clean_prompt(text: str) -> str:
    text = (text or "").strip()
    # Remove common enclosing code fences and surrounding quotation marks.
    if text.startswith("```"):
        text = text.strip("`").strip()
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()
    return text.strip().strip('"').strip("'")


def _validate_prompt(prompt: str) -> str:
    clean = _clean_prompt(prompt)
    if len(clean) < 20:
        raise ValueError("Generated prompt is empty or too short.")
    lowered = clean.lower()
    if "positive" not in lowered or "negative" not in lowered:
        raise ValueError("Prompt must preserve both positive and negative labels.")
    return clean


def mutate_prompt(llm, prompt: str) -> str:
    """Create one meaningfully modified prompt while preserving the task."""
    user = f"""
Mutate the following candidate system prompt to create ONE improved or
meaningfully different candidate. Change wording, role, criteria, or structure,
but preserve the classification task and output contract.

{TASK_CONTRACT}

PARENT PROMPT:
{prompt}

Return only the complete mutated prompt. Do not explain your changes.
"""
    raw = llm.generate_text(OPERATOR_SYSTEM, user, max_new_tokens=220)
    return _validate_prompt(raw)


def crossover_prompts(llm, parent1: str, parent2: str) -> str:
    """Combine useful characteristics of two parents into one coherent child."""
    user = f"""
Create ONE coherent child system prompt by combining useful characteristics
of the two parent prompts. Do not concatenate them blindly. Keep the result
clear, concise, and suitable for binary sentiment classification.

{TASK_CONTRACT}

PARENT 1:
{parent1}

PARENT 2:
{parent2}

Return only the complete child prompt. Do not explain your changes.
"""
    raw = llm.generate_text(OPERATOR_SYSTEM, user, max_new_tokens=240)
    return _validate_prompt(raw)
