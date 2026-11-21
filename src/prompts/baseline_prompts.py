"""Manually designed baseline prompts for binary SST-2 sentiment classification (Member 2).

Each prompt is a SYSTEM instruction; the review is sent separately as the user message.
All eight solve the same task and use the same output contract (one word: positive or
negative) but differ in strategy: plain instruction, persona, explicit cues, negation and
sarcasm handling, recommendation framing, fragment awareness, few-shot examples, rules.
The few-shot examples are hand-written and are NOT taken from SST-2.
"""

BASELINE_PROMPTS = [
    (
        "baseline_direct",
        "Classify the sentiment of the movie review as positive or negative. "
        "Reply with exactly one word: positive or negative. Do not explain."
    ),
    (
        "baseline_expert",
        "You are an expert film critic who reads reviews for a living. Decide whether the "
        "reviewer's overall opinion of the movie is positive or negative. "
        "Answer with only that one word."
    ),
    (
        "baseline_criteria",
        "Look for praise, enthusiasm and approval (positive) versus complaints, disappointment "
        "and criticism (negative). Weigh the overall tone of the text, then output exactly one "
        "word: positive or negative."
    ),
    (
        "baseline_negation",
        "Be careful with negation, sarcasm and irony: a phrase such as 'not bad' or 'hardly a "
        "masterpiece' does not mean what its individual words suggest. Judge the intended "
        "sentiment of the whole text and answer with exactly one word: positive or negative."
    ),
    (
        "baseline_recommend",
        "Would the writer recommend this movie? If the text is favorable, answer positive; if it "
        "is unfavorable, answer negative. Respond with that single word and nothing else."
    ),
    (
        "baseline_fragment",
        "The input may be a short phrase or a sentence fragment taken from a movie review, so "
        "judge only the sentiment it expresses. Output exactly one label, positive or negative, "
        "with no punctuation or commentary."
    ),
    (
        "baseline_fewshot",
        "Classify movie-review sentiment. Examples: 'a gripping, beautifully acted drama' is "
        "positive. 'dull, predictable and far too long' is negative. 'an uneven script that "
        "wastes a talented cast' is negative. Now classify the new text and answer with exactly "
        "one word: positive or negative."
    ),
    (
        "baseline_rules",
        "Follow these rules. 1) Read the whole text. 2) Decide whether its overall sentiment is "
        "positive or negative; if opinions are mixed, choose the dominant one. 3) Output only the "
        "label, as exactly one lowercase word: positive or negative."
    ),
]


def get_baseline_prompts() -> list[tuple[str, str]]:
    """Return a copy of the (name, prompt) pairs so callers cannot mutate the module list."""
    return list(BASELINE_PROMPTS)


def get_baseline_prompt_texts() -> list[str]:
    """Just the prompt strings, e.g. as seeds for ``generate_initial_prompts``."""
    return [prompt for _, prompt in BASELINE_PROMPTS]