"""Human-written baseline prompts for binary SST-2 sentiment classification."""

BASELINE_PROMPTS = [
    (
        "baseline_direct",
        "Classify the sentiment of the review. Return exactly one word: "
        "positive or negative. Do not provide an explanation."
    ),
    (
        "baseline_expert",
        "You are a careful sentiment-analysis classifier. Determine whether "
        "the review expresses an overall positive or negative opinion. "
        "Return exactly one word: positive or negative."
    ),
    (
        "baseline_polarity",
        "Identify the overall polarity expressed by the reviewer. Consider "
        "the meaning of the complete review rather than isolated words. "
        "Output only positive or negative."
    ),
    (
        "baseline_context",
        "Read the entire review and classify the writer's overall sentiment. "
        "Use the context and intended meaning. Your answer must be exactly "
        "one of these labels: positive, negative."
    ),
    (
        "baseline_concise",
        "Determine whether this review is favorable or unfavorable. "
        "Map favorable to positive and unfavorable to negative. "
        "Respond with only the corresponding label: positive or negative."
    ),
    (
        "baseline_balanced",
        "Analyze the review without assuming that strong words always indicate "
        "the overall sentiment. Choose the dominant sentiment of the complete "
        "text. Return only positive or negative."
    ),
    (
        "baseline_instructional",
        "Read the review carefully. Decide whether its overall opinion is "
        "positive or negative. Do not summarize the review. Do not add reasons. "
        "Return exactly one label: positive or negative."
    ),
    (
        "baseline_output",
        "Perform binary sentiment classification on the supplied review. "
        "The only permitted outputs are the lowercase labels positive and "
        "negative. Output one label and nothing else."
    ),
]

def get_baseline_prompts():
    """Return a copy so callers cannot mutate the module-level list."""
    return list(BASELINE_PROMPTS)
