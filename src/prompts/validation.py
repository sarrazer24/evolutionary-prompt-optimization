"""Shared prompt cleaning / validation / similarity helpers (Member 2).

Every prompt that enters or leaves a prompt operator goes through these functions, so the
Genetic Algorithm only ever sees clean, single-line, valid, comparable prompts.
"""
from __future__ import annotations

import difflib
import re

MIN_CHARS = 30
MAX_CHARS = 800

# The classifier must be told to give ONE label and nothing else, because Member 1's strict
# parser marks every other answer as invalid.
_SINGLE_LABEL = re.compile(
    r"\b(exactly one|one word|single word|one label|single label|nothing else|just one|"
    r"only (?:the |that |one |a )?(?:label|word|answer|response))\b", re.I)
_NEEDS_RULE = "must tell the classifier to answer with exactly one word/label"
# The output contract is NOT something the GA should evolve: it is part of the task definition
# (Member 1's strict parser needs a bare label). If a generated prompt forgot it, we append it.
OUTPUT_RULE = "Answer with exactly one word: positive or negative."
# The generator sometimes echoes the instructions it was given ("merge them into a new prompt").
_META_LEAK = re.compile(
    r"\bprompt\s+(?:a|b|1|2)\b|\b(?:new|combined|merged|child|parent) prompts?\b|"
    r"\bthe two (?:system )?prompts?\b|\b(?:merge|combine) them\b|\bdistinctive idea\b", re.I)
# A weaker output sentence the repair step should REPLACE instead of duplicating.
_OUTPUT_SENTENCE = re.compile(r"^\W*(?:answer|respond|reply|output|return)\b.*\b(?:positive|negative)\b", re.I)
# Unfilled placeholders: the review is supplied separately as the user message.
_PLACEHOLDER = re.compile(r"\{[^}]*\}|\[(?:review|text|input|sentence)[^\]]*\]|<(?:review|text|input)[^>]*>", re.I)
_CHATTER = re.compile(r"^(here|sure|certainly|okay|ok|of course)\b.*:\s*$", re.I)
_META_LINE = re.compile(r"^.*\bprompts?\b.*:\s*$", re.I)   # e.g. 'Combine the two system prompts into one new prompt:'
_TRAILING_NOTE = re.compile(r"^\W*(note|notes|explanation|changes|this prompt|i (changed|made|added|kept)|"
                            r"in this)\b", re.I)
_LEADING_LABEL = re.compile(
    r"^\W*(?:(?:mutated|child|new|improved|rewritten|system)\s+)?prompt\s*[a-z0-9]?\s*[:\-]\s*\**\s*", re.I)


def normalize_prompt(text) -> str:
    """Collapse all whitespace to single spaces (stable cache keys, easy de-duplication)."""
    return " ".join(str(text).split())


def clean_generated_prompt(raw) -> str:
    """Strip the usual LLM packaging: code fences, 'Here is ...:' lines, 'Prompt:' labels,
    surrounding quotes and trailing notes. Returns a single-line string (may be invalid)."""
    text = (raw or "").strip()
    fence = re.match(r"^```[A-Za-z]*\n?(.*?)\n?```$", text, re.S)
    if fence:
        text = fence.group(1).strip()
    lines = text.splitlines()
    if len(lines) > 1 and (_CHATTER.match(lines[0].strip()) or _META_LINE.match(lines[0].strip())):
        text = "\n".join(lines[1:]).strip()
    kept = []
    for i, para in enumerate(re.split(r"\n\s*\n", text)):
        if i > 0 and _TRAILING_NOTE.match(para.strip()):
            break
        kept.append(para)
    text = "\n\n".join(kept).strip()
    text = _LEADING_LABEL.sub("", text).strip()
    if len(text) >= 2 and text[0] in "\"'\u201c\u2018" and text[-1] in "\"'\u201d\u2019":
        text = text[1:-1]
    return normalize_prompt(text)


def check_prompt(prompt) -> str | None:
    """Return None if the prompt is a valid candidate, otherwise a short reason."""
    if not isinstance(prompt, str):
        return "not a string"
    n = len(prompt.strip())
    if n < MIN_CHARS:
        return f"too short ({n} chars, minimum {MIN_CHARS})"
    if n > MAX_CHARS:
        return f"too long ({n} chars, maximum {MAX_CHARS})"
    if not re.search(r"\bpositive\b", prompt, re.I) or not re.search(r"\bnegative\b", prompt, re.I):
        return "must mention both labels: positive and negative"
    if not _SINGLE_LABEL.search(prompt):
        return _NEEDS_RULE
    if _PLACEHOLDER.search(prompt):
        return "contains a placeholder; the review is supplied separately"
    if _META_LEAK.search(prompt):
        return "echoes the prompt-writing instructions (PROMPT A/B, new prompt, merge them)"
    return None


def similarity(a: str, b: str) -> float:
    """Word-level similarity in [0, 1] (1 = identical ignoring case and whitespace)."""
    return difflib.SequenceMatcher(None, a.lower().split(), b.lower().split()).ratio()


def repair_prompt(prompt) -> str:
    """Append the fixed output rule if (and only if) that is the prompt's only defect.

    A weaker closing sentence such as 'Answer with positive or negative.' is replaced rather
    than duplicated, and a dangling colon is turned into a full stop.
    """
    p = normalize_prompt(prompt)
    if check_prompt(p) != _NEEDS_RULE:
        return p
    sentences = re.split(r"(?<=[.!?])\s+", p)
    if len(sentences) > 1 and _OUTPUT_SENTENCE.match(sentences[-1]):
        sentences = sentences[:-1]
    p = " ".join(sentences).rstrip(" :;,")
    if not p.endswith((".", "!", "?")):
        p += "."
    return f"{p} {OUTPUT_RULE}"


def prepare_candidate(raw) -> tuple[str | None, str | None]:
    """raw LLM text -> (valid single-line prompt, None) or (None, reason it was rejected)."""
    candidate = repair_prompt(clean_generated_prompt(raw))
    problem = check_prompt(candidate)
    return (None, problem) if problem else (candidate, None)