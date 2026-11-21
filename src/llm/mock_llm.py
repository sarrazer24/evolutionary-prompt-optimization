"""Deterministic stand-in for LLMInterface (Member 2) - for CPU development and tests ONLY.

It has the same public surface as ``LLMInterface`` (``model_id``, ``inference_config``,
``predict_raw_batch``, ``predict_sentiment``, ``predict_batch``, ``generate_text``,
``set_seed``, ``stats``), so the Genetic Algorithm can be written and tested without a GPU.

* Classification: answers come from an ORACLE (text -> true label) corrupted by a per-prompt
  error rate, so different prompts get different, reproducible fitness. A few prompts that
  mention "mixed"/"neutral" produce invalid answers, like the real model.
* Prompt writing: ``generate_text`` assembles random but valid prompts from phrase lists.

!!! NEVER report numbers produced with this class: they are synthetic. !!!
"""
from __future__ import annotations

import hashlib
import random
from typing import Iterable, Optional

from ..evaluation.evaluation import INVALID, LABEL_NAMES, parse_response
from ..prompts.validation import OUTPUT_RULE

_ROLES = ["You are an expert film critic.", "Act as a careful annotator.",
          "You are a strict sentiment classifier.", "Imagine you are a movie reviewer reading audience feedback.",
          "You are a data labeling assistant.", "Take the role of a seasoned editor.", ""]
_TASKS = ["Decide whether the review is positive or negative.",
          "Judge the overall sentiment of the text as positive or negative.",
          "Determine if the writer's opinion is positive or negative.",
          "Classify the text as positive or negative."]
_CRITERIA = ["Weigh praise against criticism.", "Focus on the overall tone rather than single words.",
             "Treat negation and sarcasm carefully.", "Look for words that show approval or disappointment.",
             "If opinions are mixed, choose the dominant one.", "Ignore plot summary and judge the opinion."]
_ENDINGS = [OUTPUT_RULE, "Reply with only one word: positive or negative.",
            "Respond with exactly one label and nothing else: positive or negative.", ""]  # "" -> repaired later


def _u(*parts: str) -> float:
    """Deterministic pseudo-random number in [0, 1) from strings."""
    digest = hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 2**32


class MockLLM:
    model_id = "MOCK-oracle-noise-v1"

    def __init__(self, oracle: Optional[dict] = None, seed: int = 0,
                 error_floor: float = 0.06, error_spread: float = 0.14):
        self.oracle = dict(oracle or {})
        self.error_floor, self.error_spread = error_floor, error_spread
        self.inference_config = {"decoding": "greedy", "mock": True}
        self._rng = random.Random(seed)
        self.stats = {"classified_texts": 0, "classify_seconds": 0.0,
                      "generate_calls": 0, "failed_generations": 0}

    # ------------------------------------------------------------------ classification
    def error_rate(self, prompt: str) -> float:
        return self.error_floor + self.error_spread * _u("quality", prompt)

    def _answer(self, prompt: str, text: str) -> str:
        true = self.oracle.get(text)
        if true is None:
            true = int(_u("label", text) < 0.5)
        if "mixed" in prompt.lower() or "neutral" in prompt.lower():
            if _u("invalid", prompt, text) < 0.03:
                return "neutral"
        if _u("flip", prompt, text) < self.error_rate(prompt):
            true = 1 - true
        return LABEL_NAMES[true]

    def predict_raw_batch(self, prompt: str, texts: Iterable[str]) -> list[Optional[str]]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        texts = list(texts)
        self.stats["classified_texts"] += len(texts)
        return [self._answer(prompt, t) for t in texts]

    def predict_sentiment(self, prompt: str, text: str) -> str:
        return self.predict_batch(prompt, [text])[0]

    def predict_batch(self, prompt: str, texts: Iterable[str]) -> list[str]:
        out = []
        for raw in self.predict_raw_batch(prompt, texts):
            code = parse_response(raw)
            out.append("invalid" if code == INVALID else LABEL_NAMES[code])
        return out

    # ------------------------------------------------------------------ prompt writing
    def set_seed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def generate_text(self, system_prompt: str, user_prompt: str, max_new_tokens: Optional[int] = None,
                      *, sample: bool = False, temperature: Optional[float] = None,
                      top_p: Optional[float] = None) -> str:
        self.stats["generate_calls"] += 1
        r = self._rng
        parts = [r.choice(_ROLES), r.choice(_TASKS)] + r.sample(_CRITERIA, r.randint(0, 2)) + [r.choice(_ENDINGS)]
        return " ".join(p for p in parts if p)