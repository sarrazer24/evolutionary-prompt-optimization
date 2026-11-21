"""Local Hugging Face chat-model adapter (Member 2).

Contract
--------
* The candidate prompt is the SYSTEM message; the review is a separate USER message.
* Classification is greedy/deterministic. Prompt generation (mutation, crossover, initial
  population) is sampled via ``generate_text(..., sample=True)``.
* ``predict_raw_batch(prompt, texts)`` is the ``predict_fn`` for Member 1's
  ``evaluate_prompt``. It returns RAW model text (``None`` for a failed generation, which
  Member 1's cache never stores, so it is retried next time). Member 1's strict
  ``parse_response`` is the single source of truth for turning raw text into a label.
* torch / transformers are imported lazily, so unit tests run without them.
"""
from __future__ import annotations

import logging
import time
from typing import Iterable, Optional

from ..evaluation.evaluation import INVALID, LABEL_NAMES, parse_response
from .config import ModelConfig

log = logging.getLogger(__name__)

USER_TEMPLATE = "Review:\n{text}\n\nSentiment:"
USER_TEMPLATE_ID = "review-sentiment-v1"  # part of the cache key: bump it if the template changes


def label_from_raw(raw) -> str:
    """Raw model text -> 'positive' | 'negative' | 'invalid' (uses Member 1's strict parser)."""
    code = parse_response(raw)
    return "invalid" if code == INVALID else LABEL_NAMES[code]


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


class LLMInterface:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig.from_environment()
        self.tokenizer = None
        self.model = None
        self.device = None
        self.resolved_dtype: str | None = None
        self.resolved_revision: str | None = None
        self.last_error: str | None = None
        self.stats = {"classified_texts": 0, "classify_seconds": 0.0,
                      "generate_calls": 0, "failed_generations": 0}

    # ------------------------------------------------------------------ identity for caching
    @property
    def model_id(self) -> str:
        """Pass as ``model_id`` to ``evaluate_prompt`` (part of the cache key)."""
        c = self.config
        return f"{c.model_id}@{c.revision}" if c.revision else c.model_id

    @property
    def inference_config(self) -> dict:
        """Pass as ``inference_config`` to ``evaluate_prompt`` (part of the cache key)."""
        c = self.config
        return {
            "decoding": "greedy",
            "max_new_tokens": c.classify_max_new_tokens,
            "repetition_penalty": 1.0,
            "dtype": self.resolved_dtype or c.dtype,
            "user_template": USER_TEMPLATE_ID,
        }

    # ------------------------------------------------------------------ loading
    def load_model(self) -> "LLMInterface":
        """Load tokenizer + model ONCE (idempotent). Call once per Kaggle session."""
        if self.model is not None:
            return self
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        c = self.config
        if c.device == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(c.device)
        self.resolved_dtype = c.dtype if c.dtype != "auto" else (
            "float16" if self.device.type == "cuda" else "float32")
        dtype = getattr(torch, self.resolved_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            c.model_id, revision=c.revision, trust_remote_code=c.trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"   # required for batched generation

        kwargs = dict(revision=c.revision, trust_remote_code=c.trust_remote_code)
        try:  # transformers >= 4.56 / 5.x
            model = AutoModelForCausalLM.from_pretrained(c.model_id, dtype=dtype, **kwargs)
        except TypeError:  # older transformers
            model = AutoModelForCausalLM.from_pretrained(c.model_id, torch_dtype=dtype, **kwargs)
        self.model = model.to(self.device).eval()
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)
        log.info("Loaded %s (revision %s) on %s as %s", c.model_id, self.resolved_revision,
                 self.device, self.resolved_dtype)
        return self

    def _require_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

    def set_seed(self, seed: int) -> None:
        """Seed sampling (prompt generation). Call once at the start of a GA run."""
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def info(self) -> dict:
        """Everything worth logging for reproducibility."""
        import torch
        import transformers
        return {
            "model_id": self.config.model_id, "requested_revision": self.config.revision,
            "resolved_revision": self.resolved_revision, "dtype": self.resolved_dtype,
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "inference_config": self.inference_config,
        }

    # ------------------------------------------------------------------ low-level generation
    def _render(self, system_prompt: str, user_message: str) -> str:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)

    def _generate(self, rendered: list[str], *, max_new_tokens: int, sample: bool,
                  temperature: float | None = None, top_p: float | None = None) -> list[str]:
        """Generate for a batch of already-rendered chat prompts. Returns stripped new text."""
        import torch
        enc = self.tokenizer(rendered, return_tensors="pt", padding=True)
        enc.pop("token_type_ids", None)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        # Explicit values override the model's own generation_config (Qwen ships
        # repetition_penalty=1.05, top_k=20, ...), so every run uses the same settings.
        kwargs = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.0,
                      pad_token_id=self.tokenizer.pad_token_id)
        if sample:
            kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0)
        else:
            kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with torch.inference_mode():
            out = self.model.generate(**enc, **kwargs)
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        return [t.strip() for t in self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]

    # ------------------------------------------------------------------ free-form generation
    def generate_text(self, system_prompt: str, user_prompt: str,
                      max_new_tokens: Optional[int] = None, *, sample: bool = False,
                      temperature: Optional[float] = None, top_p: Optional[float] = None) -> str:
        """Free-form text for prompt generation. Greedy unless ``sample=True``.

        Raises on failure (the prompt operators catch it and retry).
        """
        self._require_loaded()
        c = self.config
        self.stats["generate_calls"] += 1
        return self._generate(
            [self._render(system_prompt, user_prompt)],
            max_new_tokens=max_new_tokens or c.generate_max_new_tokens,
            sample=sample,
            temperature=temperature if temperature is not None else c.generate_temperature,
            top_p=top_p if top_p is not None else c.generate_top_p,
        )[0]

    # ------------------------------------------------------------------ classification
    def _run_chunk(self, rendered: list[str]) -> list[Optional[str]]:
        """Greedy-generate a chunk. On ANY error split it in halves (handles CUDA OOM and
        isolates a single bad input); a single failing item yields None."""
        try:
            return self._generate(rendered, max_new_tokens=self.config.classify_max_new_tokens,
                                  sample=False)
        except Exception as exc:  # noqa: BLE001 - deliberate: isolate failures
            self.last_error = f"{type(exc).__name__}: {exc}"
            if _is_oom(exc):
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
            if len(rendered) == 1:
                self.stats["failed_generations"] += 1
                log.warning("Generation failed for one input: %s", self.last_error)
                return [None]
            mid = len(rendered) // 2
            return self._run_chunk(rendered[:mid]) + self._run_chunk(rendered[mid:])

    def predict_raw_batch(self, prompt: str, texts: Iterable[str]) -> list[Optional[str]]:
        """RAW answers, one per text, in input order. This is Member 1's ``predict_fn``.

        ``None`` marks a failed generation. If EVERY generation fails (a systematic
        problem, not one odd input) a RuntimeError is raised instead.
        """
        self._require_loaded()
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        texts = list(texts)
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("every text must be a string")
        if not texts:
            return []

        start = time.time()
        rendered = [self._render(prompt, USER_TEMPLATE.format(text=t)) for t in texts]
        order = sorted(range(len(texts)), key=lambda i: len(rendered[i]))  # less padding
        outputs: list[Optional[str]] = [None] * len(texts)
        step = max(1, self.config.batch_size)
        for s in range(0, len(order), step):
            idx = order[s:s + step]
            for i, out in zip(idx, self._run_chunk([rendered[i] for i in idx])):
                outputs[i] = out
        self.stats["classified_texts"] += len(texts)
        self.stats["classify_seconds"] += time.time() - start

        if all(o is None for o in outputs):
            raise RuntimeError(f"All {len(texts)} generations failed. Last error: {self.last_error}")
        return outputs

    @staticmethod
    def parse_sentiment(raw_output) -> str:
        """Kept for compatibility; delegates to Member 1's strict parser."""
        return label_from_raw(raw_output)

    def predict_sentiment(self, prompt: str, text: str) -> str:
        """Classify one review -> 'positive' | 'negative' | 'invalid'."""
        return label_from_raw(self.predict_raw_batch(prompt, [text])[0])

    def predict_batch(self, prompt: str, texts: Iterable[str]) -> list[str]:
        """Classify many reviews -> list of 'positive' | 'negative' | 'invalid'."""
        return [label_from_raw(r) for r in self.predict_raw_batch(prompt, texts)]