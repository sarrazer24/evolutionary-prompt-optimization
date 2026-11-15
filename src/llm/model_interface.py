"""Minimal local Hugging Face chat-model adapter.

A candidate prompt is passed as the system message. The review is passed
separately as the user message. This is the shared interface for the team.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

import torch

from .config import ModelConfig


class LLMInterface:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig.from_environment()
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Load model once. Call this once per Kaggle session."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # FP16 is generally suitable for Kaggle NVIDIA GPUs such as T4.
        # If this model does not fit, select a smaller model rather than
        # silently changing quantization or using a paid hosted endpoint.
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.model.eval()
        return self

    def _require_loaded(self):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

    def _render_messages(self, system_prompt: str, user_message: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate free-form text, used by prompt generation/operators."""
        self._require_loaded()
        rendered = self._render_messages(system_prompt, user_prompt)
        inputs = self.tokenizer(rendered, return_tensors="pt")
        device = self.model.get_input_embeddings().weight.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature if self.config.do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def parse_sentiment(raw_output: str) -> str:
        """Return positive, negative, or invalid; never guess if ambiguous."""
        text = (raw_output or "").strip().lower()
        positive = bool(re.search(r"\bpositive\b", text))
        negative = bool(re.search(r"\bnegative\b", text))
        if positive and not negative:
            return "positive"
        if negative and not positive:
            return "negative"
        return "invalid"

    def predict_sentiment(self, prompt: str, text: str) -> str:
        """Classify one review using prompt as system instruction."""
        raw = self.generate_text(
            system_prompt=prompt,
            user_prompt=f"Review:\n{text}\n\nSentiment:",
            max_new_tokens=self.config.max_new_tokens,
        )
        return self.parse_sentiment(raw)

    def predict_batch(self, prompt: str, texts: Iterable[str]) -> list[str]:
        """Simple reliable batch API; currently runs one review at a time."""
        return [self.predict_sentiment(prompt, text) for text in texts]
