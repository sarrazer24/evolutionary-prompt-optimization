"""Configuration for the local Hugging Face instruction model (Member 2)."""
from __future__ import annotations

import os
from dataclasses import dataclass

# Apache-2.0, ungated, ~2.9 GB in fp16: fits one Kaggle T4 (15 GB) with lots of headroom.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class ModelConfig:
    model_id: str = DEFAULT_MODEL_ID
    revision: str | None = None        # pin a commit hash for exact reproducibility
    dtype: str = "auto"                # "auto" -> float16 on CUDA, float32 on CPU
    device: str = "auto"               # "auto" -> cuda:0 if available, else cpu
    trust_remote_code: bool = False

    # Classification: ALWAYS greedy (deterministic). Same settings for every prompt.
    classify_max_new_tokens: int = 8   # a label is 1-2 tokens; longer answers are invalid anyway
    batch_size: int = 32

    # Prompt generation / mutation / crossover: sampled, for diversity.
    generate_max_new_tokens: int = 256
    generate_temperature: float = 0.8
    generate_top_p: float = 0.95

    @classmethod
    def from_environment(cls) -> "ModelConfig":
        """Override defaults from environment variables (handy on Kaggle). No secrets."""
        return cls(
            model_id=os.getenv("HF_MODEL_ID", DEFAULT_MODEL_ID),
            revision=os.getenv("HF_MODEL_REVISION") or None,
            dtype=os.getenv("LLM_DTYPE", "auto"),
            batch_size=int(os.getenv("LLM_BATCH_SIZE", "32")),
        )