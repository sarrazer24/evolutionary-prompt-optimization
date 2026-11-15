"""Configuration for the local Hugging Face instruction model."""
from dataclasses import dataclass
import os


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    max_new_tokens: int = 32
    temperature: float = 0.0
    do_sample: bool = False
    trust_remote_code: bool = False

    @classmethod
    def from_environment(cls):
        """Allow Kaggle/local environment overrides without storing secrets."""
        return cls(
            model_id=os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"),
            max_new_tokens=int(os.getenv("LLM_MAX_NEW_TOKENS", "32")),
        )
