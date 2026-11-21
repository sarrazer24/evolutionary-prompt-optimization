from src.llm.config import DEFAULT_MODEL_ID, DEFAULT_REVISION, ModelConfig
from src.llm.model_interface import LLMInterface


def test_default_model_is_pinned_to_an_exact_commit():
    cfg = ModelConfig()
    assert cfg.model_id == DEFAULT_MODEL_ID and cfg.revision == DEFAULT_REVISION
    assert len(DEFAULT_REVISION) == 40 and all(c in "0123456789abcdef" for c in DEFAULT_REVISION)
    assert LLMInterface(cfg).model_id == f"{DEFAULT_MODEL_ID}@{DEFAULT_REVISION}"   # part of the cache key


def test_other_models_are_not_given_the_qwen_revision():
    assert ModelConfig(model_id="microsoft/Phi-3.5-mini-instruct").revision is None


def test_explicit_revision_is_respected():
    assert ModelConfig(revision="main").revision == "main"
    assert ModelConfig(model_id="org/model", revision="abc").revision == "abc"


def test_environment_override(monkeypatch):
    monkeypatch.setenv("HF_MODEL_ID", "org/other")
    monkeypatch.setenv("HF_MODEL_REVISION", "deadbeef")
    monkeypatch.setenv("LLM_BATCH_SIZE", "16")
    cfg = ModelConfig.from_environment()
    assert (cfg.model_id, cfg.revision, cfg.batch_size) == ("org/other", "deadbeef", 16)


def test_environment_defaults_keep_the_pin(monkeypatch):
    for var in ("HF_MODEL_ID", "HF_MODEL_REVISION", "LLM_BATCH_SIZE", "LLM_DTYPE"):
        monkeypatch.delenv(var, raising=False)
    assert ModelConfig.from_environment().revision == DEFAULT_REVISION