"""CPU-only tests for src/llm: no torch, no model download (generation is faked)."""
import json

import pytest

from src.llm.config import ModelConfig
from src.llm.model_interface import USER_TEMPLATE_ID, LLMInterface, label_from_raw

PROMPT = "Classify the review. Return exactly one word: positive or negative."


def make_llm(generate, batch_size=2):
    llm = LLMInterface(ModelConfig(batch_size=batch_size))
    llm.model = llm.tokenizer = object()          # pretend it is loaded
    llm._render = lambda system, user: f"{system}||{user}"
    llm._generate = generate
    return llm


def keyword_generate(rendered, **kw):
    return ["positive" if "good" in r else "negative" for r in rendered]


@pytest.mark.parametrize("raw,expected", [
    ("positive", "positive"), (" Negative.\n", "negative"), ("**positive**", "positive"),
    ("The answer is negative.", "invalid"), ("not positive", "invalid"),
    ("positive and negative", "invalid"), ("", "invalid"), (None, "invalid"),
])
def test_label_from_raw_uses_strict_policy(raw, expected):
    assert label_from_raw(raw) == expected
    assert LLMInterface.parse_sentiment(raw) == expected


def test_not_loaded_raises():
    with pytest.raises(RuntimeError):
        LLMInterface(ModelConfig()).predict_raw_batch(PROMPT, ["x"])


def test_order_preserved_and_batches_respect_batch_size():
    sizes = []

    def gen(rendered, **kw):
        sizes.append(len(rendered))
        return keyword_generate(rendered)

    llm = make_llm(gen, batch_size=2)
    texts = ["good good good good", "bad", "good", "bad bad bad bad bad", "fine good day"]
    assert llm.predict_raw_batch(PROMPT, texts) == ["positive", "negative", "positive",
                                                    "negative", "positive"]
    assert max(sizes) <= 2 and sum(sizes) == 5


def test_empty_input_returns_empty_list():
    assert make_llm(keyword_generate).predict_raw_batch(PROMPT, []) == []


def test_one_bad_input_becomes_none_and_others_survive():
    def gen(rendered, **kw):
        if any("BOOM" in r for r in rendered):
            raise ValueError("boom")
        return keyword_generate(rendered)

    llm = make_llm(gen, batch_size=3)
    assert llm.predict_raw_batch(PROMPT, ["good", "BOOM", "bad"]) == ["positive", None, "negative"]
    assert llm.stats["failed_generations"] == 1


def test_oom_splits_batches_until_they_fit():
    def gen(rendered, **kw):
        if len(rendered) > 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        return keyword_generate(rendered)

    llm = make_llm(gen, batch_size=4)
    assert llm.predict_raw_batch(PROMPT, ["good", "bad", "good", "bad"]) == [
        "positive", "negative", "positive", "negative"]


def test_systematic_failure_raises():
    def gen(rendered, **kw):
        raise ValueError("model is broken")

    with pytest.raises(RuntimeError, match="All 2 generations failed"):
        make_llm(gen).predict_raw_batch(PROMPT, ["a", "b"])


def test_input_validation():
    llm = make_llm(keyword_generate)
    with pytest.raises(ValueError):
        llm.predict_raw_batch("  ", ["x"])
    with pytest.raises(TypeError):
        llm.predict_raw_batch(PROMPT, ["ok", 3])


def test_predict_sentiment_and_predict_batch_return_labels():
    def gen(rendered, **kw):
        return ["Positive." if "good" in r else "I think it is negative" for r in rendered]

    llm = make_llm(gen)
    assert llm.predict_sentiment(PROMPT, "good film") == "positive"
    assert llm.predict_batch(PROMPT, ["good", "bad"]) == ["positive", "invalid"]


def test_classification_is_greedy_and_review_is_in_user_message():
    seen = {}

    def gen(rendered, **kw):
        seen.update(kw, rendered=rendered)
        return ["positive"] * len(rendered)

    llm = make_llm(gen)
    llm.predict_raw_batch(PROMPT, ["a fine movie"])
    assert seen["sample"] is False and seen["max_new_tokens"] == llm.config.classify_max_new_tokens
    system, user = seen["rendered"][0].split("||")
    assert system == PROMPT and "a fine movie" in user and "a fine movie" not in system


def test_generate_text_samples_only_when_asked():
    calls = []

    def gen(rendered, **kw):
        calls.append(kw)
        return ["some prompt text"]

    llm = make_llm(gen)
    llm.generate_text("sys", "user")
    llm.generate_text("sys", "user", sample=True, temperature=1.0)
    assert calls[0]["sample"] is False
    assert calls[1]["sample"] is True and calls[1]["temperature"] == 1.0
    assert calls[1]["top_p"] == llm.config.generate_top_p
    assert llm.stats["generate_calls"] == 2


def test_cache_identity_properties():
    llm = LLMInterface(ModelConfig(model_id="org/model", revision="abc123"))
    assert llm.model_id == "org/model@abc123"
    assert LLMInterface(ModelConfig(model_id="org/model")).model_id == "org/model"
    cfg = llm.inference_config
    assert cfg["decoding"] == "greedy" and cfg["user_template"] == USER_TEMPLATE_ID
    json.dumps(cfg)  # must be JSON-serialisable for the cache key