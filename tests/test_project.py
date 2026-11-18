"""Unit tests. Everything is mocked: no LLM, no internet, no Hugging Face download.

Run from the project root:  python -m pytest -v
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.data import sst2_data
from src.evaluation import evaluation
from src.evaluation.evaluation import (INVALID, PredictionCache, compute_metrics, evaluate_prompt, parse_response,
                        results_to_dataframe, save_results)
from src.data.sst2_data import (check_splits, load_final_test, load_saved_splits, make_splits, prepare_splits,
                       stratified_sample)

TEXTS = ["great movie", "awful film", "loved it", "terrible plot"]
LABELS = [1, 0, 1, 0]
KW = dict(model_id="mock-model-v1")


def perfect_fn(prompt, texts):
    return ["Positive" if t in ("great movie", "loved it") else "negative" for t in texts]


def counting_fn(calls):
    def fn(prompt, texts):
        calls.append(list(texts))
        return perfect_fn(prompt, texts)
    return fn


def fake_train(n=1000, seed=0):
    """Synthetic SST-2-like frame: ~56% positive, with duplicates."""
    rng = np.random.default_rng(seed)

    texts = [f"phrase number {i}" for i in range(n)]

    # Include short texts and an ordinary duplicate.
    texts[0], texts[1], texts[2] = "good", "bad", "good"

    # Generate labels.
    labels = (rng.random(n) < 0.56).astype(int)

    # Ensure both occurrences of "good" have the same label.
    labels[2] = labels[0]

    return pd.DataFrame({
        "text": texts,
        "label": labels
    })


# ---------------------------------------------------------------- parsing (1-3)
@pytest.mark.parametrize("raw", ["positive", "Positive", "POSITIVE", " positive\n", "positive.",
                                 "**Positive**", '"positive"'])
def test_parse_positive(raw):
    assert parse_response(raw) == 1


@pytest.mark.parametrize("raw", ["negative", "Negative", " NEGATIVE ", "negative!", "`negative`"])
def test_parse_negative(raw):
    assert parse_response(raw) == 0


@pytest.mark.parametrize("raw", [None, "", "   ", "positive and negative", "positive, negative", "uncertain",
                                 "neutral", "not positive", "not negative", "Sentiment: positive",
                                 "The sentiment is negative.", 1, True, ["positive"]])
def test_parse_invalid(raw):
    assert parse_response(raw) == INVALID


# ---------------------------------------------------------------- metrics (4-9)
def test_perfect_predictions():
    r = evaluate_prompt("p", TEXTS, LABELS, perfect_fn, **KW)
    m = r["metrics"]
    assert m["accuracy"] == m["f1_macro"] == m["precision_macro"] == m["recall_macro"] == 1.0
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0]]
    assert m["n_invalid"] == 0 and m["invalid_rate"] == 0.0


def test_incorrect_predictions():
    r = evaluate_prompt("p", TEXTS, LABELS, lambda p, t: ["positive"] * len(t), **KW)
    m = r["metrics"]
    assert m["accuracy"] == 0.5
    assert m["confusion_matrix"] == [[0, 2, 0], [0, 2, 0]]
    assert m["f1_macro"] == pytest.approx((0.0 + 2 / 3) / 2)  # neg F1 = 0, pos F1 = 2/3


def test_all_wrong_flipped_labels():
    r = evaluate_prompt("p", TEXTS, LABELS, lambda p, t: ["negative", "positive", "negative", "positive"], **KW)
    assert r["metrics"]["accuracy"] == 0.0 and r["metrics"]["f1_macro"] == 0.0


def test_hand_computed_metrics_with_invalid():
    # y_true = [1,1,0,0,1,0]; y_pred = [1,1,0,1,INVALID,0]
    y_true = [1, 1, 0, 0, 1, 0]
    raw = ["positive", "positive", "negative", "positive", "???", "negative"]
    y_pred = [parse_response(x) for x in raw]
    m = compute_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(4 / 6)
    # class 1: TP=2, FP=1, true=3 -> P=2/3, R=2/3, F1=2/3 ; class 0: TP=2, FP=0, true=3 -> P=1, R=2/3, F1=0.8
    assert m["precision_macro"] == pytest.approx((2 / 3 + 1) / 2)
    assert m["recall_macro"] == pytest.approx(2 / 3)
    assert m["f1_macro"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert m["confusion_matrix"] == [[2, 1, 0], [0, 2, 1]]


def test_invalid_counted_consistently():
    r = evaluate_prompt("p", TEXTS, LABELS, lambda p, t: ["positive", "???", None, ""], **KW)
    m = r["metrics"]
    assert m["n_invalid"] == 3 == r["y_pred"].count(INVALID)
    assert m["invalid_rate"] == pytest.approx(0.75)
    assert sum(map(sum, m["confusion_matrix"])) == m["n_examples"] == 4  # every example is in the matrix
    assert m["confusion_matrix"] == [[0, 0, 2], [0, 1, 1]]  # true neg: 2 invalid; true pos: 1 ok, 1 invalid


def test_all_invalid_gives_zero_not_error():
    r = evaluate_prompt("p", TEXTS, LABELS, lambda p, t: ["???"] * len(t), **KW)
    m = r["metrics"]
    assert m["accuracy"] == 0.0 and m["f1_macro"] == 0.0 and m["invalid_rate"] == 1.0


# ---------------------------------------------------------------- interface & validation (10-12)
def test_result_structure_and_fitness_key():
    r = evaluate_prompt("p", TEXTS, LABELS, perfect_fn, split_name="opt", inference_config={"temperature": 0}, **KW)
    assert r["split"] == "opt" and r["model_id"] == "mock-model-v1" and r["inference_config"] == {"temperature": 0}
    assert r["n_examples"] == 4 and len(r["raw_responses"]) == 4 == len(r["y_pred"])
    assert r["metrics"]["f1_macro"] == 1.0  # GA fitness


def test_wrong_output_length_raises():
    with pytest.raises(ValueError):
        evaluate_prompt("p", TEXTS, LABELS, lambda p, t: ["positive"], **KW)


def test_predict_fn_must_return_a_list():
    with pytest.raises(TypeError):
        evaluate_prompt("p", TEXTS, LABELS, lambda p, t: "positive", **KW)


@pytest.mark.parametrize("kwargs", [
    dict(examples=[], labels=[]),
    dict(examples=TEXTS, labels=[1, 0]),
    dict(examples=TEXTS, labels=[1, 0, 1, 2]),
    dict(examples=TEXTS, labels=[1, 0, 1, -1]),
    dict(examples=TEXTS, labels=[1, 0, 1, "positive"]),
    dict(examples=["ok", 5, "ok", "ok"], labels=LABELS),
    dict(examples=TEXTS, labels=LABELS, prompt="  "),
    dict(examples=TEXTS, labels=LABELS, model_id=""),
    dict(examples=TEXTS, labels=LABELS, predict_fn=None),
    dict(examples=TEXTS, labels=LABELS, batch_size=0),
    dict(examples=TEXTS, labels=LABELS, cache="not a cache"),
])
def test_input_validation(kwargs):
    args = dict(prompt="p", predict_fn=perfect_fn, model_id="m")
    args.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        evaluate_prompt(**args)


def test_batch_size_splits_calls():
    calls = []
    texts = [f"text {i}" for i in range(7)]
    evaluate_prompt("p", texts, [1] * 7, counting_fn(calls), batch_size=3, **KW)
    assert [len(c) for c in calls] == [3, 3, 1]


# ---------------------------------------------------------------- cache (13-18)
def test_cache_prevents_repeated_calls():
    calls, cache = [], PredictionCache()
    evaluate_prompt("p", TEXTS, LABELS, counting_fn(calls), cache=cache, **KW)
    r2 = evaluate_prompt("p", TEXTS, LABELS, counting_fn(calls), cache=cache, **KW)
    assert len(calls) == 1 and r2["n_cached"] == 4 and r2["n_model_texts"] == 0
    evaluate_prompt("another prompt", TEXTS, LABELS, counting_fn(calls), cache=cache, **KW)
    assert len(calls) == 2  # a different prompt is a cache miss


def test_cache_key_depends_on_model_prompt_text_and_config():
    base = PredictionCache.make_key("m1", "p", "t", {"temperature": 0})
    assert base == PredictionCache.make_key("m1", "p", "t", {"temperature": 0})
    assert base != PredictionCache.make_key("m2", "p", "t", {"temperature": 0})  # model version
    assert base != PredictionCache.make_key("m1", "p2", "t", {"temperature": 0})
    assert base != PredictionCache.make_key("m1", "p", "t2", {"temperature": 0})
    assert base != PredictionCache.make_key("m1", "p", "t", {"temperature": 0.7})  # inference config
    assert PredictionCache.make_key("m1", "p", "t", {"a": 1, "b": 2}) == PredictionCache.make_key("m1", "p", "t", {"b": 2, "a": 1})


def test_changing_model_or_config_invalidates_cache():
    calls, cache = [], PredictionCache()
    evaluate_prompt("p", TEXTS, LABELS, counting_fn(calls), cache=cache, model_id="m1", inference_config={"temperature": 0})
    evaluate_prompt("p", TEXTS, LABELS, counting_fn(calls), cache=cache, model_id="m2", inference_config={"temperature": 0})
    evaluate_prompt("p", TEXTS, LABELS, counting_fn(calls), cache=cache, model_id="m1", inference_config={"temperature": 1})
    assert len(calls) == 3


def test_cache_persistence(tmp_path):
    path = tmp_path / "sub" / "cache.jsonl"
    calls = []
    evaluate_prompt("p", TEXTS, LABELS, counting_fn(calls), cache=PredictionCache(path), **KW)
    reloaded = PredictionCache(path)
    assert len(reloaded) == 4
    r = evaluate_prompt("p", TEXTS, LABELS, counting_fn(calls), cache=reloaded, **KW)
    assert len(calls) == 1 and r["n_cached"] == 4 and r["metrics"]["accuracy"] == 1.0


def test_none_responses_are_not_cached_and_are_retried():
    cache, state = PredictionCache(), {"fail": True}

    def flaky(prompt, texts):
        return [None] * len(texts) if state["fail"] else ["positive"] * len(texts)

    r1 = evaluate_prompt("p", ["a", "b"], [1, 1], flaky, cache=cache, **KW)
    assert r1["metrics"]["n_invalid"] == 2 and len(cache) == 0
    state["fail"] = False
    r2 = evaluate_prompt("p", ["a", "b"], [1, 1], flaky, cache=cache, **KW)
    assert r2["metrics"]["accuracy"] == 1.0 and len(cache) == 2


def test_corrupted_cache_lines_are_skipped(tmp_path):
    path = tmp_path / "cache.jsonl"
    key = PredictionCache.make_key("m", "p", "t")
    path.write_text(json.dumps({"key": key, "raw": "positive"}) + "\n{not json\n" + '{"key": 1}\n', encoding="utf-8")
    cache = PredictionCache(path)
    assert len(cache) == 1 and cache.n_skipped_lines == 2 and cache.get(key) == "positive"


def test_duplicate_texts_are_sent_once():
    calls = []
    texts = ["same", "other", "same"]
    r = evaluate_prompt("p", texts, [1, 0, 1], lambda p, t: (calls.append(list(t)) or ["positive"] * len(t)), **KW)
    assert calls == [["same", "other"]] and r["n_model_texts"] == 2 and len(r["y_pred"]) == 3


# ---------------------------------------------------------------- saving (19-20)
def test_save_json(tmp_path):
    r = evaluate_prompt("my prompt", TEXTS, LABELS, perfect_fn, split_name="opt", inference_config={"temperature": 0}, **KW)
    save_results(r, tmp_path / "out" / "r.json")
    data = json.loads((tmp_path / "out" / "r.json").read_text(encoding="utf-8"))
    assert len(data) == 1 and data[0]["prompt"] == "my prompt" and data[0]["metrics"]["f1_macro"] == 1.0
    assert data[0]["raw_responses"] and data[0]["metrics"]["confusion_matrix"] == [[2, 0, 0], [0, 2, 0]]
    save_results(r, tmp_path / "small.json", include_predictions=False)
    assert "y_pred" not in json.loads((tmp_path / "small.json").read_text(encoding="utf-8"))[0]


def test_save_csv_columns_and_append(tmp_path):
    r1 = evaluate_prompt("p1", TEXTS, LABELS, perfect_fn, split_name="opt", **KW)
    r2 = evaluate_prompt("p2", TEXTS, LABELS, lambda p, t: ["???"] * len(t), split_name="val", **KW)
    path = tmp_path / "r.csv"
    save_results([r1, r2], path)
    save_results(r1, path)  # appended, header not repeated
    df = pd.read_csv(path)
    assert len(df) == 3 and list(df["split"]) == ["opt", "val", "opt"]
    for col in ["prompt", "model_id", "inference_config", "n_examples", "accuracy", "precision_macro",
                "recall_macro", "f1_macro", "n_invalid", "invalid_rate", "tn", "fp", "neg_invalid",
                "fn", "tp", "pos_invalid"]:
        assert col in df.columns
    assert df.loc[1, "n_invalid"] == 4 and df.loc[1, "neg_invalid"] == 2 and df.loc[1, "pos_invalid"] == 2
    save_results(r1, path, append=False)
    assert len(pd.read_csv(path)) == 1


def test_save_rejects_unknown_extension(tmp_path):
    r = evaluate_prompt("p", TEXTS, LABELS, perfect_fn, **KW)
    with pytest.raises(ValueError):
        save_results(r, tmp_path / "r.txt")
    assert len(results_to_dataframe(r)) == 1


# ---------------------------------------------------------------- dataset splits (21-27)
def test_split_sizes_80_20_and_nothing_removed():
    train = fake_train(1000)
    s = make_splits(train)
    assert abs(len(s.opt) - 800) <= 1
    assert abs(len(s.val) - 200) <= 1
    assert set(s.opt["train_idx"]) | set(s.val["train_idx"]) == set(range(1000))
    assert "good" in set(s.opt["text"]) | set(s.val["text"])  # 1-word texts are kept
    assert (s.opt["text"].tolist() + s.val["text"].tolist()).count("good") == 2  # duplicates are kept


def test_split_reproducible():
    train = fake_train(1000)
    a, b = make_splits(train, seed=7), make_splits(train, seed=7)
    assert a.opt.equals(b.opt) and a.val.equals(b.val)
    c = make_splits(train, seed=8)
    assert not a.opt["train_idx"].equals(c.opt["train_idx"])


def test_split_is_stratified():
    train = fake_train(1000)
    s = make_splits(train)
    overall = train["label"].mean()
    assert abs(s.opt["label"].mean() - overall) <= 2 / len(s.opt)
    assert abs(s.val["label"].mean() - overall) <= 2 / len(s.val)
    assert s.info["opt"]["n_positive"] + s.info["val"]["n_positive"] == int(train["label"].sum())


def test_no_overlap_between_splits():
    s = make_splits(fake_train(1000))
    assert not set(s.opt["train_idx"]) & set(s.val["train_idx"])
    assert all(check_splits(s, fake_train(1000)).values())


def test_check_splits_detects_problems():
    train = fake_train(1000)
    s = make_splits(train)
    s.val = pd.concat([s.val, s.opt.head(5)], ignore_index=True)  # leak 5 rows
    with pytest.raises(AssertionError):
        check_splits(s, train)
    assert check_splits(s, train, raise_on_fail=False)["no_row_overlap_between_opt_and_val"] is False


@pytest.mark.parametrize("bad", [
    pd.DataFrame({"text": ["a", "b"]}),
    pd.DataFrame({"text": ["a", None], "label": [0, 1]}),
    pd.DataFrame({"text": ["a", "b"], "label": [0, 2]}),
    pd.DataFrame({"text": ["a", "b"], "label": [0, -1]}),
    pd.DataFrame({"text": ["a", 3], "label": [0, 1]}),
    pd.DataFrame({"text": [], "label": []}),
])
def test_make_splits_input_validation(bad):
    with pytest.raises((ValueError, TypeError)):
        make_splits(bad)


def test_make_splits_rejects_bad_fraction():
    for frac in (0, 1, -0.2, 1.5):
        with pytest.raises(ValueError):
            make_splits(fake_train(100), opt_fraction=frac)


def test_stratified_sample_is_fixed_and_balanced():
    df = make_splits(fake_train(1000)).opt
    a, b = stratified_sample(df, 100, seed=1), stratified_sample(df, 100, seed=1)
    assert a.equals(b) and len(a) == 100
    assert abs(a["label"].mean() - df["label"].mean()) <= 0.02
    with pytest.raises(ValueError):
        stratified_sample(df, 1)


# ---------------------------------------------------------------- final-test protection (28-30)
def test_final_test_requires_explicit_confirmation(monkeypatch):
    def boom(split):
        raise AssertionError("test data must not be loaded without confirmation")

    monkeypatch.setattr(sst2_data, "_load_hf_split", boom)
    with pytest.raises(PermissionError):
        load_final_test()
    with pytest.raises(PermissionError):
        load_final_test(confirm="yes")  # only the boolean True counts


def test_prepare_splits_never_saves_or_leaks_the_test_set(tmp_path, monkeypatch):
    train = fake_train(1000)
    test = pd.DataFrame({"text": ["unique held out sentence one", "unique held out sentence two", "phrase number 5"],
                         "label": [1, 0, 1]})  # the last test text also occurs in train -> must be reported
    requested = []

    def fake_loader(split):
        requested.append(split)
        df = (train if split == "train" else test).copy()
        df.attrs["source"] = "fake"
        return df

    monkeypatch.setattr(sst2_data, "_load_hf_split", fake_loader)
    s = prepare_splits(save_dir=tmp_path,check_test_leakage=True,
)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["opt.csv", "split_info.json", "val.csv"]
    saved = "".join(p.read_text(encoding="utf-8") for p in tmp_path.iterdir())
    assert "unique held out sentence" not in saved  # no test text anywhere in shared files
    leak = s.info["test_leakage_check"]
    assert leak["n_test"] == 3 and leak["test_texts_also_in_opt"] + leak["test_texts_also_in_val"] == 1
    assert all(s.info["checks"].values()) and s.info["source"] == "fake"
    assert requested[0] == "train"


def test_load_saved_splits_roundtrip(tmp_path, monkeypatch):
    train = fake_train(200)
    monkeypatch.setattr(sst2_data, "_load_hf_split", lambda split: train.copy())
    s = prepare_splits(save_dir=tmp_path, check_test_leakage=False)
    r = load_saved_splits(tmp_path)
    assert r.opt["text"].tolist() == s.opt["text"].tolist() and r.val["label"].tolist() == s.val["label"].tolist()
    assert r.info["seed"] == 42 and r.info["n_total"] == 200
