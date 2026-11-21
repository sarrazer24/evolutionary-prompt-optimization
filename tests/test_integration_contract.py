"""Contract tests: Member 1 (evaluation) + Member 2 (LLM, prompts) + a Member-3-style GA loop.

Runs on CPU with MockLLM. If these pass, Member 3 can rely on the interfaces below.
"""
import pytest

from scripts.ga_integration_demo import make_fitness, run_demo, select_final
from src.data import sst2_data
from src.evaluation.evaluation import PredictionCache, evaluate_prompt
from src.llm import MockLLM
from src.prompts.baseline_prompts import get_baseline_prompt_texts
from src.prompts.initial_population import generate_initial_prompts
from src.prompts.operators import crossover_prompts, mutate_prompt
from src.prompts.validation import check_prompt

N = 120
TEXTS = [f"review number {i} {'wonderful' if i % 2 else 'dreadful'}" for i in range(N)]
LABELS = [i % 2 for i in range(N)]
ORACLE = dict(zip(TEXTS, LABELS))


def make_llm(seed=0):
    return MockLLM(oracle=ORACLE, seed=seed)


# ------------------------------------------------------------------ MockLLM itself
def test_mock_has_the_llm_interface_surface():
    llm = make_llm()
    assert llm.model_id.startswith("MOCK") and isinstance(llm.inference_config, dict)
    raw = llm.predict_raw_batch("Classify. Answer with exactly one word: positive or negative.", TEXTS[:5])
    assert len(raw) == 5 and all(isinstance(r, str) for r in raw)
    assert llm.predict_batch("Classify. Answer with exactly one word: positive or negative.", TEXTS[:5]) \
        == [r if r in ("positive", "negative") else "invalid" for r in raw]


def test_mock_classification_is_deterministic_and_order_independent():
    prompt = "Judge the tone. Answer with exactly one word: positive or negative."
    a = make_llm().predict_raw_batch(prompt, TEXTS)
    b = make_llm().predict_raw_batch(prompt, list(reversed(TEXTS)))
    assert a == list(reversed(b))


def test_mock_prompts_differ_in_quality():
    llm = make_llm()
    rates = {llm.error_rate(p) for p in get_baseline_prompt_texts()}
    assert len(rates) > 4


# ------------------------------------------------------------------ the contract
def test_initial_population_contract():
    llm = make_llm()
    seeds = get_baseline_prompt_texts()
    population = generate_initial_prompts(llm, 12, seed_prompts=seeds)
    assert len(population) == 12 and len(set(population)) == 12
    assert population[:len(seeds)] == seeds                      # seeds come first, unchanged
    assert all(check_prompt(p) is None for p in population)


def test_operators_return_valid_strings_or_the_parent():
    llm = make_llm()
    a, b = get_baseline_prompt_texts()[:2]
    for _ in range(10):
        child = mutate_prompt(llm, a)
        assert isinstance(child, str) and (child == a or check_prompt(child) is None)
        child = crossover_prompts(llm, a, b)
        assert isinstance(child, str) and (child in (a, b) or check_prompt(child) is None)


def test_evaluate_prompt_works_with_llm_predict_raw_batch_and_caches():
    llm, cache = make_llm(), PredictionCache()
    prompt = get_baseline_prompt_texts()[0]
    kw = dict(model_id=llm.model_id, inference_config=llm.inference_config, cache=cache, batch_size=50)
    first = evaluate_prompt(prompt, TEXTS, LABELS, llm.predict_raw_batch, **kw)
    sent = llm.stats["classified_texts"]
    second = evaluate_prompt(prompt, TEXTS, LABELS, llm.predict_raw_batch, **kw)
    assert 0.0 <= first["metrics"]["f1_macro"] <= 1.0
    assert second["n_cached"] == N and llm.stats["classified_texts"] == sent      # nothing re-sent
    assert second["metrics"]["f1_macro"] == first["metrics"]["f1_macro"]


def test_each_distinct_prompt_is_scored_exactly_once():
    llm = make_llm()
    fitness, scores = make_fitness(llm, TEXTS, LABELS)
    p = get_baseline_prompt_texts()[0]
    values = {fitness(p) for _ in range(5)}
    assert len(values) == 1 and llm.stats["classified_texts"] == N and list(scores) == [p]


# ------------------------------------------------------------------ the demo GA loop
def run(seed=1):
    llm = make_llm(seed)
    out = run_demo(llm, TEXTS, LABELS, pop_size=10, generations=3, seed=seed, log=lambda *_: None)
    return llm, out


def test_demo_loop_elitism_valid_unique_and_cached():
    llm, out = run()
    best = [h["best"] for h in out["history"]]
    assert best == sorted(best)                                              # elitism: never worse
    prompts = [p for p, _ in out["ranking"]]
    assert len(prompts) == len(set(prompts)) == 10
    assert all(check_prompt(p) is None for p in prompts)
    assert llm.stats["classified_texts"] == len(out["scores"]) * N           # no prompt scored twice


def test_demo_loop_is_reproducible_for_a_fixed_seed():
    _, a = run(seed=3)
    _, b = run(seed=3)
    assert a["ranking"] == b["ranking"] and a["history"] == b["history"]


def test_final_selection_uses_validation_only_and_never_loads_the_test_set(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the held-out test set must not be loaded during evolution")

    monkeypatch.setattr(sst2_data, "load_final_test", boom)
    monkeypatch.setattr(sst2_data, "_load_hf_split", boom)
    llm, out = run()
    top = [p for p, _ in out["ranking"]]
    final = select_final(llm, top, TEXTS, LABELS, top_k=3)
    assert len(final) == 3 and final == sorted(final, key=lambda x: -x[1])