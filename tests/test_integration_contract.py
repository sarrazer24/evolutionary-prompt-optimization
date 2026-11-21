"""Contract tests for the interfaces between Member 1 (evaluation) and Member 2 (LLM, prompts).

Runs on CPU with MockLLM. It tests ONLY Member 1 + Member 2 code (no genetic algorithm).
"""
from scripts.member2_demo import run
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
RULE = "Answer with exactly one word: positive or negative."


def make_llm(seed=0):
    return MockLLM(oracle=ORACLE, seed=seed)


# ------------------------------------------------------------------ MockLLM itself
def test_mock_has_the_llm_interface_surface():
    llm = make_llm()
    assert llm.model_id.startswith("MOCK") and isinstance(llm.inference_config, dict)
    raw = llm.predict_raw_batch("Classify. " + RULE, TEXTS[:5])
    assert len(raw) == 5 and all(isinstance(r, str) for r in raw)
    assert llm.predict_batch("Classify. " + RULE, TEXTS[:5]) \
        == [r if r in ("positive", "negative") else "invalid" for r in raw]


def test_mock_classification_is_deterministic_and_order_independent():
    prompt = "Judge the tone. " + RULE
    a = make_llm().predict_raw_batch(prompt, TEXTS)
    b = make_llm().predict_raw_batch(prompt, list(reversed(TEXTS)))
    assert a == list(reversed(b))


def test_mock_prompts_differ_in_quality():
    llm = make_llm()
    assert len({llm.error_rate(p) for p in get_baseline_prompt_texts()}) > 4


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


# ------------------------------------------------------------------ the Member 2 demo
def test_member2_demo_runs_and_never_loads_the_test_set(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the held-out test set must not be loaded by the demo")

    monkeypatch.setattr(sst2_data, "load_final_test", boom)
    monkeypatch.setattr(sst2_data, "_load_hf_split", boom)
    out = run(make_llm(), TEXTS, LABELS, log=lambda *_: None)
    assert len(out["table"]) == 8 and len(out["population"]) == 12
    assert all(check_prompt(p) is None for p in out["population"])
    assert len(out["mutations"]) == 3 and len(out["crossovers"]) == 2