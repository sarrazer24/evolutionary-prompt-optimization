"""Member 2 demo: the LLM interface and the prompt components, shown on sample reviews.

It does NOT run any evolutionary search (selection, elitism, generations belong to Member 3).
It shows: (1) classifying sample reviews, (2) scoring the manual baselines with Member 1's
evaluator, (3) generating an initial population, mutations and crossovers.

Run from the repository root:
    python -m scripts.member2_demo               # CPU, MockLLM (synthetic numbers!)
    python -m scripts.member2_demo --real        # Kaggle GPU, real Qwen model
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluation import PredictionCache, evaluate_prompt, results_to_dataframe  # noqa: E402
from src.prompts.baseline_prompts import get_baseline_prompt_texts, get_baseline_prompts  # noqa: E402
from src.prompts.initial_population import generate_initial_prompts  # noqa: E402
from src.prompts.operators import crossover_prompts, mutate_prompt  # noqa: E402


def run(llm, texts, labels, *, n_population=12, n_mutations=3, n_crossovers=2, seed=42, log=print) -> dict:
    rng = random.Random(seed)
    llm.set_seed(seed)
    baselines = get_baseline_prompts()

    log("== 1. Classifying sample reviews with the first baseline prompt ==")
    sample_texts = list(texts[:6])
    raw = llm.predict_raw_batch(baselines[0][1], sample_texts)
    parsed = llm.predict_batch(baselines[0][1], sample_texts)
    for text, true, r, label in zip(sample_texts, labels[:6], raw, parsed):
        log(f"  true={'positive' if true else 'negative':8} raw={r!r:12} parsed={label:9} | {text[:60]!r}")

    log("\n== 2. Manual baselines scored with Member 1's evaluator ==")
    cache = PredictionCache()
    results = [evaluate_prompt(p, texts, labels, llm.predict_raw_batch, model_id=llm.model_id,
                               inference_config=llm.inference_config, cache=cache, batch_size=100,
                               split_name="opt") for _, p in baselines]
    table = results_to_dataframe(results)
    table.insert(0, "name", [n for n, _ in baselines])
    log(table[["name", "f1_macro", "accuracy", "n_invalid"]].round(4).to_string(index=False))

    log(f"\n== 3. Initial population ({n_population} prompts: baselines first, then LLM-written) ==")
    population = generate_initial_prompts(llm, n_population, seed_prompts=get_baseline_prompt_texts(), rng=rng)
    for i, p in enumerate(population[len(baselines):], len(baselines) + 1):
        log(f"  [{i}] {p}")

    log("\n== 4. Mutation (a result equal to the parent means the operator fell back) ==")
    seeds = get_baseline_prompt_texts()
    mutations = [(seeds[i], mutate_prompt(llm, seeds[i], rng=rng)) for i in range(n_mutations)]
    for parent, child in mutations:
        log(f"  PARENT: {parent}\n  CHILD : {child}{'   <-- fell back' if child == parent else ''}\n")

    log("== 5. Crossover ==")
    crossovers = []
    for _ in range(n_crossovers):
        a, b = rng.sample(seeds, 2)
        crossovers.append((a, b, crossover_prompts(llm, a, b)))
    for a, b, child in crossovers:
        log(f"  A    : {a}\n  B    : {b}\n  CHILD: {child}{'   <-- fell back' if child == a else ''}\n")
    return {"table": table, "population": population, "mutations": mutations, "crossovers": crossovers}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true", help="use the real LLM (needs a GPU; run on Kaggle)")
    ap.add_argument("--n", type=int, default=200, help="number of reviews from opt used for scoring")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from src.data.sst2_data import load_saved_splits, stratified_sample
    sample = stratified_sample(load_saved_splits("data/processed").opt, args.n, seed=args.seed)
    if args.real:
        from src.llm import LLMInterface
        llm = LLMInterface().load_model()
        print(json.dumps(llm.info(), indent=2))
    else:
        from src.llm import MockLLM
        llm = MockLLM(oracle=dict(zip(sample.text, sample.label)), seed=args.seed)
        print("*** MockLLM: numbers below are SYNTHETIC and must never be reported ***\n")
    run(llm, sample.text.tolist(), sample.label.tolist(), seed=args.seed)


if __name__ == "__main__":
    main()