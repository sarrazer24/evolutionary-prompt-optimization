"""Integration DEMO: a tiny genetic loop that proves Member 1 + Member 2 pieces fit together.

This is NOT the project's Genetic Algorithm (that is Member 3's job). It only shows, with
real calls, HOW to use the interfaces:  initial population -> fitness (Member 1) ->
tournament selection -> crossover / mutation (Member 2) -> elitism -> final re-ranking on val.
The held-out test set is never loaded here.

Run from the repository root:
    python -m scripts.ga_integration_demo                # CPU, MockLLM (synthetic numbers!)
    python -m scripts.ga_integration_demo --real         # Kaggle GPU, real Qwen model
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluation import PredictionCache, evaluate_prompt  # noqa: E402
from src.prompts.baseline_prompts import get_baseline_prompt_texts  # noqa: E402
from src.prompts.initial_population import generate_initial_prompts  # noqa: E402
from src.prompts.operators import crossover_prompts, mutate_prompt  # noqa: E402


def make_fitness(llm, texts, labels, cache=None, split_name="opt"):
    """Return (fitness_fn, scores). fitness(prompt) -> macro F1; each prompt is scored ONCE."""
    scores: dict[str, float] = {}

    def fitness(prompt: str) -> float:
        if prompt not in scores:
            result = evaluate_prompt(prompt, texts, labels, llm.predict_raw_batch,
                                     model_id=llm.model_id, inference_config=llm.inference_config,
                                     cache=cache, batch_size=100, split_name=split_name)
            scores[prompt] = result["metrics"]["f1_macro"]
        return scores[prompt]

    return fitness, scores


def tournament(rng, population, fitness, k=3):
    return max(rng.sample(population, min(k, len(population))), key=fitness)


def run_demo(llm, fit_texts, fit_labels, *, pop_size=10, generations=3, n_elite=2,
             crossover_rate=0.5, seed=42, cache=None, log=print) -> dict:
    """Tiny GA loop. Returns the final ranking and per-generation history."""
    rng = random.Random(seed)
    llm.set_seed(seed)
    fitness, scores = make_fitness(llm, fit_texts, fit_labels, cache)

    population = generate_initial_prompts(llm, pop_size, seed_prompts=get_baseline_prompt_texts(), rng=rng)
    history = []
    for gen in range(generations + 1):
        ranked = sorted(population, key=fitness, reverse=True)
        values = [fitness(p) for p in ranked]
        history.append({"generation": gen, "best": values[0], "mean": sum(values) / len(values),
                        "distinct_prompts_scored": len(scores), "texts_classified": llm.stats["classified_texts"]})
        log(f"gen {gen}: best={values[0]:.4f} mean={history[-1]['mean']:.4f} "
            f"| distinct prompts scored={len(scores)} | texts classified={llm.stats['classified_texts']}")
        if gen == generations:
            break

        children, attempts = [], 0
        while n_elite + len(children) < pop_size and attempts < 10 * pop_size:
            attempts += 1
            if rng.random() < crossover_rate and len(population) > 1:
                a, b = tournament(rng, population, fitness), tournament(rng, population, fitness)
                child, parents = crossover_prompts(llm, a, b), (a, b)
            else:
                a = tournament(rng, population, fitness)
                child, parents = mutate_prompt(llm, a, rng=rng), (a,)
            if child in parents or child in scores or child in children:
                continue            # operator fell back to a parent, or prompt already evaluated
            children.append(child)
        population = ranked[:n_elite] + children
        population += [p for p in ranked[n_elite:] if p not in population][:pop_size - len(population)]

    final = sorted(population, key=fitness, reverse=True)
    return {"ranking": [(p, fitness(p)) for p in final], "history": history, "scores": dict(scores)}


def select_final(llm, candidates, val_texts, val_labels, cache=None, top_k=3):
    """Re-rank the top candidates on the validation split (never on the test set)."""
    fitness, scores = make_fitness(llm, val_texts, val_labels, cache, split_name="val")
    ranked = sorted(candidates[:top_k], key=fitness, reverse=True)
    return [(p, scores[p]) for p in ranked]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", action="store_true", help="use the real LLM (needs a GPU; run on Kaggle)")
    ap.add_argument("--pop-size", type=int, default=10)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--n-fit", type=int, default=300, help="size of the fixed fitness sample from opt")
    ap.add_argument("--n-val", type=int, default=500, help="size of the validation sample for final re-ranking")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache", default=None, help="optional path of a persistent JSONL prediction cache")
    ap.add_argument("--out", default=None, help="optional JSON file for the results")
    args = ap.parse_args()

    from src.data.sst2_data import load_saved_splits, stratified_sample
    splits = load_saved_splits("data/processed")
    fit = stratified_sample(splits.opt, args.n_fit, seed=args.seed)
    val = stratified_sample(splits.val, args.n_val, seed=args.seed)

    if args.real:
        from src.llm import LLMInterface
        llm = LLMInterface().load_model()
        print(json.dumps(llm.info(), indent=2))
    else:
        from src.llm import MockLLM
        oracle = dict(zip(fit.text, fit.label)) | dict(zip(val.text, val.label))
        llm = MockLLM(oracle=oracle, seed=args.seed)
        print("*** MockLLM: numbers below are SYNTHETIC and must never be reported ***")

    cache = PredictionCache(args.cache) if args.cache else PredictionCache()
    t0 = time.time()
    out = run_demo(llm, fit.text.tolist(), fit.label.tolist(), pop_size=args.pop_size,
                   generations=args.generations, seed=args.seed, cache=cache)
    top = [p for p, _ in out["ranking"]]
    final = select_final(llm, top, val.text.tolist(), val.label.tolist(), cache=cache)

    print(f"\nTop prompts re-ranked on the validation sample (took {time.time() - t0:.0f}s in total):")
    for prompt, f1 in final:
        print(f"  val F1={f1:.4f}  {prompt}")
    print("\nThe held-out test set was NOT touched. Evaluate the single final prompt on it once, at the very end,"
          "\nwith load_final_test(confirm=True).")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"history": out["history"], "final_val": final,
                                              "mock": not args.real}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()