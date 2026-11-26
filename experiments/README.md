# Member 4: Experimental Design, Baselines & Results

This directory stores Member 4's experiment configurations, reusable random-baseline code, and comparison-ready results.

For the full controlled protocol, evaluation-budget justification, literature review, and reproducibility documentation, see [`docs/experimental_methodology.md`](../docs/experimental_methodology.md).

## Contents

- `configs/` — Reproducible JSON configurations for Experiments A, B, and C:
  - `experiment_A_manual.json`
  - `experiment_B_random.json`
  - `experiment_C_comparison.json`
- `random_baseline.py` — Pure LLM-based random prompt generator (`seed_prompts=None`, no selection/crossover/mutation, disjoint from Experiment A's manual prompts).
- `results/` — Comparison-ready CSV/JSON outputs and summary plots:
  - `experiment_A_manual/` (from `notebooks/07_experiment_A_manual_prompts.ipynb`)
  - `experiment_B_random/` (from `notebooks/08_experiment_B_random_baseline.ipynb`)
  - `experiment_C_comparison/` (from `notebooks/09_experiment_C_three_way_comparison.ipynb`)

## Rules

- Do not commit large raw datasets, model weights, prediction caches, or private credentials.
- Record model ID, model revision, data split (`data/processed`), seed, decoding parameters, candidate budget, number of calls, and runtime for every run.
- Never evaluate on the held-out final test set (`load_final_test`) during optimization or baseline selection.
