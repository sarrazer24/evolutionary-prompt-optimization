# Notebooks

Create one notebook per member. Keep notebooks for exploration and demonstrations; put reusable functions in `src/`. On Kaggle, clone the GitHub repository or attach it as a Kaggle input, then run from the repository root.

---

## Notebook Index

| # | File | Member | Description |
|---|------|--------|-------------|
| 01 | `01_sst2_dataset_exploration.ipynb` | 1 | SST-2 dataset EDA, class balance, text statistics |
| 02 | `02_prompt_evaluation_demo.ipynb` | 1 | Mock evaluation pipeline demo |
| 03 | `03_llm_real_run.ipynb` | 2 | Real LLM inference run, model loading verification |
| 04 | `04_model_check_and_demo.ipynb` | 2 | Model selection, prompt operators demo |
| 05 | `05_genetic_algorithm_experiment.ipynb` | 3 | **Main GA experiment** (seed 42): evolution, manual comparison, validation |
| 06 | `06_multi_seed_experiment.ipynb` | 3 | Multi-seed GA runs (seeds 7 & 123), multi-seed analysis |
| 07 | `07_experiment_A_manual_prompts.ipynb` | 4 | **Experiment A**: evaluate all 8 manual prompts on opt + val |
| 08 | `08_experiment_B_random_baseline.ipynb` | 4 | **Experiment B**: random prompt generation baseline (no evolutionary pressure) |
| 09 | `09_experiment_C_three_way_comparison.ipynb` | 4 | **Experiment C**: three-way A/B/C comparison table, McNemar tests |

---

## Run Order on Kaggle

```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09
```

Notebook 09 depends on the outputs of 05/06, 07, and 08.

---

## Fair-Comparison Protocol (enforced by Member 4)

| Property | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` rev `989aa798` |
| Decoding | Greedy, `max_new_tokens=8` |
| Opt sample | 1 000 examples, `stratified_sample`, seed 42 |
| Val sample | 1 000 examples, `stratified_sample`, seed 42 |
| Primary metric | Macro F1 |
| Test set | **Never used** |
