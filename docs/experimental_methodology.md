# Experimental Methodology, Baselines & Reproducibility (Member 4)

**Project:** Evolutionary Prompt Optimization for LLM-Based Sentiment Analysis: A Genetic Algorithm Approach  
**Role:** Member 4 — Experimental Design and Baseline Implementation  
**Scope:** Controlled experimental design, fair comparison protocol, non-evolutionary baselines, configurations, and reproducibility documentation. *(Statistical interpretation and final research discussion are reserved for Member 5.)*

---

## 1. Research Context & Literature Review

Discrete natural-language prompt optimization treats prompt engineering as a black-box search problem over text instructions without updating model weights. Four key lines of prior work inform our experimental design:

1. **Automatic Prompt Engineer (APE)** — Zhou et al. (ICLR 2023) cast instruction selection as black-box program/prompt synthesis: an LLM generates candidate instructions from input–output demonstrations or task specifications, scores them on a held-out optimization subset, and optionally resamples around top performers via Monte Carlo search without evolutionary recombination.
2. **Genetic Prompt Search (GPS)** — Xu et al. (EMNLP 2022) and **GrIPS** (Prasad et al., EACL 2023) demonstrate that gradient-free discrete search—either population-based genetic search (GPS) or greedy/beam edit-based search over phrase operations (GrIPS)—can refine discrete instructions for frozen language models using validation accuracy/F1 as fitness.
3. **EvoPrompt** — Guo et al. (ICLR 2024) directly connect LLMs with evolutionary algorithms (Genetic Algorithms and Differential Evolution), using the LLM itself as the semantic mutation and crossover operator over a population of discrete prompts while preserving evolutionary selection pressure.
4. **Promptbreeder** — Fernando et al. (2023) extend evolutionary prompt search by co-evolving both task-prompts and mutation-prompts in a self-referential evolutionary loop, highlighting how mutation operator diversity and population dynamics affect convergence.

### Positioning of Our Controlled Study

In EvoPrompt-style pipelines, improvements over manual prompts can come from **two distinct sources**:
1. **LLM-assisted prompt generation** (the generator LLM phrasing diverse instructions), or
2. **Evolutionary search pressure** (fitness-proportional/tournament selection, crossover, and mutation across generations).

To isolate these factors, Member 4 designs a **three-way controlled comparison** (`A` vs `B` vs `C`):
- **Experiment A vs C** tests whether GA beats conventional human prompt engineering.
- **Experiment B vs C** tests whether evolutionary search adds measurable value over generating the same budget of random candidate prompts without selection, crossover, or mutation (an APE-style unguided sampling baseline).

---

## 2. Three Experimental Approaches

| Experiment | Method | Notebook | Candidates & Operators | Purpose |
|---|---|---|---|---|
| **A** | Manual prompting | `notebooks/07_experiment_A_manual_prompts.ipynb` | 8 hand-crafted prompts authored by Member 2 (`src/prompts/baseline_prompts.py`) | Conventional human prompt-engineering baseline |
| **B** | Random prompt generation | `notebooks/08_experiment_B_random_baseline.ipynb` | 50 unique LLM-generated prompts (`experiments/random_baseline.py`), `seed_prompts=None`, **no** selection/crossover/mutation | Non-evolutionary search baseline isolating generation from evolution |
| **C** | Genetic Algorithm | `notebooks/05_genetic_algorithm_experiment.ipynb`, `06_multi_seed_experiment.ipynb`, `09_experiment_C_three_way_comparison.ipynb` | Population size 12, 10 generations, elitism 2, tournament size 3, crossover 0.70, mutation 0.30 across seeds `[42, 7, 123]` | Evolutionary prompt optimization (Member 3) |

---

## 3. Fair Comparison Protocol

All three experiments enforce identical controlled conditions:

| Control Dimension | Fixed Value Across A, B, and C | Source |
|---|---|---|
| **Classifier LLM** | `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | `src/llm/config.py` |
| **Classification Decoding** | Deterministic greedy decoding (`sample=False`, `max_new_tokens=8`, `batch_size=32`, `dtype=auto` -> `float16` on T4) | `src/llm/model_interface.py` |
| **Prompt Interface** | Candidate prompt as `SYSTEM` message; review text as `USER` message (`review-sentiment-v1`) | `docs/interfaces.md` |
| **Response Parsing** | Strict parsing to `{0: negative, 1: positive, -1: INVALID}`; invalid counts as wrong | `src/evaluation/evaluation.py` |
| **Optimization Split** | Canonical `data/processed/opt.csv` (53,871 rows); stratified subsample of `n = 1,000` (`seed = 42`, 442 negative / 558 positive) | `src/data/sst2_data.py` |
| **Validation Split** | Canonical `data/processed/val.csv` (13,468 rows: 5,955 negative, 7,513 positive), evaluated **once** after prompt selection | `data/processed/split_info.json` |
| **Final Test Set** | Official GLUE `validation` split (`load_final_test`) **never loaded or touched** during optimization or model/prompt selection | `src/data/sst2_data.py` |
| **Primary Metric** | Macro F1 (`f1_macro`); secondary: Accuracy, Macro Precision, Macro Recall, Invalid Rate | `src/evaluation/evaluation.py` |

---

## 4. Evaluation Budget Design & Justification

- **Experiment A (Manual):** `8` candidates × `1,000` optimization examples = `8,000` optimization text evaluations. The single top-ranked prompt on `opt` is evaluated once on the `13,468`-example `val` split.
- **Experiment C (GA, Member 3):** `POPULATION_SIZE = 12` over `GENERATIONS = 10` (`120` population slots; `2` elites preserved per generation). With `PredictionCache` deduplication across unchanged elites/fallbacks, Member 3's completed runs evaluated:
  - Seed `42`: **49** uncached prompt evaluations (`48,951` text classifications)
  - Seed `7`: **56** uncached prompt evaluations (`55,944` text classifications)
  - Seed `123`: **59** uncached prompt evaluations (`58,941` text classifications)
  - 3-seed mean: **54.7** uncached prompt evaluations (`54,612` text classifications)
- **Experiment B (Random Baseline, Member 4):** `N_RANDOM_CANDIDATES = 50` purely LLM-generated unique prompts × `1,000` optimization examples = **`50,000` text evaluations**.
  - This directly matches GA seed 42's compute budget (`48,951` vs `50,000` texts) without under-budgeting Experiment B.
  - Crucially, `seed_prompts=None` and manual-overlap filtering ensure that **0 of Member 2's 8 manual prompts** appear in Experiment B, keeping A and B strictly disjoint.

---

## 5. Coordination Across Team Members

- **Member 1 (`src/data/`, `src/evaluation/`):** Member 4 consumes `load_saved_splits("data/processed")`, `stratified_sample(..., n=1000, seed=42)`, `PredictionCache`, `evaluate_prompt`, and `save_results`. `load_final_test` is never invoked.
- **Member 2 (`src/llm/`, `src/prompts/`):** Member 4 consumes `ModelConfig`, `LLMInterface`, `get_baseline_prompts()`, `generate_initial_prompts`, and `prepare_candidate` / `check_prompt` / `similarity`.
- **Member 3 (`src/genetic_algorithm/`):** Member 4 matches Member 3's exact `data/processed/` sample (`seed=42`, `n=1000`), full `val.csv` (`13,468` rows), and pinned model revision, and collects all 3 GA seed outputs (`42`, `7`, `123`) in Notebook `09`.
- **Member 5 (Analysis & Discussion):** Notebook `09` exports comparison-ready tables (`three_way_comparison.csv/json`, `ga_multi_seed_comparison.csv/json`, `all_opt_candidates.csv`, `mcnemar_tests.json`, `master_experiment_config.json`) without making final research claims.

---

## 6. Reproducibility Checklist

1. **Configurations:** Stored under `experiments/configs/` (`experiment_A_manual.json`, `experiment_B_random.json`, `experiment_C_comparison.json`).
2. **Seeds:** Data split seed = `42`; opt sampling seed = `42`; random baseline generation seed = `42`; GA seeds = `[42, 7, 123]`.
3. **Environment:** Kaggle `2 × Tesla T4` (15.6 GB), Python 3.x, `requirements.txt` free open-source libraries (`torch`, `transformers>=4.45.0`, `accelerate`, `datasets`, `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `scipy`).
4. **Execution Order on Kaggle:** `07_experiment_A_manual_prompts.ipynb` → `08_experiment_B_random_baseline.ipynb` → `09_experiment_C_three_way_comparison.ipynb`.

---

## 7. Research References

- **APE:** Zhou, Y., Muresanu, A. I., Han, Z., Paster, K., Pitis, S., Chan, H., & Ba, J. (2023). *Large Language Models Are Human-Level Prompt Engineers*. ICLR 2023. (arXiv:2211.01910)
- **GPS:** Xu, H., Chen, Y., Du, Y., Shao, N., Wang, Y., Li, H., & Yang, Z. (2022). *GPS: Genetic Prompt Search for Efficient Few-Shot Learning*. EMNLP 2022.
- **GrIPS:** Prasad, A., Hase, P., Zhou, X., & Bansal, M. (2023). *GrIPS: Gradient-free, Edit-based Instruction Search for Prompting Large Language Models*. EACL 2023.
- **EvoPrompt:** Guo, Q., Wang, R., Guo, J., Li, B., Song, K., Tan, X., Liu, G., Bian, J., & Yang, Y. (2024). *Connecting Large Language Models with Evolutionary Algorithms Yields Powerful Prompt Optimizers*. ICLR 2024. (arXiv:2309.08532)
- **Promptbreeder:** Fernando, C., Banarse, D., Michalewski, H., Osindero, S., & Rocktäschel, T. (2023). *Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution*. arXiv:2309.16797.
