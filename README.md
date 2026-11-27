# Evolutionary Prompt Optimization for LLM-Based Sentiment Analysis

An experimental study of **Genetic Algorithms for automated prompt optimization** in Large Language Model (LLM)-based sentiment classification.

## Overview

This project investigates whether evolutionary optimization can discover effective prompts for sentiment analysis, potentially outperforming manually designed instructions.

Rather than modifying model weights through fine-tuning, the approach evolves natural-language prompts using a Genetic Algorithm (GA). Candidate prompts are evaluated on a sentiment classification task, and the most effective candidates are iteratively improved through selection, mutation, and crossover.

The objective is to explore how automated prompt optimization can influence classification performance while keeping the underlying language model fixed.

## Research Objectives

The project focuses on the following questions:

- Can a Genetic Algorithm discover prompts that outperform manually written baselines?
- Which evolutionary operations contribute most to prompt improvement?
- How does prompt evolution affect classification performance?
- Can the optimization process remain reproducible and computationally accessible?

## Methodology

The proposed workflow consists of four main stages:

**1. Baseline Prompt Design**

Establish manually written prompts to serve as reference points for evaluating the effectiveness of evolutionary optimization.

**2. Initial Population Generation**

Generate an initial population of candidate prompts using predefined seeds and, where appropriate, LLM-assisted generation.

**3. Evolutionary Optimization**

Apply Genetic Algorithm operators to evolve the prompt population:

- **Selection:** Retain promising candidates based on their evaluation scores.
- **Mutation:** Introduce modifications to existing prompts.
- **Crossover:** Combine elements from two parent prompts to generate new candidates.

**4. Evaluation**

Evaluate the resulting prompts using consistent classification settings and compare their performance against the manually designed baselines.

## Experimental Setup

| Component | Configuration |
|---|---|
| Task | Binary sentiment classification |
| Classification labels | Positive / Negative |
| Optimization method | Genetic Algorithm |
| Optimization target | Natural-language system prompts |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Pinned model revision | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| Framework | Python, PyTorch, Hugging Face Transformers |
| Compute environment | Kaggle GPU (Tesla T4) |
| Fine-tuning | Not performed (fixed frozen weights) |

The classifier model and inference configuration are kept consistent across prompt comparisons to isolate the effect of prompt optimization.

## Experimental Results

The project conducted a rigorous three-way comparison on the SST-2 benchmark using standardized splits (`data/processed/opt.csv` with $n=1,000$ examples for optimization; `data/processed/val.csv` with $N=13,468$ examples for full validation):

### Summary of Experiments

- **Experiment A (Manual Prompt Baselines):** Evaluates 8 human-crafted prompts reflecting different prompting styles (concise, role-based, few-shot style, criteria-based). Served as the reference baseline.
- **Experiment B (Random Prompt Generation Baseline):** Evaluates 50 independent LLM-generated prompts without evolutionary operators or manual seed contamination, matching the computational evaluation budget.
- **Experiment C (Genetic Algorithm & Controlled Comparison):** Evaluates prompts evolved through selection, single-point/two-point crossover, and mutation across three distinct random seeds (`42`, `7`, `123`).

### Verified Performance Comparison

| Method | Candidates / Text Evals | Best Prompt ID / Name | Optimization Macro F1 ($n=1,000$) | Validation Macro F1 ($N=13,468$) | Validation Accuracy ($N=13,468$) | McNemar vs Manual ($p$-value) |
|:---|:---:|:---|:---:|:---:|:---:|:---:|
| **Experiment A (Manual)** | 8 / 8,000 | `baseline_criteria` | 0.9364 | 0.9262 | 0.9269 | — |
| **Experiment B (Random)** | 50 / 50,000 | `random_047` | **0.9435** | 0.9314 | 0.9321 | $p = 0.00035$ (Sig) |
| **Experiment C: GA Seed 42** | 120 / 48,951 | Individual #44 | 0.9434 | **0.9331** | **0.9334** | $p = 8.0 \times 10^{-6}$ (Sig) |
| **GA Seed 7** | 120 / 55,944 | Individual #38 | 0.9415 | 0.9260 | 0.9263 | $p = 0.561$ (Non-sig) |
| **GA Seed 123** | 120 / 58,941 | Individual #75 | 0.9415 | 0.9267 | 0.9269 | $p = 0.712$ (Non-sig) |

*Key Findings:*
- **GA vs Manual:** The best evolved prompt (Seed 42) achieved a statistically significant improvement over the best manual baseline on the full validation split (+0.69 F1 points, McNemar exact binomial $p = 8.0 \times 10^{-6}$).
- **GA vs Random:** The best evolved prompt also achieved higher validation performance than the best pure random prompt (0.9331 vs 0.9314), though the pairwise difference between top single candidates was not statistically significant ($p = 0.272$).
- **Multi-Seed Variance:** Across three random seeds, GA performance ranged from 0.9260 to 0.9331 on validation, highlighting the necessity of multi-seed evaluations in prompt evolution studies.

Relevant executed notebooks and result directories:
- Experiment A: [`notebooks/07_experiment_A_manual_prompts.ipynb`](notebooks/07_experiment_A_manual_prompts.ipynb) & [`experiments/results/experiment_A_manual/`](experiments/results/experiment_A_manual/)
- Experiment B: [`notebooks/08_experiment_B_random_baseline.ipynb`](notebooks/08_experiment_B_random_baseline.ipynb) & [`experiments/results/experiment_B_random/`](experiments/results/experiment_B_random/)
- Experiment C: [`notebooks/09_experiment_C_three_way_comparison.ipynb`](notebooks/09_experiment_C_three_way_comparison.ipynb) & [`experiments/results/experiment_C_comparison/`](experiments/results/experiment_C_comparison/)
- Multi-seed GA Results: [`experiments/results/genetic_algorithm/`](experiments/results/genetic_algorithm/)

## Evaluation and Reproducibility

The experimental design separates prompt optimization from final evaluation to prevent data leakage.

- A dedicated optimization split is used for evolving and selecting candidate prompts.
- Validation data may be used for model selection according to the experimental protocol.
- The final test set remains untouched throughout prompt optimization.
- The selected prompt is evaluated on the held-out test set after optimization.
- Random seeds, model configuration, decoding parameters, and evaluation settings are recorded wherever possible.

Performance is assessed using classification metrics, including accuracy, precision, recall, and F1-score.

## Repository Structure

```text
.
├── data/                  # Dataset storage
├── notebooks/             # Experimental notebooks
├── src/
│   ├── data/              # Dataset loading and splitting
│   ├── llm/               # Model loading and inference
│   ├── prompts/           # Prompt generation and operators
│   ├── genetic_algorithm/# Evolutionary optimization
│   └── evaluation/        # Classification metrics
├── experiments/           # Experiment configurations and results
├── docs/                  # Technical documentation
├── tests/                 # Unit tests
├── scripts/               # Utility and testing scripts
├── requirements.txt
└── README.md
```

## Getting Started

### Requirements

- Python 3.10+
- Git
- A Kaggle account for GPU-based experiments
- Internet access for downloading public Hugging Face model files, where permitted

### Installation

Clone the repository:

```bash
git clone https://github.com/sarrazer24/evolutionary-prompt-optimization.git
cd evolutionary-prompt-optimization
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run Tests

```bash
python -m pytest -q
```

### Verify Model Integration

```bash
python scripts/smoke_test.py
```

The first model execution may require downloading model weights and tokenizer files.

For GPU experiments, use Kaggle notebooks with a compatible GPU accelerator.

## Project Constraints

The experiments are designed around the following constraints:

- Free and open-source software.
- No paid inference APIs or hosted GPU services.
- Kaggle as the GPU execution environment.
- No model fine-tuning or weight updates.
- One fixed classifier model across experimental comparisons.
- Strict separation between optimization and final testing.

## Project Status

This repository contains the implementation and experimental workflow for investigating evolutionary prompt optimization in LLM-based sentiment analysis.

Experimental findings and quantitative results will be documented as the evaluation progresses.


## Acknowledgements

Developed as part of an Advanced Artificial Intelligence course project.
