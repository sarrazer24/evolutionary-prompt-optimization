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
| Model | Hugging Face instruction-tuned causal language model |
| Initial model candidate | Qwen2.5-3B-Instruct |
| Framework | Python and Hugging Face Transformers |
| Compute environment | Kaggle GPU |
| Fine-tuning | Not performed |

The classifier model and inference configuration are kept consistent across prompt comparisons to isolate the effect of prompt optimization.

The initial model choice is subject to hardware compatibility testing.

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
