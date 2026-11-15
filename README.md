# Evolutionary Prompt Optimization for LLM-Based Sentiment Analysis

A four-member Advanced AI course project investigating whether a Genetic Algorithm (GA) can discover prompts that improve LLM-based sentiment classification compared with manually written prompts.

## Project constraints

- Free and open-source software only.
- Kaggle is the only GPU environment.
- No paid APIs, hosted inference subscriptions, or paid GPU services.
- No model fine-tuning.
- Use one fixed classifier model and consistent inference settings across comparisons.
- Keep the final test set completely separate from prompt optimization.

## Team ownership

| Member | Responsibility | Main code |
|---|---|---|
| Member 1 | Dataset preparation and evaluation | `src/data/`, `src/evaluation/` |
| Member 2 | LLM integration and prompt engineering | `src/llm/`, `src/prompts/` |
| Member 3 | Genetic Algorithm | `src/genetic_algorithm/` |
| Member 4 | Experimental design and analysis | `experiments/`, `notebooks/` |

## Repository layout

```text
data/                  Dataset storage (do not commit large datasets)
notebooks/             Kaggle/Jupyter experiments, one notebook per member
src/data/              Dataset loading and splitting
src/llm/               Model loading and inference
src/prompts/           Baselines, population generation, mutation, crossover
src/genetic_algorithm/ Population, selection, evolutionary loop
src/evaluation/        Metrics and prompt evaluation
experiments/           Experiment configurations, logs and results
docs/                  Architecture, interfaces and contribution rules
tests/                 Unit tests
```

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR-ORG-OR-USERNAME>/evolutionary-prompt-optimization.git
cd evolutionary-prompt-optimization
```

### 2. Create an environment (local CPU development only)

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For Kaggle, enable Internet if your course rules permit downloading public Hugging Face model files, enable a GPU accelerator, then install the requirements in a notebook cell if needed.

### 4. Run unit tests

```bash
python -m pytest -q
```

### 5. Run the LLM smoke test

```bash
python scripts/smoke_test.py
```

The model download and inference require Internet access on first run. Set `HF_MODEL_ID` if you want to test a different compatible Hugging Face causal instruction model.

## Initial model choice

The initial candidate is `Qwen/Qwen2.5-3B-Instruct`. This is a starting point, not a final decision. Verify that it fits the actual Kaggle GPU assigned to your account. If it does not, test a smaller model and document the change. Do not compare methods using different classifier models.

## Important prompt interface contract

A candidate prompt is the **system instruction only**. The review text is supplied separately as the user message. Every candidate prompt must instruct the model to return exactly one label:

- `positive`
- `negative`

This avoids mixing the review text into the evolving instruction and makes prompts comparable.

The model adapter exposes:

```python
load_model()
predict_sentiment(prompt, text)
predict_batch(prompt, texts)
generate_text(system_prompt, user_prompt, max_new_tokens=...)
```

The prompt module exposes:

```python
generate_initial_prompts(llm, n, seed_prompts=None)
mutate_prompt(llm, prompt)
crossover_prompts(llm, parent1, parent2)
```

Member 1's evaluation module should accept a prediction callable, rather than importing a particular LLM implementation. Member 3 should call Member 2's genetic operators through the interfaces documented in `docs/interfaces.md`.

## Reproducibility and data leakage

- Use a fixed random seed for data splits and GA runs where possible.
- Never use held-out test examples or labels to generate, mutate, select or tune prompts.
- Fit/select prompts using only the agreed optimization split; use validation for model selection if the team protocol requires it.
- Evaluate the selected final prompt once on the untouched test set.
- Log model ID, tokenizer/model revision if available, decoding parameters, dataset split, seed and package versions.
- Do not commit model weights, Kaggle secrets, API tokens, private datasets or large generated artifacts.

## Collaboration

Create one branch per member. See `docs/contribution_guide.md`.
