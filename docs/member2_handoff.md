# Member 2 -> Member 3 handoff: LLM integration and prompt engineering

Status: **done, tested on CPU (mock) and on a Kaggle T4 (real model).**
Member 2 delivers the LLM interface, the prompt templates and the genetic *operators* (initial prompts,
mutation, crossover). The Genetic Algorithm itself (population, selection, elitism, generations, logging)
is Member 3's work and is not included here. You never need to edit Member 2's or Member 1's code.

## 1. Files

| File | Purpose |
|---|---|
| `src/llm/config.py` | `ModelConfig` (model id, dtype, batch size, generation settings) |
| `src/llm/model_interface.py` | `LLMInterface`: loads the model once, classifies reviews, writes prompts |
| `src/llm/mock_llm.py` | `MockLLM`: same interface, no GPU, synthetic answers (for developing the GA) |
| `src/prompts/baseline_prompts.py` | 8 manually designed baseline prompts |
| `src/prompts/initial_population.py` | `generate_initial_prompts` |
| `src/prompts/operators.py` | `mutate_prompt`, `crossover_prompts` |
| `src/prompts/validation.py` | prompt cleaning, validation, similarity |
| `scripts/member2_demo.py` | runnable demo of Member 2's components on sample reviews (no evolutionary search) |
| `tests/test_llm_interface.py`, `tests/test_prompts*.py`, `tests/test_prompt_leaks_and_repair.py`, `tests/test_integration_contract.py` | all CPU-only |

## 2. Public API (exact signatures)

```python
# --- LLM (src/llm) ---------------------------------------------------------
LLMInterface(config: ModelConfig | None = None)
llm.load_model() -> LLMInterface              # call ONCE per session (idempotent)
llm.model_id: str                             # pass to evaluate_prompt (part of the cache key)
llm.inference_config: dict                    # pass to evaluate_prompt (part of the cache key)
llm.predict_raw_batch(prompt: str, texts: Iterable[str]) -> list[str | None]   # THE predict_fn
llm.predict_batch(prompt: str, texts) -> list[str]        # "positive" | "negative" | "invalid"
llm.predict_sentiment(prompt: str, text: str) -> str      # same labels, one review
llm.generate_text(system_prompt, user_prompt, max_new_tokens=None, *, sample=False,
                  temperature=None, top_p=None) -> str    # greedy unless sample=True
llm.set_seed(seed: int) -> None               # seeds SAMPLING (prompt writing); call once at the start of a run
llm.info() -> dict                            # model, GPU, library versions: log it with every run
llm.stats: dict                               # classified_texts, generate_calls, failed_generations, ...

MockLLM(oracle: dict[str, int] | None = None, seed: int = 0)   # same surface, synthetic numbers

# --- prompts (src/prompts) -------------------------------------------------
get_baseline_prompts() -> list[tuple[str, str]]               # (name, prompt)
get_baseline_prompt_texts() -> list[str]
generate_initial_prompts(llm, n: int, seed_prompts=None, *, rng: random.Random | None = None,
                         max_attempts_per_prompt: int = 5) -> list[str]
mutate_prompt(llm, prompt: str, *, rng: random.Random | None = None, strategy: str | None = None,
              max_attempts: int = 3) -> str
crossover_prompts(llm, parent1: str, parent2: str, *, max_attempts: int = 3) -> str
check_prompt(prompt: str) -> str | None       # None = valid, otherwise the reason
similarity(a: str, b: str) -> float           # word-level, 1.0 = identical

# --- Member 1 (what you call to score a prompt) ------------------------------
load_saved_splits("data/processed") -> Splits(.opt, .val, .info)      # NOTE the folder
stratified_sample(df, n, seed=42) -> DataFrame                         # fixed fitness sample
PredictionCache(path: str | Path | None = None)
evaluate_prompt(prompt, examples, labels, predict_fn, *, model_id, inference_config=None,
                cache=None, batch_size=None, split_name=None) -> dict   # fitness = result["metrics"]["f1_macro"]
```

## 3. Behaviour you can rely on

* A **prompt is a plain `str`**: the SYSTEM message. The review is sent separately as the USER message.
* Every prompt returned by Member 2 code is **single-line** (whitespace normalised), 30-800 characters,
  mentions both `positive` and `negative`, and tells the model to answer with exactly one word.
  The fixed output rule `Answer with exactly one word: positive or negative.` is part of the task
  definition, so it is appended automatically when a generated prompt forgot it: **the operators change the
  instruction, not the output contract.**
* **Classification is deterministic** (greedy, same settings for all prompts). We checked that batch sizes
  8, 32 and 64 give identical outputs, so batch size does not need to be in any cache key.
* **Sampling is used only** inside `generate_initial_prompts`, `mutate_prompt` and `crossover_prompts`.
* **Failures:**
  * `mutate_prompt` / `crossover_prompts` never raise because of a bad generation. After `max_attempts`
    they return the parent **unchanged** (`parent` for mutation, `parent1` for crossover).
    You can detect it with `child == parent` (or `child in (parent1, parent2)`).
  * `generate_initial_prompts` raises `ValueError` (bad `n` or an invalid seed) or `RuntimeError`
    (could not make `n` unique prompts; the message lists the rejection reasons). It is a one-time setup step.
  * `predict_raw_batch` returns `None` for one failed generation (Member 1's cache never stores it, so it is
    retried and counted invalid) and raises `RuntimeError` only if EVERY generation failed.
* **Caching:** `evaluate_prompt(..., cache=cache)` sends only unseen (prompt, review) pairs to the model.
  Use one `PredictionCache` for the whole run (give it a path to survive restarts).

## 4. Minimal example of the calls (runs on CPU; tested)

```python
import random
from src.data.sst2_data import load_saved_splits, stratified_sample
from src.evaluation.evaluation import PredictionCache, evaluate_prompt
from src.llm import MockLLM            # on Kaggle: from src.llm import LLMInterface; llm = LLMInterface().load_model()
from src.prompts.baseline_prompts import get_baseline_prompt_texts
from src.prompts.initial_population import generate_initial_prompts
from src.prompts.operators import crossover_prompts, mutate_prompt

splits = load_saved_splits("data/processed")
fit = stratified_sample(splits.opt, 300, seed=42)                  # ONE fixed sample for every candidate
llm = MockLLM(oracle=dict(zip(fit.text, fit.label)))
llm.set_seed(42)
rng = random.Random(42)
cache = PredictionCache()                                          # or PredictionCache("path/to/cache.jsonl")

def fitness(prompt: str) -> float:
    result = evaluate_prompt(prompt, fit.text, fit.label, llm.predict_raw_batch,
                             model_id=llm.model_id, inference_config=llm.inference_config,
                             cache=cache, batch_size=100, split_name="opt")
    return result["metrics"]["f1_macro"]

population = generate_initial_prompts(llm, 10, seed_prompts=get_baseline_prompt_texts(), rng=rng)
parent_a, parent_b = population[0], population[1]
child_1 = mutate_prompt(llm, parent_a, rng=rng)                    # == parent_a  -> operator fell back
child_2 = crossover_prompts(llm, parent_a, parent_b)               # == parent_a  -> operator fell back
```

How the pieces are used in the real GA (calling `fitness`, choosing parents, handling a fallback child,
generations, elitism) is Member 3's design; the snippet above only shows that the calls fit together.

A runnable demo of Member 2's own components on sample reviews:

```powershell
python -m scripts.member2_demo            # Windows or Kaggle, CPU, MockLLM, a few seconds
python -m scripts.member2_demo --real     # Kaggle GPU only, real model
```

## 5. Data protocol (defined by Member 1)

| Split | Size | Use |
|---|---|---|
| `opt` (80% of GLUE train) | 53,871 | GA fitness, via ONE fixed `stratified_sample(splits.opt, N, seed=42)`, the same for every candidate |
| `val` (20% of GLUE train) | 13,468 | choosing among the best prompts after evolution |
| test (official GLUE validation) | 872 | the single final prompt, **once**, at the very end: `load_final_test(confirm=True)` |

Never load the test set during evolution. Use the same `llm` (same model, same settings) for every
prompt that will be compared, manual or evolved.

## 6. Measured numbers (Kaggle T4, Qwen2.5-1.5B-Instruct, fp16, greedy; sample N=1000 from opt, seed 42)

| Baseline | macro F1 | invalid |
|---|---|---|
| baseline_criteria | 0.936 | 0 |
| baseline_fewshot | 0.921 | 7 |
| baseline_rules | 0.921 | 0 |
| baseline_direct | 0.919 | 0 |
| baseline_fragment | 0.916 | 0 |
| baseline_recommend | 0.912 | 0 |
| baseline_expert | 0.894 | 9 |
| baseline_negation | 0.877 | 1 |

* Speed: about 62-95 reviews/second; GPU memory 2.9-3.4 GiB.
* Writing one prompt (mutation, crossover or initial) is one short generation: negligible.
* Cost per prompt: about 11-27 s on 1000 reviews; about 3.5 minutes on the full `val` split (13,468 reviews).
  Identical (prompt, review) pairs are free once cached, so a prompt that is evaluated twice costs nothing the second time.
* Model-written prompts scored 0.868-0.943 on the same sample, i.e. **not systematically better than the
  baselines**.

## 7. Known limitations and decisions

1. **Fitness is noisy.** On 1000 reviews at ~92% accuracy one standard error is ~0.8 points. Differences below
   ~1.5 points between prompts are not reliable, and the best of many noisy scores is optimistic (winner's curse).
   A larger sample (1500-2000 reviews) reduces the noise at proportionally higher cost.
2. **Small headroom.** The best baseline is already ~93.6%, so improvements may be small and "no significant
   improvement" is a possible outcome; a paired significance test is advisable when comparing prompts.
3. **Invalid answers.** Prompts that mention "mixed" or "neutral" can make the model answer `mixed`, `neutral`
   or `critical`, which Member 1's strict parser scores as wrong. Fitness already penalises this.
4. **Operator quality is uneven.** The classifier and the prompt writer are the same 1.5B model. Children can be
   semantically poor (e.g. one mutation said "treat sarcastic statements as positive").
   Mutation children must differ from the parent (word similarity < 0.85) so superficial paraphrases are rejected.
5. **Reproducibility.** `llm.set_seed(seed)` seeds sampling, but GPU generation is not guaranteed to be bit-identical
   across hardware. `resolved_revision` is `None` under transformers 5: pin the model with
   `ModelConfig(revision="<commit sha>")` (the sha is printed by `huggingface_hub.model_info(model_id).sha`);
   it then becomes part of `llm.model_id` and of the cache key.
6. **One GPU used.** Kaggle gives two T4s; the code uses `cuda:0` only. Kaggle sessions are limited: keep the
   cache file in `/kaggle/working/outputs` and download it or save it as a Kaggle dataset between sessions.
7. `MockLLM` numbers are synthetic. Use it for development and tests only; never report them.

## 8. What Member 3 can start right now

The interfaces above are stable, so the GA can be developed against `MockLLM` on any computer (no GPU)
and switched to `LLMInterface` on Kaggle when ready. Population representation, selection, elitism,
generation management, stopping criteria, saving the best prompt per generation and the optimisation logs are
all Member 3's design. Member 2 can adjust an operator if the GA needs a different behaviour; ask.

Kaggle recipe (GPU on, Internet on):

```python
import os, subprocess, sys
REPO = "/kaggle/working/repo"
if not os.path.exists(REPO):
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/sarrazer24/evolutionary-prompt-optimization.git", REPO], check=True)
else:
    subprocess.run(["git", "-C", REPO, "pull", "--ff-only"], check=True)
os.chdir(REPO); sys.path.insert(0, REPO)

from src.llm import LLMInterface
llm = LLMInterface().load_model()          # ~45 s, 2.9 GiB
print(llm.info())
```

## 9. Checks to run before relying on this

```powershell
python -m pytest -q                         # Windows or Kaggle, CPU only, no model needed
python -m scripts.member2_demo              # shows the Member 2 components with the mock in a few seconds
```