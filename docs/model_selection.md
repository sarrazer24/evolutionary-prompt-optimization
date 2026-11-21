# Model selection (Member 2)

**Decision: `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
(pinned as the default in `src/llm/config.py`), run in float16 with greedy decoding.**

## 1. Requirements

* Free and open weights; no paid API; no gated access (a token and licence acceptance would block teammates).
* Runs on the only available GPU: Kaggle (2 x Tesla T4, 14.6 GiB each, compute capability 7.5, so float16 and
  not bfloat16). One T4 is used.
* Fast enough for many prompt evaluations: each candidate prompt is scored on about 1000 reviews.
* The same model classifies reviews and writes/mutates prompts.

## 2. Candidates checked (licence and access read from the Hugging Face Hub on 2026-11-21)

| Model | Licence | Gated | Size (safetensors) | Verdict |
|---|---|---|---|---|
| Qwen/Qwen2.5-1.5B-Instruct | apache-2.0 | no | 2.9 GB | **selected** |
| Qwen/Qwen2.5-3B-Instruct | other (Qwen Research Licence) | no | 5.7 GB | measured, not selected |
| microsoft/Phi-3.5-mini-instruct | mit | no | 7.1 GB | measured, not selected |
| Qwen/Qwen2.5-7B-Instruct | apache-2.0 | no | 14.2 GB | rejected: does not fit one 14.6 GiB T4 in fp16 with room for the KV cache |
| meta-llama/Llama-3.2-3B-Instruct | llama3.2 | manual | 6.0 GB | rejected: gated |
| google/gemma-2-2b-it | gemma | manual | 4.9 GB | rejected: gated |

## 3. Measurement

Same 8 manual baseline prompts, same 500 reviews (stratified sample of the optimisation split, seed 42),
same evaluator (Member 1's `evaluate_prompt`, strict parsing), greedy decoding, float16, batch size 32,
one run on a Kaggle T4. Raw data: `experiments/results/model_comparison.csv`.

| | Qwen2.5-1.5B | Qwen2.5-3B | Phi-3.5-mini |
|---|---|---|---|
| GPU memory | 2.9 GiB | 5.9 GiB | 7.1 GiB |
| Mean macro F1 over the 8 baselines | 0.926 | 0.865 | 0.927 |
| Best / worst baseline F1 | 0.952 / 0.889 | 0.938 / 0.712 | 0.940 / 0.902 |
| Invalid answers (of 4000) | 7 | 17 | 208 |
| Reviews per second | 66 | 35 | 21 |
| LLM calls to obtain 6 valid new prompts | 8 | 6 | 6 |
| Fallbacks in 6 mutation requests | 1 | 0 | 0 |

(The 1.5B load time of 9 s excludes the download, which was already cached; its first download took about 45 s.)

## 4. Decision rule and outcome

Rule, fixed before the numbers were seen: keep Qwen2.5-1.5B unless another candidate has a mean baseline F1 at
least 3 points higher (a non-Apache licence would additionally need the team's agreement).

* Phi-3.5-mini: +0.1 point, a tie within noise, but 3x slower, 1.4x more memory and about 5% invalid answers.
* Qwen2.5-3B: 6 points lower on average, 2x slower, non-Apache licence.
* Therefore **Qwen2.5-1.5B-Instruct is kept.**

## 5. Limits of this comparison

* One run, 500 reviews, 8 prompts, one sampling seed, float16 only. Differences of about 1 point are not meaningful
  (one standard error on 500 reviews at ~92% accuracy is about 1.2 points).
* The causes of Phi-3.5's invalid answers and of the 3B's weak worst-case prompt were not investigated.
* A model that is less sensitive to prompt wording (smaller best-to-worst spread, e.g. Phi) may leave less for
  prompt optimisation to gain; this is an observation, not a tested claim.
* Larger or instruction-stronger models might be better in absolute terms; they were excluded by the constraints
  above (memory, licence/access, speed), not shown to be worse.

## 6. Reproducing

```python
from src.llm import LLMInterface            # default config = the pinned model
llm = LLMInterface().load_model()
print(llm.info())                           # model, revision, GPU, library versions
```
Library versions used: torch 2.11.0+cu130, transformers 5.20.2, Python 3.13.