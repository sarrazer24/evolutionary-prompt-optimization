# Results, Discussion, and Limitations

## Results

### Experimental scope

The reported experiments use SST-2 and the pinned model revision
`Qwen/Qwen2.5-1.5B-Instruct@989aa7980e4cf806f80c7fef2b1adb7bc71aa306` with
greedy decoding. Prompts are selected on a stratified optimization sample of
1,000 examples and evaluated on the full 13,468-example validation split. The
saved comparison configuration states that the final test set was not used.
The best manual prompt is `baseline_criteria`; the best of 50 random prompts is
`random_047`. The three GA runs use seeds 42, 7, and 123.

### Validation performance

| Selected prompt | Accuracy | Macro precision | Macro recall | Macro-F1 | Invalid outputs |
|---|---:|---:|---:|---:|---:|
| Best manual (`baseline_criteria`) | 0.9269 | 0.9250 | 0.9279 | 0.9262 | 0 |
| Best random (`random_047`) | 0.9321 | 0.9304 | 0.9328 | 0.9314 | 1 |
| GA, seed 42 | 0.9334 | 0.9327 | 0.9338 | 0.9331 | 13 |
| GA, seed 7 | 0.9260 | 0.9240 | 0.9283 | 0.9254 | 0 |
| GA, seed 123 | 0.9275 | 0.9258 | 0.9300 | 0.9271 | 4 |

On the validation set, GA seed 42 exceeds the best manual prompt by 0.70
Macro-F1 percentage points and the best random prompt by 0.18 points. This is
not a stable GA advantage across seeds: seeds 7 and 123 score 0.60 and 0.43
points below the best random prompt, respectively. Across the three GA seeds,
validation Macro-F1 has mean 0.9285, sample standard deviation 0.0041, and
range 0.9254–0.9331. This small-sample summary is descriptive, not a precise
estimate of the distribution of GA performance.

The best random prompt narrowly exceeds the GA seed-42 prompt on optimization
Macro-F1 (0.943479 versus 0.943434); validation reverses this ranking. The
reversal is consistent with selection and sampling variability and argues
against treating the top optimization score as conclusive evidence.

The available class-level confusion counts also support stratified
nonparametric bootstrap intervals for the individual manual and random
metrics, following the resampling approach introduced by Efron (1979). The
following 95% percentile intervals use 10,000 resamples within
each observed true class (random seed 42); they are not paired intervals for
the difference between methods.

| Method | Metric | Estimate | Stratified bootstrap 95% CI |
|---|---|---:|---:|
| Manual | Accuracy | 0.9269 | [0.9224, 0.9312] |
| Manual | Macro precision | 0.9250 | [0.9205, 0.9294] |
| Manual | Macro recall | 0.9279 | [0.9235, 0.9322] |
| Manual | Macro-F1 | 0.9262 | [0.9217, 0.9305] |
| Random best | Accuracy | 0.9321 | [0.9278, 0.9363] |
| Random best | Macro precision | 0.9304 | [0.9260, 0.9347] |
| Random best | Macro recall | 0.9328 | [0.9286, 0.9370] |
| Random best | Macro-F1 | 0.9314 | [0.9271, 0.9357] |

These intervals quantify resampling uncertainty conditional on the observed
class-conditional outcome counts. They do not account for prompt-selection
uncertainty and should not be used as an unpaired substitute for a paired
method comparison.

### Genetic-algorithm trajectories

All three runs improve their best recorded optimization fitness after
generation 0: the best fitness is first reached at generation 3 for seeds 42
and 7 and generation 6 for seed 123. Their generation-0 best scores are
0.9406, 0.9366, and 0.9364; their final best scores are 0.9434, 0.9415, and
0.9415. Population-average fitness generally rises but fluctuates across
generations. The fitness trajectories are optimization-set results and are
not independent validation evidence.

The seed-42 prompt first recorded as best at generation 3 is: “Is the text
positive or negative? Answer with exactly one word: positive or negative.”
The seed-7 prompt is: “Classify the sentiment as positive or negative: positive
or negative. Answer with exactly one word: positive or negative.” The seed-123
prompt is: “Read the text. Classify sentiment as positive or negative. Answer
with one word: positive or negative.”

### Paired significance tests

McNemar exact tests compare paired correct/incorrect outcomes on the same
13,468 validation examples (McNemar, 1947). The table applies the
Bonferroni-adjusted p-value, `min(7p, 1)`, across the seven comparisons saved
with the experiment outputs (Hochberg, 1988).

| Comparison | Discordances (method 1 only / method 2 only correct) | Exact p | Bonferroni p | Significant at 0.05? |
|---|---:|---:|---:|---|
| Manual vs random | 152 / 222 | 0.000347 | 0.002429 | Yes |
| Manual vs GA seed 42 | 148 / 236 | 0.000008 | 0.000056 | Yes |
| Random vs GA seed 42 | 111 / 129 | 0.272457 | 1.000000 | No |
| Manual vs GA seed 7 | 185 / 173 | 0.561053 | 1.000000 | No |
| Random vs GA seed 7 | 173 / 91 | 0.000001 | 0.000007 | Yes |
| Manual vs GA seed 123 | 176 / 184 | 0.712233 | 1.000000 | No |
| Random vs GA seed 123 | 161 / 99 | 0.000145 | 0.001015 | Yes |

These are tests of paired correctness disagreement, not tests of Macro-F1
differences. In particular, the GA42-versus-random result is not significant;
this is not evidence that the two methods are equivalent. The comparisons are
also dependent because they reuse the same validation examples and prompts.

### Evaluation and compute costs

| Method/run | Optimization text evaluations | Reported duration | Prompt-generation calls | Tokens |
|---|---:|---:|---:|---|
| Manual (8 prompts) | 8,000 | 278.45 s (optimization + validation) | 0 recorded | Not recorded |
| Random (50 prompts) | 50,000 | 803.94 s (generation + optimization + validation) | 92 recorded | Not recorded |
| GA seed 42 | 48,951 | 778.41 s (run scope not separated) | Not recorded | Not recorded |
| GA seed 7 | 55,944 | 932.42 s (run scope not separated) | Not recorded | Not recorded |
| GA seed 123 | 58,941 | 959.11 s (run scope not separated) | Not recorded | Not recorded |

Optimization classification budgets for the random baseline and GA seed 42
are similar. Runtime scopes are not fully comparable: the manual and random
durations include different phases, while GA timing does not identify phase
boundaries in the saved summary. The 92 random-baseline generation calls
include its generation procedure; comparable GA operator-call counts and
token consumption were not recorded. No claim about total token cost or
relative wall-clock efficiency is therefore justified.

## Discussion

### Does the GA improve over manual prompts?

For the selected seed-42 prompt on this validation split, yes: Macro-F1 is
0.9331 versus 0.9262 for the best manual prompt, and the paired correctness
difference is significant after the recorded seven-test Bonferroni adjustment.
The other two GA seeds do not significantly differ from manual according to
their saved McNemar tests. The evidence supports a result for the seed-42
prompt, not a uniform advantage across GA runs.

### Does the GA improve over random prompt generation?

The top GA seed has a slightly higher validation Macro-F1 than the top random
prompt, but the paired McNemar comparison is not significant. The other GA
seeds score below the random candidate and their McNemar tests significantly
favor random. Therefore, these results do not establish that evolutionary
selection and variation outperform random candidate generation at a similar
optimization evaluation budget.

### Generalization and overfitting

The full validation split is unseen during prompt search according to the
documented protocol, so GA42's higher score than manual is evidence of
generalization from the optimization sample to this validation split. It is
not evidence of generalization to a new corpus or independent final test set.
Only one optimization sample and one validation split are reported. Choosing
the best from 8 manual candidates, 50 random candidates, or a GA population
can also overfit the optimization sample through winner selection. The
optimization-to-validation ranking change between random and GA42 reinforces
this concern.

No final held-out test results are present in the repository. The final test
set is explicitly recorded as unused; a test score must not be inferred or
fabricated. A frozen method should be evaluated on that test set once, with
the result reported separately.

### Did crossover and mutation cause the improvements?

This cannot be determined from the available experiments. All three GA runs
improve their best optimization fitness after generation 0, but the initial
population already contains strong generated prompts in seeds 42 and 7, and
seed 123 begins with the best manual prompt. The logs do not record operator
counts, and there is no no-crossover/no-mutation ablation or matched
random-search control per seed. The post-generation-0 gains are descriptive
changes during the algorithm run, not causal estimates of crossover or
mutation effects.

## Limitations and missing evidence

1. **Three seeds only:** too few to precisely characterize run-to-run
   variability.
2. **No per-example predictions for GA:** available GA summaries contain
   aggregate metrics but no class-level confusion counts or prediction arrays.
   The current artifacts allow validation confusion matrices for manual and
   random only; a GA confusion matrix cannot be recovered from aggregate
   Macro-F1, precision, and recall alone.
3. **Paired bootstrap inputs missing:** paired prediction vectors are not
   saved. A valid paired bootstrap confidence interval for Macro-F1
   differences cannot be calculated from the reported aggregate metrics or
   McNemar discordance counts. Individual stratified bootstrap intervals are
   reported only for manual and random, for which class-level confusion
   counts are available; no GA or paired-difference bootstrap interval is
   claimed.
4. **No test results:** held-out final test was not used and no test outputs
   exist in the supplied results.
5. **No component ablation:** crossover and mutation effects cannot be
   isolated.
6. **Incomplete compute telemetry:** GA generation-call counts and token
   consumption are absent; run-time scopes differ.
7. **Multiple comparisons:** Bonferroni adjustment is reported, but the tests
   share methods and examples and remain statistically dependent.
8. **Single model and dataset:** claims are limited to this SST-2 setup and
   pinned Qwen model.

## Conclusion and future work

The best GA prompt beats the best manual prompt on the supplied validation
set, but the GA does not demonstrate a statistically significant advantage
over the best random prompt, and performance varies materially across three
seeds. The evidence is therefore promising for prompt search but insufficient
to conclude that the genetic algorithm robustly improves on random generation.

Future experiments should save per-example predictions for every selected
prompt; run more independent GA and random-search seeds; add matched ablations
without crossover and without mutation; instrument operator calls, token
usage, and phase-specific runtime; and evaluate a frozen selection procedure
once on the untouched final test set. Bootstrap intervals for paired
Macro-F1 differences should be computed only after paired per-example
predictions are available.

## Reproducible analysis outputs

Run `python experiments/analyze_member5_results.py` from the repository root.
The script writes CSV tables and PNG figures to
`experiments/results/member5_analysis/`. It reads existing experiment
artifacts only and does not invoke a model or modify raw results.

The generated figures are:

- [GA best and mean fitness by generation](../experiments/results/member5_analysis/ga_fitness_by_generation.png)
- [Selected prompts' validation Macro-F1](../experiments/results/member5_analysis/validation_macro_f1.png)
- [Available manual and random confusion counts](../experiments/results/member5_analysis/available_confusion_matrices.png)
- [Evaluation and reported runtime costs](../experiments/results/member5_analysis/computational_cost.png)

The individual manual/random bootstrap intervals are in
`experiments/results/member5_analysis/bootstrap_ci_available_methods.csv`.

## References

- Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The
  Annals of Statistics, 7*(1), 1–26.
  https://doi.org/10.1214/aos/1176344552
- Hochberg, Y. (1988). A sharper Bonferroni procedure for multiple tests of
  significance. *Biometrika, 75*(4), 800–802.
  https://doi.org/10.1093/biomet/75.4.800
- McNemar, Q. (1947). Note on the sampling error of the difference between
  correlated proportions or percentages. *Psychometrika, 12*(2), 153–157.
  https://doi.org/10.1007/BF02295996
