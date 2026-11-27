"""Build reproducible Member 5 tables and figures from saved experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
OUTPUT = RESULTS / "member5_analysis"


def load_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    comparison_dir = RESULTS / "experiment_C_comparison"
    ga_dir = RESULTS / "genetic_algorithm"
    manual_dir = RESULTS / "experiment_A_manual"
    random_dir = RESULTS / "experiment_B_random"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    comparison = pd.read_csv(comparison_dir / "three_way_comparison.csv")
    ga_comparison = pd.read_csv(comparison_dir / "ga_multi_seed_comparison.csv")
    manual_val = pd.read_csv(manual_dir / "experiment_A_val_best.csv").iloc[0]
    random_val = pd.read_csv(random_dir / "experiment_B_val_best.csv").iloc[0]
    manual_summary = load_json(manual_dir / "experiment_A_summary.json")
    random_summary = load_json(random_dir / "experiment_B_summary.json")
    ga_summary = load_json(ga_dir / "multi_seed_summary.json")
    histories = {
        seed: load_json(ga_dir / f"seed_{seed}" / "history.json")
        for seed in (42, 7, 123)
    }

    manual = comparison.loc[comparison["experiment"] == "A"].iloc[0]
    random = comparison.loc[comparison["experiment"] == "B"].iloc[0]
    ga_validation = {
        int(row["prompt_name"].split("seed ")[1].rstrip(")")): row
        for row in ga_summary["validation_descriptive"]["rows"]
        if row["type"] == "Evolved"
    }
    ga_optimization = {int(row["seed"]): row for _, row in ga_comparison.iterrows()}

    rows = [
        {
            "method": "Manual",
            "seed": None,
            "prompt_id": manual["best_prompt_id"],
            "prompt": manual["best_prompt_text"],
            "optimization_macro_f1": manual["macro_f1_opt"],
            "optimization_accuracy": manual["accuracy_opt"],
            "validation_accuracy": manual_val["accuracy"],
            "validation_precision_macro": manual_val["precision_macro"],
            "validation_recall_macro": manual_val["recall_macro"],
            "validation_macro_f1": manual_val["f1_macro"],
            "invalid_validation": int(manual_val["n_invalid"]),
            "optimization_text_evaluations": int(manual["opt_text_evals"]),
            "reported_runtime_seconds": manual_summary["opt_elapsed_s"]
            + manual_summary["val_elapsed_s"],
            "runtime_scope": "opt + validation wall time",
            "prompt_generation_calls": manual_summary["llm_stats"]["generate_calls"],
            "token_count": None,
        },
        {
            "method": "Random best",
            "seed": None,
            "prompt_id": random["best_prompt_id"],
            "prompt": random["best_prompt_text"],
            "optimization_macro_f1": random["macro_f1_opt"],
            "optimization_accuracy": random["accuracy_opt"],
            "validation_accuracy": random_val["accuracy"],
            "validation_precision_macro": random_val["precision_macro"],
            "validation_recall_macro": random_val["recall_macro"],
            "validation_macro_f1": random_val["f1_macro"],
            "invalid_validation": int(random_val["n_invalid"]),
            "optimization_text_evaluations": int(random["opt_text_evals"]),
            "reported_runtime_seconds": random_summary["generation_elapsed_s"]
            + random_summary["opt_eval_elapsed_s"]
            + random_summary["val_elapsed_s"],
            "runtime_scope": "generation + opt + validation wall time",
            "prompt_generation_calls": random_summary["llm_stats"]["generate_calls"],
            "token_count": None,
        },
    ]

    for seed in (42, 7, 123):
        opt_row = ga_optimization[seed]
        val_row = ga_validation[seed]
        run = next(
            item for item in ga_summary["per_seed"] if int(item["ga_seed"]) == seed
        )
        prompt = load_json(ga_dir / f"seed_{seed}" / "best_individual.json")
        rows.append(
            {
                "method": "Genetic algorithm",
                "seed": seed,
                "prompt_id": f"ga_seed_{seed}_best",
                "prompt": prompt["prompt"],
                "optimization_macro_f1": float(opt_row["macro_f1_opt"]),
                "optimization_accuracy": float(opt_row["accuracy_opt"]),
                "validation_accuracy": val_row["accuracy"],
                "validation_precision_macro": val_row["precision_macro"],
                "validation_recall_macro": val_row["recall_macro"],
                "validation_macro_f1": val_row["macro_f1"],
                "invalid_validation": val_row["n_invalid"],
                "optimization_text_evaluations": int(opt_row["uncached_opt_texts"]),
                "reported_runtime_seconds": run["execution_time_s"],
                "runtime_scope": "GA run; phase boundary not separately logged",
                "prompt_generation_calls": None,
                "token_count": None,
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT / "performance_and_cost.csv", index=False)
    metrics[["method", "seed", "prompt_id", "prompt"]].to_csv(
        OUTPUT / "best_prompts.csv", index=False
    )

    history_rows = []
    for seed, history in histories.items():
        for generation in history:
            history_rows.append({"seed": seed, **generation})
    generation_metrics = pd.DataFrame(history_rows)
    generation_metrics.to_csv(OUTPUT / "ga_generation_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for seed in (42, 7, 123):
        seed_data = generation_metrics[generation_metrics["seed"] == seed]
        sns.lineplot(
            data=seed_data,
            x="generation",
            y="best_fitness",
            marker="o",
            label=f"Seed {seed}",
            ax=axes[0],
        )
        sns.lineplot(
            data=seed_data,
            x="generation",
            y="average_fitness",
            marker="o",
            label=f"Seed {seed}",
            ax=axes[1],
        )
    axes[0].set_title("Best optimization Macro-F1 by generation")
    axes[1].set_title("Mean population fitness by generation")
    for ax in axes:
        ax.set_xlabel("Generation")
        ax.set_ylabel("Optimization Macro-F1")
        ax.set_xticks(range(10))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "ga_fitness_by_generation.png", dpi=180)
    plt.close(fig)

    colors = ["#68778a", "#e69f00", "#0072b2", "#56b4e9", "#009e73"]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [
        "Manual",
        "Random best",
        "GA seed 42",
        "GA seed 7",
        "GA seed 123",
    ]
    y_positions = list(range(len(labels)))
    scores = metrics["validation_macro_f1"].to_numpy()
    ax.scatter(scores, y_positions, color=colors, s=80, zorder=3)
    for score, y_position in zip(scores, y_positions):
        ax.annotate(
            f"{score:.4f}",
            (score, y_position),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
        )
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(y_positions, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Validation Macro-F1 (full metric scale: 0 to 1)")
    ax.set_title("Validation Macro-F1 of selected prompts")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "validation_macro_f1.png", dpi=180)
    plt.close(fig)

    confusion_rows = []
    for method, result in (("Manual", manual_val), ("Random best", random_val)):
        confusion_rows.extend(
            [
                {
                    "method": method,
                    "actual_class": "Negative",
                    "predicted_negative": int(result["tn"]),
                    "predicted_positive": int(result["fp"]),
                    "invalid": int(result["neg_invalid"]),
                },
                {
                    "method": method,
                    "actual_class": "Positive",
                    "predicted_negative": int(result["fn"]),
                    "predicted_positive": int(result["tp"]),
                    "invalid": int(result["pos_invalid"]),
                },
            ]
        )
    confusion = pd.DataFrame(confusion_rows)
    confusion.to_csv(OUTPUT / "available_confusion_counts.csv", index=False)

    bootstrap_rows = []
    rng = np.random.default_rng(42)
    for method, result in (("Manual", manual_val), ("Random best", random_val)):
        negative_counts = np.array(
            [result["tn"], result["fp"], result["neg_invalid"]], dtype=int
        )
        positive_counts = np.array(
            [result["fn"], result["tp"], result["pos_invalid"]], dtype=int
        )
        n_negative = int(negative_counts.sum())
        n_positive = int(positive_counts.sum())
        negative_samples = rng.multinomial(
            n_negative, negative_counts / n_negative, size=10_000
        )
        positive_samples = rng.multinomial(
            n_positive, positive_counts / n_positive, size=10_000
        )
        tn, fp, _ = negative_samples.T
        fn, tp, _ = positive_samples.T
        total = n_negative + n_positive

        def safe_ratio(numerator, denominator):
            return np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator, dtype=float),
                where=denominator != 0,
            )

        precision_negative = safe_ratio(tn, tn + fn)
        precision_positive = safe_ratio(tp, tp + fp)
        recall_negative = tn / n_negative
        recall_positive = tp / n_positive
        f1_negative = safe_ratio(
            2 * precision_negative * recall_negative,
            precision_negative + recall_negative,
        )
        f1_positive = safe_ratio(
            2 * precision_positive * recall_positive,
            precision_positive + recall_positive,
        )
        samples = {
            "accuracy": (tn + tp) / total,
            "macro_precision": (precision_negative + precision_positive) / 2,
            "macro_recall": (recall_negative + recall_positive) / 2,
            "macro_f1": (f1_negative + f1_positive) / 2,
        }
        for metric, distribution in samples.items():
            low, high = np.quantile(distribution, [0.025, 0.975])
            bootstrap_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "estimate": float(result[
                        {
                            "accuracy": "accuracy",
                            "macro_precision": "precision_macro",
                            "macro_recall": "recall_macro",
                            "macro_f1": "f1_macro",
                        }[metric]
                    ]),
                    "bootstrap_ci_95_low": float(low),
                    "bootstrap_ci_95_high": float(high),
                    "bootstrap_replicates": 10_000,
                    "resampling": "stratified by actual class; seed 42",
                }
            )
    pd.DataFrame(bootstrap_rows).to_csv(
        OUTPUT / "bootstrap_ci_available_methods.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, method in zip(axes, ("Manual", "Random best")):
        subset = confusion[confusion["method"] == method].set_index("actual_class")
        matrix = subset[
            ["predicted_negative", "predicted_positive", "invalid"]
        ].to_numpy()
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Negative", "Positive", "Invalid"],
            yticklabels=["Actual negative", "Actual positive"],
        )
        ax.set_title(f"{method} (invalid is a separate column)")
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
    fig.suptitle("Available validation confusion counts (GA class counts not saved)")
    fig.savefig(OUTPUT / "available_confusion_matrices.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(labels, metrics["optimization_text_evaluations"], color=colors)
    axes[0].set_title("Optimization text evaluations")
    axes[0].set_ylabel("Classifications")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(labels, metrics["reported_runtime_seconds"], color=colors)
    axes[1].set_title("Reported runtime (phase scopes differ)")
    axes[1].set_ylabel("Seconds")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "computational_cost.png", dpi=180)
    plt.close(fig)

    tests = pd.DataFrame(load_json(comparison_dir / "mcnemar_tests.json"))
    tests["bonferroni_adjusted_p"] = (
        tests["exact_p_value"] * len(tests)
    ).clip(upper=1.0)
    tests["net_discordances_favor_method_2"] = (
        tests["c_m1_wrong_m2_right"] - tests["b_m1_right_m2_wrong"]
    )
    tests["significant_after_bonferroni"] = (
        tests["bonferroni_adjusted_p"] < tests["alpha"]
    )
    tests.to_csv(OUTPUT / "mcnemar_bonferroni.csv", index=False)

    print(f"Analysis outputs written to {OUTPUT}")
    print(f"Performance rows: {len(metrics)}; GA generation rows: {len(generation_metrics)}")
    print(
        "Per-example GA validation predictions, bootstrap inputs, token counts, "
        "and final held-out test results are not present in the source artifacts."
    )


if __name__ == "__main__":
    main()
