
"""
SST-2 data preparation for the prompt-optimization project.

Protocol
--------
1. opt: approximately 80% of the labelled GLUE train split.
   Used by the Genetic Algorithm to calculate prompt fitness.

2. val: approximately 20% of the labelled GLUE train split.
   Used to compare promising prompts and select the final prompt.

3. final test: the official GLUE validation split.
   Reserved for final evaluation after prompt selection.

Preprocessing
-------------
- Remove every row belonging to a text with conflicting labels.
- Keep ordinary duplicates (same normalized text and same label).
- Keep identical normalized texts together in the same split.

Labels: 0 = negative, 1 = positive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


# ============================================================
# 1. CONSTANTS
# ============================================================

LABEL_NAMES = {
    0: "negative",
    1: "positive",
}

DEFAULT_SEED = 42
DEFAULT_OPT_FRACTION = 0.8
STRATIFICATION_TOLERANCE = 0.01


# ============================================================
# 2. DATASET LOADING
# ============================================================

def _load_hf_split(split: str) -> pd.DataFrame:
    """Download one SST-2 split from Hugging Face."""

    from datasets import load_dataset as hf_load_dataset

    try:
        dataset = hf_load_dataset("nyu-mll/glue", "sst2", split=split)
        source = "nyu-mll/glue:sst2"

    except Exception as exc:
        print(
            f"nyu-mll/glue failed ({type(exc).__name__}); "
            "trying stanfordnlp/sst2"
        )

        dataset = hf_load_dataset("stanfordnlp/sst2", split=split)
        source = "stanfordnlp/sst2"

    df = dataset.to_pandas()[["sentence", "label"]].rename(
        columns={"sentence": "text"}
    )

    df["text"] = df["text"].str.strip()
    df.attrs["source"] = source

    return df


def load_train() -> pd.DataFrame:
    """Load the complete official labelled SST-2 training split."""
    return _load_hf_split("train")


def load_final_test(confirm: bool = False) -> pd.DataFrame:
    """
    Load the official GLUE validation split for final evaluation.

    Only call after selecting the final prompt using val.
    """

    if confirm is not True:
        raise PermissionError(
            "The final test set is reserved for final evaluation. "
            "Call load_final_test(confirm=True) only after prompt selection."
        )

    print(
        "WARNING: Loading the FINAL TEST set. "
        "Do not use it to modify prompts afterwards."
    )

    df = _load_hf_split("validation")
    _validate(df)

    return df


# ============================================================
# 3. DATA VALIDATION AND TEXT NORMALIZATION
# ============================================================

def _validate(df: pd.DataFrame) -> None:
    """Check that the DataFrame contains valid SST-2 examples."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    if not {"text", "label"} <= set(df.columns):
        raise ValueError(
            "DataFrame must contain 'text' and 'label' columns."
        )

    if df.empty:
        raise ValueError("DataFrame is empty.")

    if df["text"].isna().any() or df["label"].isna().any():
        raise ValueError("DataFrame contains missing values.")

    if not df["text"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("All texts must be strings.")

    if not df["label"].isin([0, 1]).all():
        raise ValueError("Labels must be 0 (negative) or 1 (positive).")


def _normalize(texts: pd.Series) -> pd.Series:
    """
    Normalize text only for comparison.

    Example:
        '  VERY good movie '
        'very   good movie'

    Both become:
        'very good movie'

    Original text is not modified.
    """

    return texts.str.lower().str.split().str.join(" ")


# ============================================================
# 4. DATA CLEANING
# ============================================================

def clean_train_data(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Remove all rows belonging to texts with conflicting labels.

    Ordinary duplicates with the same label are kept.
    """

    _validate(train_df)

    df = train_df[["text", "label"]].copy().reset_index(drop=True)
    normalized_texts = _normalize(df["text"])

    label_counts = (
        df.assign(_normalized=normalized_texts)
        .groupby("_normalized")["label"]
        .nunique()
    )

    conflicting_texts = set(
        label_counts[label_counts > 1].index
    )

    conflict_mask = normalized_texts.isin(conflicting_texts)

    cleaned_df = df.loc[~conflict_mask].copy().reset_index(drop=True)

    cleaned_normalized = _normalize(cleaned_df["text"])

    duplicate_rows_kept = int(
        cleaned_normalized.duplicated(keep="first").sum()
    )

    cleaning_info = {
        "n_original": int(len(df)),
        "n_removed_conflicting_rows": int(conflict_mask.sum()),
        "n_conflicting_unique_texts": int(len(conflicting_texts)),
        "n_after_cleaning": int(len(cleaned_df)),
        "n_ordinary_duplicate_rows_kept": duplicate_rows_kept,
    }

    return cleaned_df, cleaning_info


# ============================================================
# 5. DATASET STATISTICS
# ============================================================

def _split_stats(df: pd.DataFrame) -> dict:
    """Calculate the number and proportion of each sentiment label."""

    n_positive = int((df["label"] == 1).sum())
    n_negative = int((df["label"] == 0).sum())

    return {
        "n": int(len(df)),
        "n_negative": n_negative,
        "n_positive": n_positive,
        "positive_ratio": round(n_positive / len(df), 6),
    }


# ============================================================
# 6. OPTIMIZATION / VALIDATION SPLITTING
# ============================================================

@dataclass
class Splits:
    """Container for the optimization split, validation split and metadata."""

    opt: pd.DataFrame
    val: pd.DataFrame
    info: dict


def make_splits(
    train_df: pd.DataFrame,
    opt_fraction: float = DEFAULT_OPT_FRACTION,
    seed: int = DEFAULT_SEED,
    source: str = "unspecified",
) -> Splits:
    """
    Clean and split the labelled training data.

    StratifiedGroupKFold keeps identical normalized texts together.
    Several deterministic candidates are evaluated, and the candidate
    with the best class balance and split size is selected.
    """

    if not 0 < opt_fraction < 1:
        raise ValueError("opt_fraction must be strictly between 0 and 1.")

    df, cleaning_info = clean_train_data(train_df)

    if len(df) < 2:
        raise ValueError("Not enough examples after cleaning.")

    # Preserve row positions for verification.
    df["train_idx"] = np.arange(len(df))

    # Identical normalized texts belong to the same group.
    groups = _normalize(df["text"])

    n_splits = max(2, round(1 / (1 - opt_fraction)))

    if n_splits > len(df):
        raise ValueError("Not enough examples for the requested split.")

    overall_ratio = float(df["label"].mean())
    target_val_size = len(df) * (1 - opt_fraction)

    best_score = float("inf")
    best_indices = None
    best_details = None

    # Try 10 deterministic shuffles, each producing n_splits candidates.
    for seed_offset in range(10):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed + seed_offset,
        )

        for opt_idx, val_idx in splitter.split(
            X=df["text"],
            y=df["label"],
            groups=groups,
        ):
            opt_candidate = df.iloc[opt_idx]
            val_candidate = df.iloc[val_idx]

            # Both subsets should resemble the overall class distribution.
            opt_diff = abs(
                float(opt_candidate["label"].mean()) - overall_ratio
            )
            val_diff = abs(
                float(val_candidate["label"].mean()) - overall_ratio
            )

            # Group sizes may prevent an exact 80/20 split.
            size_diff = (
                abs(len(val_candidate) - target_val_size) / len(df)
            )

            # Class balance has priority over exact split size.
            score = max(opt_diff, val_diff) + 0.25 * size_diff

            if score < best_score:
                best_score = score
                best_indices = (opt_idx.copy(), val_idx.copy())
                best_details = {
                    "score": float(score),
                    "opt_class_ratio_difference": float(opt_diff),
                    "val_class_ratio_difference": float(val_diff),
                    "validation_size_difference": float(size_diff),
                    "seed_used": int(seed + seed_offset),
                }

    if best_indices is None:
        raise RuntimeError("Could not generate a valid stratified split.")

    opt_indices, val_indices = best_indices

    opt = df.iloc[opt_indices].copy()
    val = df.iloc[val_indices].copy()

    opt = opt.sort_values("train_idx").reset_index(drop=True)
    val = val.sort_values("train_idx").reset_index(drop=True)

    opt_texts = set(_normalize(opt["text"]))
    val_texts = set(_normalize(val["text"]))

    text_overlap = opt_texts.intersection(val_texts)

    info = {
        "source": source,
        "seed": seed,
        "selected_candidate_seed": best_details["seed_used"],
        "split_selection_score": best_details["score"],
        "opt_class_ratio_difference": best_details[
            "opt_class_ratio_difference"
        ],
        "val_class_ratio_difference": best_details[
            "val_class_ratio_difference"
        ],
        "requested_opt_fraction": opt_fraction,
        "actual_opt_fraction": round(len(opt) / len(df), 6),
        "actual_val_fraction": round(len(val) / len(df), 6),
        "label_encoding": {
            "0": "negative",
            "1": "positive",
        },
        "cleaning": cleaning_info,
        "n_total": int(len(df)),
        "n_total_after_cleaning": int(len(df)),
        "opt": _split_stats(opt),
        "val": _split_stats(val),
        "n_normalized_texts_shared_between_opt_and_val": len(text_overlap),
    }

    return Splits(opt=opt, val=val, info=info)


# ============================================================
# 7. SPLIT VERIFICATION
# ============================================================

def check_splits(
    splits: Splits,
    train_df: pd.DataFrame | None = None,
    raise_on_fail: bool = True,
) -> dict:
    """
    Verify:
    - No row appears in both splits.
    - No normalized text appears in both splits.
    - Labels are valid.
    - Class proportions are reasonably balanced.
    - All cleaned training rows are covered.
    - Splitting is reproducible with the same seed.
    """

    opt = splits.opt
    val = splits.val
    info = splits.info

    opt_texts = set(_normalize(opt["text"]))
    val_texts = set(_normalize(val["text"]))

    combined_size = len(opt) + len(val)

    if combined_size == 0:
        raise ValueError("Both splits are empty.")

    overall_ratio = (
        opt["label"].sum() + val["label"].sum()
    ) / combined_size

    checks = {
        "no_row_overlap_between_opt_and_val": not (
            set(opt["train_idx"]) & set(val["train_idx"])
        ),
        "no_text_overlap_between_opt_and_val": (
            len(opt_texts.intersection(val_texts)) == 0
        ),
        "labels_are_0_or_1": bool(
            opt["label"].isin([0, 1]).all()
            and val["label"].isin([0, 1]).all()
        ),
    }

    for name, part in (("opt", opt), ("val", val)):
        checks[f"{name}_is_reasonably_stratified"] = (
            abs(float(part["label"].mean()) - overall_ratio)
            <= STRATIFICATION_TOLERANCE
        )

    if train_df is not None:
        cleaned_train, _ = clean_train_data(train_df)

        checks["opt_plus_val_cover_every_cleaned_train_row"] = (
            len(opt) + len(val) == len(cleaned_train)
            and (
                set(opt["train_idx"]) | set(val["train_idx"])
            ) == set(range(len(cleaned_train)))
        )

        again = make_splits(
            train_df=train_df,
            opt_fraction=info["requested_opt_fraction"],
            seed=info["seed"],
            source=info["source"],
        )

        checks["reproducible_with_same_seed"] = (
            again.opt["train_idx"].equals(opt["train_idx"])
            and again.val["train_idx"].equals(val["train_idx"])
        )

    checks = {
        name: bool(result)
        for name, result in checks.items()
    }

    if raise_on_fail and not all(checks.values()):
        failed = [
            name for name, result in checks.items() if not result
        ]
        raise AssertionError(f"Split checks failed: {failed}")

    return checks


# ============================================================
# 8. PREPARING AND SAVING THE SPLITS
# ============================================================

def prepare_splits(
    opt_fraction: float = DEFAULT_OPT_FRACTION,
    seed: int = DEFAULT_SEED,
    save_dir: str | Path | None = None,
    check_test_leakage: bool = False,
) -> Splits:
    """
    Load SST-2 train, clean it, create opt/val, verify and optionally save.

    Saved files:
        opt.csv
        val.csv
        split_info.json

    By default, the final test set is NOT loaded or inspected.
    """

    train = load_train()

    splits = make_splits(
        train_df=train,
        opt_fraction=opt_fraction,
        seed=seed,
        source=train.attrs.get("source", "unknown"),
    )

    splits.info["checks"] = check_splits(
        splits,
        train_df=train,
    )

    # Optional diagnostic. This loads the official validation split.
    if check_test_leakage:
        test = _load_hf_split("validation")
        _validate(test)

        test_texts = set(_normalize(test["text"]))

        splits.info["test_leakage_check"] = {
            "n_test": int(len(test)),
            "test_texts_also_in_opt": int(
                _normalize(splits.opt["text"]).isin(test_texts).sum()
            ),
            "test_texts_also_in_val": int(
                _normalize(splits.val["text"]).isin(test_texts).sum()
            ),
            "warning": (
                "The final test texts were loaded for this diagnostic. "
                "Do not use this diagnostic during normal preparation."
            ),
        }

    if save_dir is not None:
        directory = Path(save_dir)
        directory.mkdir(parents=True, exist_ok=True)

        splits.opt.to_csv(directory / "opt.csv", index=False)
        splits.val.to_csv(directory / "val.csv", index=False)

        (directory / "split_info.json").write_text(
            json.dumps(splits.info, indent=2),
            encoding="utf-8",
        )

    return splits


# ============================================================
# 9. LOADING PREVIOUSLY SAVED SPLITS
# ============================================================

def load_saved_splits(
    data_dir: str | Path = "data",
) -> Splits:
    """Load opt.csv, val.csv and split_info.json."""

    directory = Path(data_dir)
    info_path = directory / "split_info.json"

    info = (
        json.loads(info_path.read_text(encoding="utf-8"))
        if info_path.exists()
        else {}
    )

    return Splits(
        opt=pd.read_csv(directory / "opt.csv"),
        val=pd.read_csv(directory / "val.csv"),
        info=info,
    )


# ============================================================
# 10. FIXED SAMPLE FOR GA FITNESS EVALUATION
# ============================================================

def stratified_sample(
    df: pd.DataFrame,
    n: int,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """
    Select a fixed, label-stratified sample from opt.

    Use the SAME sample for every candidate prompt so fitness scores
    can be compared fairly and LLM API calls can be reduced.
    """

    _validate(df)

    if not isinstance(n, (int, np.integer)):
        raise TypeError("n must be an integer.")

    if not 2 <= n <= len(df):
        raise ValueError(f"n must be between 2 and {len(df)}.")

    if n == len(df):
        return df.reset_index(drop=True).copy()

    sample, _ = train_test_split(
        df,
        train_size=n,
        stratify=df["label"],
        random_state=seed,
    )

    return sample.sort_index().reset_index(drop=True)


# ============================================================
# 11. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys

    output_directory = (
        sys.argv[1] if len(sys.argv) > 1 else "data"
    )

    result = prepare_splits(save_dir=output_directory)

    print(json.dumps(result.info, indent=2))
