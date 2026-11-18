"""Provider-independent prompt evaluation for SST-2 sentiment classification.

The GA developer only needs ``evaluate_prompt``; the LLM developer only needs to supply a
``predict_fn(prompt, texts) -> list[str]``. Nothing in this file knows about any LLM provider.

Labels: 0 = negative, 1 = positive, INVALID (-1) = the response could not be parsed.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

INVALID = -1
LABEL_NAMES = {0: "negative", 1: "positive"}
PredictFn = Callable[[str, list], list]

# Characters that may surround a one-word answer: "Positive.", '"negative"', "**positive**" ...
_STRIP_CHARS = " \t\r\n\"'`*_.!,;:()[]{}\u201c\u201d\u2018\u2019"


# --------------------------------------------------------------------------- parsing
def parse_response(raw) -> int:
    """Strictly map a raw model response to 1 (positive), 0 (negative) or INVALID (-1).

    Policy: after trimming whitespace, quotes, markdown stars and end punctuation, and
    ignoring case, the response must be EXACTLY "positive" or "negative". Everything else is
    INVALID: None / non-strings, empty text, both words, negation ("not positive"),
    hedging ("uncertain"), explanations or prefixes ("Sentiment: positive").
    """
    if not isinstance(raw, str):
        return INVALID
    cleaned = raw.strip(_STRIP_CHARS).lower()
    if cleaned == "positive":
        return 1
    if cleaned == "negative":
        return 0
    return INVALID


# --------------------------------------------------------------------------- metrics
def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    """Metrics for predictions in {0, 1, INVALID}.

    How invalid predictions are treated (so every number is reproducible by hand):
      * accuracy  = correct / ALL examples          -> an invalid answer is a wrong answer
      * precision = per class, over answers that were actually assigned to that class;
                    an invalid answer is assigned to no class, so it lowers no precision
      * recall    = per class, over ALL true examples of that class -> an invalid answer is a miss
      * macro P/R/F1 = unweighted mean of the two classes (negative, positive)
      * confusion_matrix: rows = true [negative, positive];
                          columns = predicted [negative, positive, invalid]; every example is counted
    """
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be non-empty and have the same length")
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, INVALID])[:2]
    n_invalid = int((y_pred == INVALID).sum())
    return {
        "n_examples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p),
        "recall_macro": float(r),
        "f1_macro": float(f1),  # recommended GA fitness
        "n_invalid": n_invalid,
        "invalid_rate": n_invalid / len(y_true),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": {"rows": ["true_negative", "true_positive"],
                                    "columns": ["pred_negative", "pred_positive", "invalid"]},
    }


# --------------------------------------------------------------------------- cache
class PredictionCache:
    """Cache of RAW model responses, one entry per (model, prompt, text, inference config).

    With ``path`` the cache is persisted as JSONL (one JSON object per line, append-only), so
    it survives restarts. Only string responses are cached: a ``None`` (failed call) is never
    stored, so it is retried next time. Unreadable lines in an existing file are skipped.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._data: dict[str, str] = {}
        self.n_skipped_lines = 0
        if self.path is not None and self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        key, raw = rec["key"], rec["raw"]
                        if not isinstance(key, str) or not isinstance(raw, str):
                            raise ValueError
                    except (ValueError, KeyError, TypeError):
                        self.n_skipped_lines += 1
                        continue
                    self._data[key] = raw

    @staticmethod
    def make_key(model_id: str, prompt: str, text: str, inference_config: dict | None = None) -> str:
        payload = json.dumps([model_id, prompt, text, inference_config or {}],
                             sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> str:
        return self._data[key]

    def put_many(self, items: dict, model_id: str | None = None) -> int:
        """Store {key: raw_response}; non-string responses are ignored. Returns how many were stored."""
        good = {k: v for k, v in items.items() if isinstance(v, str)}
        self._data.update(good)
        if self.path is not None and good:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for k, v in good.items():
                    f.write(json.dumps({"key": k, "model_id": model_id, "raw": v}, ensure_ascii=False) + "\n")
        return len(good)


# --------------------------------------------------------------------------- evaluation
def _validate_inputs(prompt, examples, labels, predict_fn, model_id, inference_config, cache, batch_size):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string (it is part of the cache key)")
    if inference_config is not None and not isinstance(inference_config, dict):
        raise TypeError("inference_config must be a dict or None")
    if not callable(predict_fn):
        raise TypeError("predict_fn must be callable: predict_fn(prompt, texts) -> list[str]")
    if cache is not None and not isinstance(cache, PredictionCache):
        raise TypeError("cache must be a PredictionCache or None")
    if batch_size is not None and (not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1):
        raise ValueError("batch_size must be a positive integer or None")
    texts, y_true = list(examples), list(labels)
    if len(texts) == 0:
        raise ValueError("examples is empty")
    if len(texts) != len(y_true):
        raise ValueError(f"examples ({len(texts)}) and labels ({len(y_true)}) must have the same length")
    if not all(isinstance(t, str) for t in texts):
        raise TypeError("every example must be a string")
    if not all(y in (0, 1) and not isinstance(y, str) for y in y_true):
        raise ValueError("labels must be 0 (negative) or 1 (positive)")
    return texts, np.asarray(y_true, dtype=int)


def evaluate_prompt(prompt: str, examples: Sequence[str], labels: Sequence[int], predict_fn: PredictFn,
                    *, model_id: str, inference_config: dict | None = None,
                    cache: PredictionCache | None = None, batch_size: int | None = None,
                    split_name: str | None = None) -> dict:
    """Evaluate ONE candidate prompt. This is the function the GA calls.

    ``predict_fn(prompt, texts)`` must return a list of raw responses, one per text, in order.
    Only texts that are not already cached are sent to it, each distinct text once.
    ``model_id`` (include the model VERSION) and ``inference_config`` (temperature, max tokens,
    ...) are part of the cache key: change either and cached answers are not reused.

    Returns a dict; the GA fitness is ``result["metrics"]["f1_macro"]``.
    """
    texts, y_true = _validate_inputs(prompt, examples, labels, predict_fn, model_id,
                                     inference_config, cache, batch_size)
    keys = [PredictionCache.make_key(model_id, prompt, t, inference_config) for t in texts]
    raw: list = [None] * len(texts)
    pending: dict[str, list[int]] = {}  # cache key -> positions that still need an answer
    n_cached = 0
    for i, k in enumerate(keys):
        if cache is not None and k in cache:
            raw[i] = cache.get(k)
            n_cached += 1
        else:
            pending.setdefault(k, []).append(i)

    unique_keys = list(pending)
    step = batch_size or max(len(unique_keys), 1)
    for s in range(0, len(unique_keys), step):
        batch_keys = unique_keys[s:s + step]
        batch_texts = [texts[pending[k][0]] for k in batch_keys]
        outs = predict_fn(prompt, batch_texts)
        if not isinstance(outs, (list, tuple)):
            raise TypeError(f"predict_fn must return a list of responses, got {type(outs).__name__}")
        if len(outs) != len(batch_texts):
            raise ValueError(f"predict_fn returned {len(outs)} responses for {len(batch_texts)} texts")
        for k, o in zip(batch_keys, outs):
            for i in pending[k]:
                raw[i] = o
        if cache is not None:
            cache.put_many(dict(zip(batch_keys, outs)), model_id=model_id)

    y_pred = np.array([parse_response(r) for r in raw], dtype=int)
    return {
        "prompt": prompt, "split": split_name, "model_id": model_id,
        "inference_config": inference_config or {},
        "n_examples": len(texts), "n_cached": n_cached, "n_model_texts": len(unique_keys),
        "metrics": compute_metrics(y_true, y_pred),
        "y_pred": y_pred.tolist(), "raw_responses": raw,
    }


# --------------------------------------------------------------------------- saving
def results_to_dataframe(results: dict | list[dict]) -> pd.DataFrame:
    """One summary row per evaluation result (the same rows save_results writes to CSV)."""
    results = [results] if isinstance(results, dict) else results
    rows = []
    for r in results:
        m, cm = r["metrics"], r["metrics"]["confusion_matrix"]
        rows.append({
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "split": r["split"], "model_id": r["model_id"],
            "inference_config": json.dumps(r["inference_config"], sort_keys=True, default=str),
            "prompt": r["prompt"], "n_examples": r["n_examples"], "n_cached": r["n_cached"],
            "accuracy": m["accuracy"], "precision_macro": m["precision_macro"],
            "recall_macro": m["recall_macro"], "f1_macro": m["f1_macro"],
            "n_invalid": m["n_invalid"], "invalid_rate": m["invalid_rate"],
            # positive class = 1: tn = true neg predicted neg, fp = true neg predicted pos, ...
            "tn": cm[0][0], "fp": cm[0][1], "neg_invalid": cm[0][2],
            "fn": cm[1][0], "tp": cm[1][1], "pos_invalid": cm[1][2],
        })
    return pd.DataFrame(rows)


def save_results(results: dict | list[dict], path: str | Path, *, append: bool = True,
                 include_predictions: bool = True) -> None:
    """Save evaluation results.

    ``.json``: full information (always overwritten); ``include_predictions=False`` drops the
               per-example ``y_pred`` / ``raw_responses`` lists to keep the file small.
    ``.csv`` : one summary row per result, appended to an existing file unless ``append=False``
               (handy for logging every candidate of every GA generation).
    """
    results = [results] if isinstance(results, dict) else list(results)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".json":
        if not include_predictions:
            results = [{k: v for k, v in r.items() if k not in ("y_pred", "raw_responses")} for r in results]
        p.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    elif p.suffix.lower() == ".csv":
        df = results_to_dataframe(results)
        write_header = not (append and p.exists() and p.stat().st_size > 0)
        df.to_csv(p, mode="a" if append else "w", header=write_header, index=False)
    else:
        raise ValueError("path must end with .json or .csv")
