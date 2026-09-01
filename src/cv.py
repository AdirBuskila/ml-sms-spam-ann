"""Model selection inside the training set: stratified k-fold CV and a small grid runner.

The feature pipeline is fitted inside every fold on that fold's training part only, so the
validation score never sees vocabulary, idf or scaling statistics computed from validation rows.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.ann import NeuralNetwork
from src.features import FeaturePipeline
from src.metrics import f1_score, precision, recall


def stratified_kfold_indices(y, k: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """k (train_idx, val_idx) pairs; each class is shuffled and dealt round-robin over the folds."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    fold_of_row = np.empty(len(y), dtype=int)
    offset = 0
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        fold_of_row[idx] = (np.arange(len(idx)) + offset) % k
        offset += len(idx)                      # rotate so no fold is always the "big" one
    folds = []
    for f in range(k):
        val_idx = np.flatnonzero(fold_of_row == f)
        train_idx = np.flatnonzero(fold_of_row != f)
        folds.append((train_idx, val_idx))
    return folds


def cross_validate(texts, y, feature_params: dict, model_params: dict, k: int = 5, seed: int = 0) -> dict:
    """Run the full flow (fit features -> fit model -> score) in each fold; return mean/std F1 etc."""
    texts = list(texts)
    y = np.asarray(y)
    f1s, precs, recs = [], [], []
    diverged = 0
    t0 = time.perf_counter()
    for train_idx, val_idx in stratified_kfold_indices(y, k=k, seed=seed):
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        pipe = FeaturePipeline(**feature_params).fit(train_texts)
        model = NeuralNetwork(**model_params).fit(pipe.transform(train_texts), y[train_idx])
        pred = model.predict(pipe.transform(val_texts))
        f1s.append(f1_score(y[val_idx], pred))
        precs.append(precision(y[val_idx], pred))
        recs.append(recall(y[val_idx], pred))
        diverged += int(model.diverged_)
    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "f1_folds": [float(v) for v in f1s],
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recs)),
        "seconds": time.perf_counter() - t0,
        "diverged_folds": diverged,          # folds where SGD blew up (non-finite loss)
    }


def run_grid(texts, y, configs: list[dict], k: int = 5, seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Evaluate every config with cross_validate and return a table sorted by mean F1 (best first).

    Each config is {"name": str, "features": FeaturePipeline kwargs, "model": NeuralNetwork kwargs}.
    """
    rows = []
    for cfg in configs:
        pipe = FeaturePipeline(**cfg["features"])
        model = NeuralNetwork(**cfg["model"])
        res = cross_validate(texts, y, cfg["features"], cfg["model"], k=k, seed=seed)
        row = {
            "name": cfg["name"],
            "hidden_layers": model.hidden_layers,
            "activation": model.activation,
            "learning_rate": model.learning_rate,
            "batch_size": model.batch_size,
            "epochs": model.epochs,
            "l2": model.l2,
            "class_weight": model.class_weight,
            "use_extra": pipe.use_extra,
            "max_features": pipe.max_features,
            "min_df": pipe.min_df,
        }
        row.update(res)
        rows.append(row)
        if verbose:
            print(f"{cfg['name']:<34} F1 = {res['f1_mean']:.3f} +/- {res['f1_std']:.3f}   "
                  f"P = {res['precision_mean']:.3f}  R = {res['recall_mean']:.3f}   ({res['seconds']:.1f}s)")
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False, kind="stable").reset_index(drop=True)
