import numpy as np
import pandas as pd
import pytest

from src.cv import cross_validate, run_grid, stratified_kfold_indices


def toy_corpus(n=90, seed=0):
    """Synthetic 'spam' vs 'ham' messages: every third message is spam."""
    rng = np.random.default_rng(seed)
    spam_words = ["free", "win", "prize", "claim", "urgent", "cash", "txt", "call"]
    ham_words = ["home", "dinner", "later", "meeting", "lol", "tomorrow", "sorry", "ok"]
    texts, y = [], []
    for i in range(n):
        is_spam = i % 3 == 0
        pool = spam_words if is_spam else ham_words
        words = list(rng.choice(pool, size=8)) + list(rng.choice(spam_words + ham_words, size=2))
        texts.append(" ".join(words))
        y.append(int(is_spam))
    return texts, np.array(y)


class TestStratifiedKFold:
    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_folds_are_disjoint_cover_everything_and_keep_class_ratios(self, k):
        y = np.array([1] * 20 + [0] * 70)
        folds = stratified_kfold_indices(y, k=k, seed=0)
        assert len(folds) == k
        all_val = np.concatenate([val for _, val in folds])
        assert sorted(all_val.tolist()) == list(range(len(y)))          # every row validated exactly once
        for train_idx, val_idx in folds:
            assert set(train_idx).isdisjoint(val_idx)
            assert len(train_idx) + len(val_idx) == len(y)
            assert abs(y[val_idx].sum() - 20 / k) <= 1                  # positives spread evenly
            assert abs(len(val_idx) - len(y) / k) <= 1                  # folds of (almost) equal size

    def test_is_deterministic_and_seed_dependent(self):
        y = np.array([1] * 10 + [0] * 30)
        a = stratified_kfold_indices(y, k=4, seed=0)
        b = stratified_kfold_indices(y, k=4, seed=0)
        c = stratified_kfold_indices(y, k=4, seed=1)
        assert all(np.array_equal(x[1], z[1]) for x, z in zip(a, b))
        assert any(not np.array_equal(x[1], z[1]) for x, z in zip(a, c))


class TestCrossValidate:
    def test_returns_scores_and_learns_the_toy_problem(self):
        texts, y = toy_corpus()
        res = cross_validate(texts, y, {"max_features": None, "min_df": 1, "use_extra": False},
                             {"hidden_layers": (), "learning_rate": 1.0, "epochs": 30, "seed": 0}, k=3, seed=0)
        assert set(res) == {"f1_mean", "f1_std", "f1_folds", "precision_mean", "recall_mean", "seconds",
                            "diverged_folds"}
        assert len(res["f1_folds"]) == 3
        assert res["f1_mean"] == pytest.approx(np.mean(res["f1_folds"]))
        assert res["f1_mean"] > 0.9
        assert res["seconds"] >= 0
        assert res["diverged_folds"] == 0

    def test_counts_folds_where_training_diverged(self):
        texts, y = toy_corpus()
        res = cross_validate(texts, y, {"max_features": None, "min_df": 1, "use_extra": True},
                             {"hidden_layers": (32, 32), "learning_rate": 1e6, "epochs": 20, "seed": 0}, k=3, seed=0)
        assert res["diverged_folds"] == 3

    def test_is_deterministic(self):
        texts, y = toy_corpus()
        args = (texts, y, {"max_features": None, "min_df": 1, "use_extra": True},
                {"hidden_layers": (4,), "learning_rate": 1.0, "epochs": 5, "seed": 0})
        assert cross_validate(*args, k=3, seed=0)["f1_folds"] == cross_validate(*args, k=3, seed=0)["f1_folds"]


class TestRunGrid:
    def test_frame_has_one_row_per_config_sorted_by_f1(self):
        texts, y = toy_corpus()
        configs = [
            {"name": "lr", "features": {"max_features": None, "min_df": 1, "use_extra": False},
             "model": {"hidden_layers": (), "learning_rate": 1.0, "epochs": 20, "seed": 0}},
            {"name": "mlp", "features": {"max_features": None, "min_df": 1, "use_extra": True},
             "model": {"hidden_layers": (8,), "learning_rate": 1.0, "epochs": 20, "seed": 0}},
        ]
        df = run_grid(texts, y, configs, k=3, seed=0, verbose=False)
        assert isinstance(df, pd.DataFrame) and len(df) == 2
        assert set(df["name"]) == {"lr", "mlp"}
        assert df["f1_mean"].is_monotonic_decreasing
        for col in ["hidden_layers", "activation", "learning_rate", "batch_size", "l2", "class_weight",
                    "epochs", "use_extra", "max_features", "min_df", "f1_mean", "f1_std",
                    "precision_mean", "recall_mean", "seconds", "diverged_folds"]:
            assert col in df.columns
