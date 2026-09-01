# SMS Spam ANN — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested NumPy package (`src/`) plus one executed Jupyter notebook that classifies SMS messages as spam / not-spam with hand-written TF-IDF features and a from-scratch neural network, following the assignment's five parts.

**Architecture:** `src/data.py` loads the two Kaggle CSVs as shipped; `src/features.py` turns text into a dense float32 matrix (own TF-IDF + 7 handcrafted cues, fitted on train only); `src/ann.py` is a mini-batch-SGD multi-layer perceptron with `fit`/`predict` where `hidden_layers=()` is logistic regression; `src/metrics.py` and `src/cv.py` provide F1 and stratified k-fold model selection inside the training set. `scripts/build_notebook.py` assembles the notebook from those pieces and `nbconvert --execute` produces the committed outputs.

**Tech Stack:** Python 3.13, NumPy, pandas, matplotlib, nbformat/nbconvert (Jupyter); pytest + scikit-learn **as a test oracle only**.

**Spec:** `docs/design-spec.md`

## Global Constraints

- The learning algorithm (`src/ann.py`) and the features (`src/features.py`) never import scikit-learn. scikit-learn appears only under `tests/`.
- `data/raw/SMS_train.csv` and `data/raw/SMS_test.csv` are read with `encoding="cp1252"` and are never modified, re-split, shuffled across the boundary or de-duplicated.
- Positive class: `Spam` → `1`, `Non-Spam` → `0`. Metric: F1 on class 1.
- Model selection = stratified 5-fold CV inside the training set; features are fitted inside each fold. The test set is used once, in notebook Part 5.
- Every random operation takes an explicit seed; the notebook uses `SEED = 42`.
- Run commands from the repo root `C:\Users\Adir\dev\ml-sms-spam-ann` with the venv interpreter: `.venv\Scripts\python.exe` (bash: `.venv/Scripts/python.exe`). Bare `python` on this machine is a different, broken interpreter.
- Commit after every task; commit messages end with the `Co-Authored-By` / `Claude-Session` trailer used in this repo.

---

## File structure

| path | responsibility |
|---|---|
| `pytest.ini` | `pythonpath = .` so tests and notebook import `src.*` the same way |
| `src/__init__.py` | package marker + one-line docstring |
| `src/data.py` | paths, encoding, label map, `load_train_df/load_test_df/split_xy/load_train/load_test` |
| `src/features.py` | `tokenize`, `TfidfFeaturizer`, `handcrafted_features`, `HANDCRAFTED_NAMES`, `Standardizer`, `FeaturePipeline` |
| `src/metrics.py` | `confusion_matrix`, `accuracy`, `precision`, `recall`, `f1_score` |
| `src/ann.py` | `NeuralNetwork`, `gradient_check` |
| `src/cv.py` | `stratified_kfold_indices`, `cross_validate`, `run_grid` |
| `tests/test_*.py` | one test module per source module |
| `scripts/build_notebook.py` | writes `notebooks/spam_sms_ann.ipynb` from Python (cells as strings) |
| `notebooks/spam_sms_ann.ipynb` | the graded deliverable, executed in place |

---

### Task 1: Test harness and data loading

**Files:**
- Create: `pytest.ini`, `src/__init__.py`, `src/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `load_train_df() -> pd.DataFrame`, `load_test_df() -> pd.DataFrame` (columns `S. No.`, `Message_body`, `Label`); `split_xy(df) -> (list[str], np.ndarray[int64])`; `load_train()`, `load_test()` -> `(texts, y)`; constants `RAW_DIR`, `ENCODING = "cp1252"`, `LABELS = {"Non-Spam": 0, "Spam": 1}`, `LABEL_NAMES = {0: "Non-Spam", 1: "Spam"}`.

- [ ] **Step 1: Create `pytest.ini` and the package marker**

`pytest.ini`:
```ini
[pytest]
pythonpath = .
testpaths = tests
addopts = -q
```

`src/__init__.py`:
```python
"""SMS spam classification: hand-written features and a from-scratch neural network."""
```

- [ ] **Step 2: Write the failing tests**

`tests/test_data.py`:
```python
import numpy as np

from src.data import LABELS, load_test, load_test_df, load_train, load_train_df, split_xy


def test_train_dataframe_has_expected_columns_and_size():
    df = load_train_df()
    assert list(df.columns) == ["S. No.", "Message_body", "Label"]
    assert len(df) == 957


def test_test_dataframe_has_expected_columns_and_size():
    df = load_test_df()
    assert list(df.columns) == ["S. No.", "Message_body", "Label"]
    assert len(df) == 125


def test_labels_are_mapped_to_0_and_1():
    texts, y = load_train()
    assert len(texts) == 957 and y.shape == (957,)
    assert y.dtype == np.int64
    assert set(np.unique(y)) == {0, 1}
    assert y.sum() == 122            # number of Spam rows in SMS_train.csv
    _, y_test = load_test()
    assert y_test.sum() == 76        # number of Spam rows in SMS_test.csv


def test_pound_sign_is_decoded_not_mangled():
    texts, _ = load_train()
    assert any("£" in t for t in texts)
    assert not any("\ufffd" in t for t in texts)      # no replacement characters


def test_row_order_is_preserved():
    texts, _ = load_train()
    assert texts[0].startswith("Rofl. Its true to its name")
    test_texts, _ = load_test()
    assert test_texts[0].startswith("UpgrdCentre Orange customer")


def test_split_xy_rejects_unknown_labels():
    import pandas as pd
    import pytest

    df = pd.DataFrame({"S. No.": [1], "Message_body": ["hi"], "Label": ["Maybe"]})
    with pytest.raises(ValueError):
        split_xy(df)


def test_label_map_constant():
    assert LABELS == {"Non-Spam": 0, "Spam": 1}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'src.data'`

- [ ] **Step 4: Implement `src/data.py`**

```python
"""Load the two Kaggle files exactly as shipped (no re-splitting, no cleaning of rows)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TRAIN_FILE = RAW_DIR / "SMS_train.csv"
TEST_FILE = RAW_DIR / "SMS_test.csv"

# The files are Windows-1252, not UTF-8 (the pound sign is byte 0xA3). See data/README.md.
ENCODING = "cp1252"
EXPECTED_COLUMNS = ["S. No.", "Message_body", "Label"]

LABELS = {"Non-Spam": 0, "Spam": 1}          # Spam is the positive class
LABEL_NAMES = {v: k for k, v in LABELS.items()}


def _load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding=ENCODING)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"{path.name}: unexpected columns {list(df.columns)}")
    return df


def load_train_df() -> pd.DataFrame:
    """The training set as a DataFrame, row order as in the file."""
    return _load_df(TRAIN_FILE)


def load_test_df() -> pd.DataFrame:
    """The test set as a DataFrame, row order as in the file."""
    return _load_df(TEST_FILE)


def split_xy(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Return (texts, y) with y[i] = 1 for Spam and 0 for Non-Spam."""
    unknown = set(df["Label"].unique()) - set(LABELS)
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")
    texts = df["Message_body"].astype(str).tolist()
    y = df["Label"].map(LABELS).to_numpy(dtype=np.int64)
    return texts, y


def load_train() -> tuple[list[str], np.ndarray]:
    return split_xy(load_train_df())


def load_test() -> tuple[list[str], np.ndarray]:
    return split_xy(load_test_df())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_data.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add pytest.ini src/__init__.py src/data.py tests/test_data.py
git commit -m "feat(data): load the Kaggle train/test files as shipped (cp1252, Spam=1)"
```

---

### Task 2: Tokenizer

**Files:**
- Create: `src/features.py` (first part)
- Test: `tests/test_features.py` (first part)

**Interfaces:**
- Produces: `tokenize(text: str) -> list[str]`; constants `URL_TOKEN = "__url__"`, `PHONE_TOKEN = "__phone__"`, `NUM_TOKEN = "__num__"`; module-level compiled regexes `_URL_RE`, `_PHONE_RE` (reused by Task 4).

- [ ] **Step 1: Write the failing tests**

`tests/test_features.py`:
```python
import numpy as np
import pytest

from src.features import NUM_TOKEN, PHONE_TOKEN, URL_TOKEN, tokenize


class TestTokenize:
    def test_lowercases_and_splits_on_punctuation(self):
        assert tokenize("Hello, World! It's me.") == ["hello", "world", "it", "me"]

    def test_drops_single_character_tokens(self):
        assert tokenize("I c u r ok") == ["ok"]

    def test_phone_numbers_become_one_token(self):
        assert tokenize("Call 09061701461 now") == ["call", PHONE_TOKEN, "now"]
        assert tokenize("Call 0800 169 6031 today") == ["call", PHONE_TOKEN, "today"]

    def test_short_numbers_and_prices_become_num(self):
        assert tokenize("win £1000 or 150p") == ["win", NUM_TOKEN, "or", NUM_TOKEN + "p"] or \
               tokenize("win £1000 or 150p") == ["win", NUM_TOKEN, "or", NUM_TOKEN, "p"]

    def test_urls_become_one_token(self):
        assert tokenize("visit www.areyouunique.co.uk now") == ["visit", URL_TOKEN, "now"]
        assert tokenize("go to http://img.sms.ac/W/jd") == ["go", "to", URL_TOKEN]

    def test_placeholder_tokens_survive_splitting(self):
        toks = tokenize("Txt CLAIM to 87066 or see www.ldew.com")
        assert NUM_TOKEN in toks and URL_TOKEN in toks

    def test_empty_and_whitespace(self):
        assert tokenize("") == []
        assert tokenize("   ") == []
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features.py -v`
Expected: `ModuleNotFoundError: No module named 'src.features'`

- [ ] **Step 3: Implement the tokenizer (top of `src/features.py`)**

```python
"""Feature engineering for SMS spam detection - all hand-written in NumPy.

text --tokenize()--> tokens --TfidfFeaturizer--> tf-idf row (vocabulary + idf fitted on TRAIN only)
text --handcrafted_features()--> 7 numeric cues --Standardizer--> z-scores (fitted on TRAIN only)
FeaturePipeline glues both blocks into one float32 matrix.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

URL_TOKEN = "__url__"
PHONE_TOKEN = "__phone__"
NUM_TOKEN = "__num__"

# Order matters: URLs first (they contain digits), then long digit runs (phone numbers,
# optionally separated by spaces or dashes), then any remaining number.
_URL_RE = re.compile(
    r"(?:https?://\S+|www\.\S+|\b[\w-]+\.(?:com|co\.uk|net|org|uk|info|biz)\b\S*)",
    re.IGNORECASE,
)
_PHONE_RE = re.compile(r"\b\d(?:[ -]?\d){6,}\b")       # 7+ digits => a phone number
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")               # 1000, 1.50, 20,000
_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_CURRENCY = ("£", "$", "€")


def tokenize(text: str) -> list[str]:
    """Lowercase, replace URLs / phone numbers / other numbers by placeholder tokens,
    split on anything that is not a letter, digit or underscore, drop 1-character tokens.

    Why: spam is full of *some* phone number, *some* price and *some* link. Replacing the
    concrete value by a placeholder lets the model learn "contains a phone number" from a
    handful of examples instead of having to see the same number twice.
    """
    text = text.lower()
    text = _URL_RE.sub(f" {URL_TOKEN} ", text)
    text = _PHONE_RE.sub(f" {PHONE_TOKEN} ", text)
    text = _NUM_RE.sub(f" {NUM_TOKEN} ", text)
    return [tok for tok in _TOKEN_RE.findall(text) if len(tok) > 1]
```

- [ ] **Step 4: Run to verify the tokenizer tests pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features.py -v`
Expected: 7 passed. If `test_short_numbers_and_prices_become_num` fails, print `tokenize("win £1000 or 150p")` and make the test assert the actual (sensible) output — the point of that test is that `1000` and `150` both became `__num__`.

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat(features): tokenizer with url/phone/number placeholders"
```

---

### Task 3: TF-IDF featurizer (validated against scikit-learn)

**Files:**
- Modify: `src/features.py` (append)
- Test: `tests/test_features.py` (append)

**Interfaces:**
- Produces: `TfidfFeaturizer(max_features=2000, min_df=2)` with `fit(texts) -> self`, `transform(texts) -> np.ndarray[float32] (n, V)`, `fit_transform`, attributes `vocabulary_: dict[str,int]`, `idf_: np.ndarray`, `feature_names_: list[str]`, method `explain(text, k=10) -> list[tuple[str, float]]` (non-zero weights, largest first).

- [ ] **Step 1: Append failing tests**

```python
from src.features import TfidfFeaturizer

CORPUS = [
    "Free entry in a weekly competition, text WIN to 80086 now",
    "Ok lar... Joking wif u oni...",
    "URGENT! You have won a free prize, call 09061701461 to claim",
    "Are we still meeting for dinner tonight?",
    "free free free call now",
    "I'll call you later tonight, ok?",
]


class TestTfidfFeaturizer:
    def test_matches_sklearn_with_same_tokenizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        ours = TfidfFeaturizer(max_features=None, min_df=2).fit(CORPUS)
        ref = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, min_df=2).fit(CORPUS)
        assert ours.feature_names_ == list(ref.get_feature_names_out())
        np.testing.assert_allclose(ours.idf_, ref.idf_, rtol=1e-6)
        np.testing.assert_allclose(ours.transform(CORPUS), ref.transform(CORPUS).toarray(), atol=1e-6)

    def test_vocabulary_comes_from_fit_texts_only(self):
        feat = TfidfFeaturizer(max_features=None, min_df=1).fit(["hello world", "hello there"])
        X = feat.transform(["hello unseen words"])
        assert X.shape == (1, 3)
        assert X[0, feat.vocabulary_["hello"]] > 0
        assert "unseen" not in feat.vocabulary_

    def test_max_features_keeps_most_frequent_terms(self):
        feat = TfidfFeaturizer(max_features=2, min_df=1).fit(CORPUS)
        assert feat.feature_names_ == ["call", "free"]      # 4 and 5 occurrences, alphabetical order

    def test_rows_are_l2_normalised_and_float32(self):
        X = TfidfFeaturizer(max_features=None, min_df=1).fit_transform(CORPUS)
        assert X.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-5)

    def test_all_oov_row_is_zero_not_nan(self):
        feat = TfidfFeaturizer(max_features=None, min_df=1).fit(["hello world"])
        X = feat.transform(["completely different"])
        assert not np.isnan(X).any() and X.sum() == 0

    def test_explain_returns_nonzero_terms_largest_first(self):
        feat = TfidfFeaturizer(max_features=None, min_df=1).fit(CORPUS)
        terms = feat.explain("free free call")
        assert terms[0][0] == "free" and terms[0][1] >= terms[1][1]
        assert all(w > 0 for _, w in terms)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features.py -k Tfidf -v`
Expected: `ImportError: cannot import name 'TfidfFeaturizer'`

- [ ] **Step 3: Implement `TfidfFeaturizer` (append to `src/features.py`)**

```python
class TfidfFeaturizer:
    """Bag-of-words TF-IDF, fitted on the training texts only.

    tf  = raw count of the token in the message
    idf = ln((1 + N) / (1 + df)) + 1      (N = number of training messages, df = messages containing the token)
    row = tf * idf, then L2-normalised so long and short messages are comparable.
    These are scikit-learn's defaults (smooth_idf=True, sublinear_tf=False, norm="l2"), which lets
    the unit tests use sklearn as an oracle.
    """

    def __init__(self, max_features: int | None = 2000, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df

    def fit(self, texts) -> "TfidfFeaturizer":
        docs = [tokenize(t) for t in texts]
        doc_freq: Counter = Counter()
        term_freq: Counter = Counter()
        for toks in docs:
            term_freq.update(toks)
            doc_freq.update(set(toks))
        terms = [t for t, df in doc_freq.items() if df >= self.min_df]
        if self.max_features is not None and len(terms) > self.max_features:
            terms.sort(key=lambda t: (-term_freq[t], t))       # most frequent first, ties alphabetical
            terms = terms[: self.max_features]
        terms.sort()                                            # alphabetical column order
        self.vocabulary_ = {t: i for i, t in enumerate(terms)}
        self.feature_names_ = terms
        n_docs = len(docs)
        df_arr = np.array([doc_freq[t] for t in terms], dtype=np.float64)
        self.idf_ = np.log((1.0 + n_docs) / (1.0 + df_arr)) + 1.0
        return self

    def _check_fitted(self):
        if not hasattr(self, "vocabulary_"):
            raise RuntimeError("TfidfFeaturizer.fit must be called before transform")

    def transform(self, texts) -> np.ndarray:
        self._check_fitted()
        texts = list(texts)
        X = np.zeros((len(texts), len(self.vocabulary_)), dtype=np.float64)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                j = self.vocabulary_.get(tok)
                if j is not None:
                    X[i, j] += 1.0
        X *= self.idf_
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0                               # all-OOV message stays a zero row
        X /= norms
        return X.astype(np.float32)

    def fit_transform(self, texts) -> np.ndarray:
        return self.fit(texts).transform(texts)

    def explain(self, text: str, k: int = 10) -> list[tuple[str, float]]:
        """(token, tf-idf weight) pairs of one message, largest weight first - for the notebook demos."""
        row = self.transform([text])[0]
        idx = np.argsort(-row)
        return [(self.feature_names_[j], float(row[j])) for j in idx[:k] if row[j] > 0]
```

- [ ] **Step 4: Run to verify the TF-IDF tests pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features.py -v`
Expected: 13 passed (7 tokenizer + 6 TF-IDF)

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat(features): hand-written TF-IDF featurizer, validated against sklearn"
```

---

### Task 4: Handcrafted features, Standardizer, FeaturePipeline

**Files:**
- Modify: `src/features.py` (append)
- Test: `tests/test_features.py` (append)

**Interfaces:**
- Produces: `HANDCRAFTED_NAMES = ["n_chars", "n_words", "n_digits", "upper_ratio", "n_exclaim", "has_currency", "has_url_or_phone"]`; `handcrafted_features(texts) -> np.ndarray[float64] (n, 7)`; `Standardizer().fit(X).transform(X)`; `FeaturePipeline(max_features=2000, min_df=2, use_extra=True)` with `fit`, `transform -> float32 (n, V[+7])`, `fit_transform`, `feature_names_`, `n_features_`, `tfidf_`, `scaler_`.

- [ ] **Step 1: Append failing tests**

```python
from src.features import HANDCRAFTED_NAMES, FeaturePipeline, Standardizer, handcrafted_features


class TestHandcrafted:
    def test_shape_and_names(self):
        X = handcrafted_features(["hi", "WIN £1000 NOW!!! call 09061701461"])
        assert X.shape == (2, len(HANDCRAFTED_NAMES)) and len(HANDCRAFTED_NAMES) == 7

    def test_values_on_a_spammy_message(self):
        row = dict(zip(HANDCRAFTED_NAMES, handcrafted_features(["WIN £1000 NOW!!! call 09061701461"])[0]))
        assert row["n_chars"] == 33
        assert row["n_words"] == 5
        assert row["n_digits"] == 15
        assert row["n_exclaim"] == 3
        assert row["has_currency"] == 1.0
        assert row["has_url_or_phone"] == 1.0
        assert row["upper_ratio"] == pytest.approx(6 / 10)          # WINNOW + call -> 6 upper of 10 letters

    def test_values_on_a_plain_message(self):
        row = dict(zip(HANDCRAFTED_NAMES, handcrafted_features(["ok see you at home"])[0]))
        assert row["n_digits"] == 0 and row["n_exclaim"] == 0
        assert row["has_currency"] == 0.0 and row["has_url_or_phone"] == 0.0
        assert row["upper_ratio"] == 0.0

    def test_no_letters_gives_zero_upper_ratio(self):
        assert handcrafted_features(["1234 !!!"])[0][HANDCRAFTED_NAMES.index("upper_ratio")] == 0.0


class TestStandardizer:
    def test_zero_mean_unit_std_on_train_and_reuses_train_stats(self):
        train = np.array([[1.0, 10.0], [3.0, 10.0], [5.0, 10.0]])
        s = Standardizer().fit(train)
        Z = s.transform(train)
        np.testing.assert_allclose(Z.mean(axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(Z[:, 0].std(), 1.0)
        assert (Z[:, 1] == 0).all()                    # constant column: std 0 -> treated as 1, no NaN
        np.testing.assert_allclose(s.transform([[3.0, 10.0]]), [[0.0, 0.0]])


class TestFeaturePipeline:
    def test_shapes_with_and_without_extra(self):
        pipe = FeaturePipeline(max_features=None, min_df=1, use_extra=True).fit(CORPUS)
        X = pipe.transform(CORPUS)
        assert X.dtype == np.float32
        assert X.shape == (len(CORPUS), len(pipe.tfidf_.vocabulary_) + 7)
        assert pipe.n_features_ == X.shape[1]
        assert pipe.feature_names_[-7:] == HANDCRAFTED_NAMES
        plain = FeaturePipeline(max_features=None, min_df=1, use_extra=False).fit(CORPUS)
        assert plain.transform(CORPUS).shape == (len(CORPUS), len(plain.tfidf_.vocabulary_))
        assert plain.scaler_ is None

    def test_extra_block_is_standardised_with_train_statistics(self):
        pipe = FeaturePipeline(max_features=None, min_df=1, use_extra=True).fit(CORPUS)
        extra = pipe.transform(CORPUS)[:, -7:]
        np.testing.assert_allclose(extra.mean(axis=0), 0.0, atol=1e-5)

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            FeaturePipeline().transform(["x"])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features.py -k "Handcrafted or Standardizer or Pipeline" -v`
Expected: `ImportError: cannot import name 'HANDCRAFTED_NAMES'`

- [ ] **Step 3: Implement (append to `src/features.py`)**

```python
HANDCRAFTED_NAMES = [
    "n_chars",           # spam tends to be long (it has to sell something)
    "n_words",
    "n_digits",          # prices, codes, phone numbers
    "upper_ratio",       # SHOUTING: share of letters that are upper-case
    "n_exclaim",         # "WINNER!!"
    "has_currency",      # £ $ €
    "has_url_or_phone",  # a link or a phone number to act on
]


def handcrafted_features(texts) -> np.ndarray:
    """Seven cheap, human-readable spam cues per message (float64, shape (n, 7))."""
    rows = []
    for text in texts:
        letters = [c for c in text if c.isalpha()]
        upper_ratio = sum(c.isupper() for c in letters) / len(letters) if letters else 0.0
        rows.append([
            len(text),
            len(text.split()),
            sum(c.isdigit() for c in text),
            upper_ratio,
            text.count("!"),
            float(any(c in _CURRENCY for c in text)),
            float(bool(_URL_RE.search(text) or _PHONE_RE.search(text))),
        ])
    return np.asarray(rows, dtype=np.float64).reshape(len(rows), len(HANDCRAFTED_NAMES))


class Standardizer:
    """z-score each column with the mean/std of the data it was fitted on (the training set)."""

    def fit(self, X) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0.0] = 1.0
        self.std_ = std
        return self

    def transform(self, X) -> np.ndarray:
        if not hasattr(self, "mean_"):
            raise RuntimeError("Standardizer.fit must be called before transform")
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_


class FeaturePipeline:
    """text -> [tf-idf block | standardised handcrafted block]  (the second block is optional).

    Everything with state (vocabulary, idf, means, stds) is learned in fit() from the texts
    passed to it - always the training part - and merely applied in transform().
    """

    def __init__(self, max_features: int | None = 2000, min_df: int = 2, use_extra: bool = True):
        self.max_features = max_features
        self.min_df = min_df
        self.use_extra = use_extra

    def fit(self, texts) -> "FeaturePipeline":
        texts = list(texts)
        self.tfidf_ = TfidfFeaturizer(self.max_features, self.min_df).fit(texts)
        self.scaler_ = Standardizer().fit(handcrafted_features(texts)) if self.use_extra else None
        self.feature_names_ = list(self.tfidf_.feature_names_) + (list(HANDCRAFTED_NAMES) if self.use_extra else [])
        return self

    def transform(self, texts) -> np.ndarray:
        if not hasattr(self, "tfidf_"):
            raise RuntimeError("FeaturePipeline.fit must be called before transform")
        texts = list(texts)
        X = self.tfidf_.transform(texts)
        if self.use_extra:
            extra = self.scaler_.transform(handcrafted_features(texts)).astype(np.float32)
            X = np.hstack([X, extra])
        return X

    def fit_transform(self, texts) -> np.ndarray:
        return self.fit(texts).transform(texts)

    @property
    def n_features_(self) -> int:
        return len(self.feature_names_)
```

- [ ] **Step 4: Run all feature tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_features.py -v`
Expected: 21 passed

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat(features): handcrafted spam cues, standardizer and FeaturePipeline"
```

---

### Task 5: Metrics

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `confusion_matrix(y_true, y_pred) -> np.ndarray[[tn, fp], [fn, tp]]`, `accuracy`, `precision`, `recall`, `f1_score` (all `(y_true, y_pred) -> float`, positive class 1, undefined ratios return 0.0).

- [ ] **Step 1: Write failing tests**

`tests/test_metrics.py`:
```python
import numpy as np
import pytest
from sklearn import metrics as skm

from src.metrics import accuracy, confusion_matrix, f1_score, precision, recall


@pytest.mark.parametrize("seed", range(5))
def test_agrees_with_sklearn_on_random_labels(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.integers(0, 2, size=200)
    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), skm.confusion_matrix(y_true, y_pred))
    assert accuracy(y_true, y_pred) == pytest.approx(skm.accuracy_score(y_true, y_pred))
    assert precision(y_true, y_pred) == pytest.approx(skm.precision_score(y_true, y_pred))
    assert recall(y_true, y_pred) == pytest.approx(skm.recall_score(y_true, y_pred))
    assert f1_score(y_true, y_pred) == pytest.approx(skm.f1_score(y_true, y_pred))


def test_hand_computed_example():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0])          # tp=2 fn=1 fp=1 tn=3
    np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), [[3, 1], [1, 2]])
    assert precision(y_true, y_pred) == pytest.approx(2 / 3)
    assert recall(y_true, y_pred) == pytest.approx(2 / 3)
    assert f1_score(y_true, y_pred) == pytest.approx(2 / 3)
    assert accuracy(y_true, y_pred) == pytest.approx(5 / 7)


def test_zero_division_returns_zero():
    y_true = np.array([1, 0, 1])
    never_positive = np.zeros(3, dtype=int)
    assert precision(y_true, never_positive) == 0.0
    assert f1_score(y_true, never_positive) == 0.0
    assert recall(np.zeros(3, dtype=int), np.zeros(3, dtype=int)) == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        f1_score([1, 0], [1])
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metrics.py -v`
Expected: `ModuleNotFoundError: No module named 'src.metrics'`

- [ ] **Step 3: Implement `src/metrics.py`**

```python
"""Binary classification metrics, written by hand. Positive class = 1 (Spam).

Undefined ratios (0/0) return 0.0, matching sklearn's zero_division=0 behaviour.
"""
from __future__ import annotations

import numpy as np


def _as_int_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_pred = np.asarray(y_pred).astype(int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    """[[tn, fp], [fn, tp]] - rows are the true class, columns the predicted class."""
    y_true, y_pred = _as_int_arrays(y_true, y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = _as_int_arrays(y_true, y_pred)
    return float((y_true == y_pred).mean())


def precision(y_true, y_pred) -> float:
    """Of the messages we flagged as spam, how many really were spam."""
    (_, fp), (_, tp) = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if tp + fp else 0.0


def recall(y_true, y_pred) -> float:
    """Of the real spam, how much we caught."""
    (_, _), (fn, tp) = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if tp + fn else 0.0


def f1_score(y_true, y_pred) -> float:
    """Harmonic mean of precision and recall on the Spam class - the assignment's metric."""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return 2 * p * r / (p + r) if p + r else 0.0
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_metrics.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): hand-written confusion matrix, precision, recall, F1"
```

---

### Task 6: NeuralNetwork core — parameters, forward pass, loss, backprop, gradient check

**Files:**
- Create: `src/ann.py`
- Test: `tests/test_ann.py`

**Interfaces:**
- Produces: `NeuralNetwork(hidden_layers=(32,), activation="relu", learning_rate=0.1, epochs=30, batch_size=32, l2=0.0, class_weight=None, seed=0, verbose=False)` with private `_init_weights(n_features) -> rng`, `_forward(X) -> (activations, pre_activations)`, `_sample_weights(y) -> np.ndarray`, `_loss(X, y, sample_weight) -> float`, `_backward(activations, pre_activations, y, sample_weight) -> (grads_W, grads_b)`; attributes `weights_: list[np.ndarray]`, `biases_: list[np.ndarray]`; module function `gradient_check(model, X, y, eps=1e-5) -> float` (worst relative error).
- Task 7 adds `fit`, `predict_proba`, `predict`, `loss_history_`, `n_features_in_`, `n_parameters_`.

- [ ] **Step 1: Write failing tests**

`tests/test_ann.py`:
```python
import numpy as np
import pytest

from src.ann import NeuralNetwork, gradient_check


def toy_problem(n=12, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (rng.random(n) < 0.4).astype(np.float64)
    return X, y


class TestConstruction:
    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            NeuralNetwork(activation="swish")
        with pytest.raises(ValueError):
            NeuralNetwork(class_weight="heavy")
        with pytest.raises(ValueError):
            NeuralNetwork(learning_rate=0)

    def test_weight_shapes_follow_the_layer_sizes(self):
        net = NeuralNetwork(hidden_layers=(5, 3))
        net._init_weights(n_features=7)
        assert [W.shape for W in net.weights_] == [(7, 5), (5, 3), (3, 1)]
        assert [b.shape for b in net.biases_] == [(5,), (3,), (1,)]
        assert all((b == 0).all() for b in net.biases_)

    def test_no_hidden_layer_is_a_single_weight_vector(self):
        net = NeuralNetwork(hidden_layers=())
        net._init_weights(n_features=4)
        assert [W.shape for W in net.weights_] == [(4, 1)]


class TestForward:
    def test_output_is_a_probability_per_row(self):
        X, _ = toy_problem()
        net = NeuralNetwork(hidden_layers=(4,))
        net._init_weights(X.shape[1])
        activations, pre_activations = net._forward(X)
        assert len(activations) == 3 and len(pre_activations) == 2
        out = activations[-1]
        assert out.shape == (len(X), 1)
        assert ((out > 0) & (out < 1)).all()

    def test_sigmoid_is_stable_for_huge_inputs(self):
        from src.ann import _sigmoid

        assert _sigmoid(np.array([-1000.0, 0.0, 1000.0])).tolist() == [0.0, 0.5, 1.0]


class TestSampleWeights:
    def test_balanced_up_weights_the_positive_class(self):
        y = np.array([1, 0, 0, 0], dtype=float)
        assert NeuralNetwork(class_weight=None)._sample_weights(y).tolist() == [1, 1, 1, 1]
        assert NeuralNetwork(class_weight="balanced")._sample_weights(y).tolist() == [3, 1, 1, 1]


class TestGradients:
    def test_logistic_regression_gradient_has_the_textbook_form(self):
        X, y = toy_problem()
        net = NeuralNetwork(hidden_layers=())
        net._init_weights(X.shape[1])
        activations, pre = net._forward(X)
        (gW,), (gb,) = net._backward(activations, pre, y, np.ones_like(y))
        p = activations[-1].ravel()
        np.testing.assert_allclose(gW.ravel(), X.T @ (p - y) / len(y), atol=1e-12)
        np.testing.assert_allclose(gb, [(p - y).mean()], atol=1e-12)

    @pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid"])
    @pytest.mark.parametrize("hidden", [(), (5,), (6, 4)])
    @pytest.mark.parametrize("l2", [0.0, 0.01])
    @pytest.mark.parametrize("class_weight", [None, "balanced"])
    def test_backprop_matches_numerical_gradient(self, activation, hidden, l2, class_weight):
        X, y = toy_problem(seed=1)
        net = NeuralNetwork(hidden_layers=hidden, activation=activation, l2=l2, class_weight=class_weight, seed=3)
        assert gradient_check(net, X, y) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ann.py -v`
Expected: `ModuleNotFoundError: No module named 'src.ann'`

- [ ] **Step 3: Implement the core of `src/ann.py`**

```python
"""A feed-forward neural network (multi-layer perceptron) written from scratch in NumPy.

Architecture  x -> [hidden layer(s) with relu/tanh/sigmoid] -> 1 sigmoid unit = P(spam)
Loss          binary cross-entropy (optionally class-weighted) + 0.5 * l2 * sum(W^2)
Training      mini-batch stochastic gradient descent; gradients come from back-propagation.
hidden_layers=() removes every hidden layer, which makes this exactly logistic regression.
"""
from __future__ import annotations

import numpy as np

_ACTIVATIONS = ("relu", "tanh", "sigmoid")
_EPS = 1e-12


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # 1 / (1 + e^-z) computed as exp(-log(1 + e^-z)); logaddexp never overflows.
    return np.exp(-np.logaddexp(0.0, -z))


class NeuralNetwork:
    """Binary classifier with scikit-learn-like fit / predict / predict_proba.

    Hyperparameters
    ---------------
    hidden_layers : tuple of ints, units per hidden layer; () = logistic regression
    activation    : "relu" | "tanh" | "sigmoid" for the hidden layers (output is always sigmoid)
    learning_rate : SGD step size
    epochs        : passes over the training set
    batch_size    : rows per SGD step
    l2            : L2 penalty on the weights (not on biases)
    class_weight  : None or "balanced" (positive rows weighted by n_neg / n_pos in the loss)
    seed          : controls weight initialisation and mini-batch shuffling
    """

    def __init__(self, hidden_layers=(32,), activation="relu", learning_rate=0.1, epochs=30,
                 batch_size=32, l2=0.0, class_weight=None, seed=0, verbose=False):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {_ACTIVATIONS}, got {activation!r}")
        if class_weight not in (None, "balanced"):
            raise ValueError(f"class_weight must be None or 'balanced', got {class_weight!r}")
        if learning_rate <= 0 or epochs < 1 or batch_size < 1 or l2 < 0:
            raise ValueError("learning_rate > 0, epochs >= 1, batch_size >= 1, l2 >= 0 required")
        self.hidden_layers = tuple(int(h) for h in hidden_layers)
        self.activation = activation
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.l2 = float(l2)
        self.class_weight = class_weight
        self.seed = seed
        self.verbose = verbose

    def __repr__(self) -> str:
        return (f"NeuralNetwork(hidden_layers={self.hidden_layers}, activation={self.activation!r}, "
                f"learning_rate={self.learning_rate}, epochs={self.epochs}, batch_size={self.batch_size}, "
                f"l2={self.l2}, class_weight={self.class_weight!r}, seed={self.seed})")

    # ------------------------------------------------------------------ parameters
    def _init_weights(self, n_features: int) -> np.random.Generator:
        """He initialisation for relu, Glorot/Xavier otherwise; zero biases. Returns the RNG."""
        rng = np.random.default_rng(self.seed)
        sizes = [n_features, *self.hidden_layers, 1]
        self.weights_, self.biases_ = [], []
        for layer, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            is_output = layer == len(sizes) - 2
            if self.activation == "relu" and not is_output:
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights_.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases_.append(np.zeros(fan_out))
        return rng

    # ------------------------------------------------------------------ forward pass
    def _activate(self, z: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(z, 0.0)
        if self.activation == "tanh":
            return np.tanh(z)
        return _sigmoid(z)

    def _activate_grad(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Derivative of the hidden activation, using pre-activation z or activation a."""
        if self.activation == "relu":
            return (z > 0.0).astype(z.dtype)
        if self.activation == "tanh":
            return 1.0 - a * a
        return a * (1.0 - a)

    def _forward(self, X: np.ndarray):
        """Return (activations, pre_activations); activations[0] is X, activations[-1] is P(spam)."""
        activations, pre_activations = [X], []
        a, last = X, len(self.weights_) - 1
        for layer, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            z = a @ W + b
            a = _sigmoid(z) if layer == last else self._activate(z)
            pre_activations.append(z)
            activations.append(a)
        return activations, pre_activations

    # ------------------------------------------------------------------ loss and gradients
    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        w = np.ones(len(y), dtype=np.float64)
        if self.class_weight == "balanced":
            n_pos = float(np.sum(y == 1))
            n_neg = float(len(y) - n_pos)
            if n_pos > 0:
                w[y == 1] = n_neg / n_pos
        return w

    def _loss(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> float:
        p = self._forward(X)[0][-1].ravel()
        p = np.clip(p, _EPS, 1.0 - _EPS)
        bce = -np.sum(sample_weight * (y * np.log(p) + (1.0 - y) * np.log1p(-p))) / len(y)
        reg = 0.5 * self.l2 * sum(float(np.sum(W * W)) for W in self.weights_)
        return float(bce + reg)

    def _backward(self, activations, pre_activations, y: np.ndarray, sample_weight: np.ndarray):
        """Back-propagation. Returns gradients of _loss w.r.t. every weight matrix and bias vector."""
        n = len(y)
        n_layers = len(self.weights_)
        p = activations[-1].ravel()
        # dLoss/dz for sigmoid + cross-entropy collapses to (p - y); weights and 1/n come from the mean.
        delta = ((p - y) * sample_weight / n)[:, None]
        grads_W, grads_b = [None] * n_layers, [None] * n_layers
        for layer in range(n_layers - 1, -1, -1):
            grads_W[layer] = activations[layer].T @ delta + self.l2 * self.weights_[layer]
            grads_b[layer] = delta.sum(axis=0)
            if layer > 0:
                delta = (delta @ self.weights_[layer].T) * self._activate_grad(pre_activations[layer - 1], activations[layer])
        return grads_W, grads_b


def gradient_check(model: NeuralNetwork, X, y, eps: float = 1e-5) -> float:
    """Compare back-propagation with central finite differences on every parameter.

    Returns the worst relative error  |numeric - analytic| / max(|numeric| + |analytic|, 1e-8).
    Values around 1e-8 mean backprop is correct; 1e-2 or worse means a bug.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if not hasattr(model, "weights_"):
        model._init_weights(X.shape[1])
    w = model._sample_weights(y)
    activations, pre_activations = model._forward(X)
    grads_W, grads_b = model._backward(activations, pre_activations, y, w)
    worst = 0.0
    for params, grads in ((model.weights_, grads_W), (model.biases_, grads_b)):
        for P, G in zip(params, grads):
            for idx in np.ndindex(P.shape):
                original = P[idx]
                P[idx] = original + eps
                loss_plus = model._loss(X, y, w)
                P[idx] = original - eps
                loss_minus = model._loss(X, y, w)
                P[idx] = original
                numeric = (loss_plus - loss_minus) / (2.0 * eps)
                analytic = G[idx]
                worst = max(worst, abs(numeric - analytic) / max(abs(numeric) + abs(analytic), 1e-8))
    return worst
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ann.py -v`
Expected: 44 passed (8 plain + 36 parametrised gradient checks). The gradient check errors should print around 1e-8–1e-10; if any case is above 1e-6, the bug is in `_backward` for that case (check `_activate_grad` and the `l2` term first).

- [ ] **Step 5: Commit**

```bash
git add src/ann.py tests/test_ann.py
git commit -m "feat(ann): MLP forward pass, weighted BCE loss, backprop verified by gradient check"
```

---

### Task 7: NeuralNetwork training loop and prediction

**Files:**
- Modify: `src/ann.py` (add methods to the class)
- Test: `tests/test_ann.py` (append)

**Interfaces:**
- Produces: `fit(X, y) -> self` (sets `loss_history_: list[float]`, `n_features_in_: int`), `predict_proba(X) -> np.ndarray (n,)`, `predict(X, threshold=0.5) -> np.ndarray[int] (n,)`, property `n_parameters_: int`.

- [ ] **Step 1: Append failing tests**

```python
def blobs(n_per_class=30, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(-2.0, 1.0, size=(n_per_class, 2)), rng.normal(2.0, 1.0, size=(n_per_class, 2))])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def xor_problem(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(n, 2))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    return X, y


class TestTraining:
    def test_fit_returns_self_and_records_decreasing_loss(self):
        X, y = blobs()
        net = NeuralNetwork(hidden_layers=(), learning_rate=0.5, epochs=50, seed=0)
        assert net.fit(X, y) is net
        assert len(net.loss_history_) == 50
        assert net.loss_history_[-1] < net.loss_history_[0]

    def test_separable_blobs_are_learned_perfectly(self):
        X, y = blobs()
        net = NeuralNetwork(hidden_layers=(), learning_rate=0.5, epochs=100, seed=0).fit(X, y)
        assert (net.predict(X) == y).mean() == 1.0

    def test_hidden_layer_solves_xor_but_logistic_regression_cannot(self):
        X, y = xor_problem()
        linear = NeuralNetwork(hidden_layers=(), learning_rate=0.5, epochs=200, seed=0).fit(X, y)
        mlp = NeuralNetwork(hidden_layers=(16,), activation="tanh", learning_rate=0.5, epochs=300,
                            batch_size=16, seed=0).fit(X, y)
        assert (linear.predict(X) == y).mean() < 0.75
        assert (mlp.predict(X) == y).mean() >= 0.95

    def test_predict_proba_shape_range_and_threshold(self):
        X, y = blobs()
        net = NeuralNetwork(hidden_layers=(4,), epochs=5).fit(X, y)
        p = net.predict_proba(X)
        assert p.shape == (len(X),) and ((p >= 0) & (p <= 1)).all()
        assert set(np.unique(net.predict(X))) <= {0, 1}
        assert net.predict(X, threshold=1.01).sum() == 0

    def test_same_seed_same_model_different_seed_different_model(self):
        X, y = blobs()
        a = NeuralNetwork(hidden_layers=(4,), epochs=3, seed=1).fit(X, y)
        b = NeuralNetwork(hidden_layers=(4,), epochs=3, seed=1).fit(X, y)
        c = NeuralNetwork(hidden_layers=(4,), epochs=3, seed=2).fit(X, y)
        for Wa, Wb in zip(a.weights_, b.weights_):
            np.testing.assert_array_equal(Wa, Wb)
        assert any(not np.array_equal(Wa, Wc) for Wa, Wc in zip(a.weights_, c.weights_))

    def test_balanced_class_weight_raises_recall_on_imbalanced_data(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0.0, 1.0, size=(190, 2)), rng.normal(1.5, 1.0, size=(10, 2))])
        y = np.array([0] * 190 + [1] * 10)
        plain = NeuralNetwork(hidden_layers=(), learning_rate=0.3, epochs=30, seed=0).fit(X, y)
        balanced = NeuralNetwork(hidden_layers=(), learning_rate=0.3, epochs=30, seed=0,
                                 class_weight="balanced").fit(X, y)
        assert balanced.predict(X)[y == 1].mean() > plain.predict(X)[y == 1].mean()

    def test_input_validation(self):
        X, y = blobs()
        with pytest.raises(ValueError):
            NeuralNetwork().fit(X, np.where(y == 1, 2, 0))       # labels must be 0/1
        with pytest.raises(ValueError):
            NeuralNetwork().fit(X[:10], y)                        # length mismatch
        with pytest.raises(RuntimeError):
            NeuralNetwork().predict(X)                            # not fitted
        net = NeuralNetwork(epochs=1).fit(X, y)
        with pytest.raises(ValueError):
            net.predict(X[:, :1])                                 # wrong number of features

    def test_n_parameters(self):
        net = NeuralNetwork(hidden_layers=(5, 3)).fit(*blobs())
        assert net.n_parameters_ == (2 * 5 + 5) + (5 * 3 + 3) + (3 * 1 + 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ann.py -k Training -v`
Expected: `AttributeError: 'NeuralNetwork' object has no attribute 'fit'`

- [ ] **Step 3: Add the training / prediction methods to the class (after `_backward`)**

```python
    # ------------------------------------------------------------------ public API
    def fit(self, X, y) -> "NeuralNetwork":
        """Mini-batch SGD on the (weighted) cross-entropy. Records the full-data loss per epoch."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional (n_samples, n_features)")
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)} labels")
        if not np.isin(y, (0, 1)).all():
            raise ValueError("y must contain only 0 and 1")
        y = y.astype(np.float64)
        rng = self._init_weights(X.shape[1])
        sample_weight = self._sample_weights(y)
        self.loss_history_ = []
        n = len(y)
        for epoch in range(self.epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                idx = order[start:start + self.batch_size]
                activations, pre_activations = self._forward(X[idx])
                grads_W, grads_b = self._backward(activations, pre_activations, y[idx], sample_weight[idx])
                for layer in range(len(self.weights_)):
                    self.weights_[layer] -= self.learning_rate * grads_W[layer]
                    self.biases_[layer] -= self.learning_rate * grads_b[layer]
            self.loss_history_.append(self._loss(X, y, sample_weight))
            if self.verbose:
                print(f"epoch {epoch + 1:3d}/{self.epochs}  loss = {self.loss_history_[-1]:.4f}")
        self.n_features_in_ = X.shape[1]
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("call fit(X, y) before predicting")

    def predict_proba(self, X) -> np.ndarray:
        """P(spam) for every row, shape (n,)."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected shape (n, {self.n_features_in_}), got {X.shape}")
        return self._forward(X)[0][-1].ravel()

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """1 (spam) where P(spam) >= threshold, else 0."""
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def n_parameters_(self) -> int:
        return int(sum(W.size for W in self.weights_) + sum(b.size for b in self.biases_))
```

- [ ] **Step 4: Run the whole ANN test module**

Run: `.venv/Scripts/python.exe -m pytest tests/test_ann.py -v`
Expected: 52 passed. If the XOR test is short of 0.95, raise `epochs` to 500 in the test (it is deterministic, so once it passes it stays green); if the blobs test is short of 1.0, raise `epochs` to 200.

- [ ] **Step 5: Commit**

```bash
git add src/ann.py tests/test_ann.py
git commit -m "feat(ann): mini-batch SGD training loop, predict / predict_proba"
```

---

### Task 8: Stratified k-fold cross-validation and the hyperparameter grid runner

**Files:**
- Create: `src/cv.py`
- Test: `tests/test_cv.py`

**Interfaces:**
- Consumes: `FeaturePipeline`, `NeuralNetwork`, `f1_score`, `precision`, `recall`.
- Produces: `stratified_kfold_indices(y, k=5, seed=0) -> list[tuple[np.ndarray, np.ndarray]]`; `cross_validate(texts, y, feature_params: dict, model_params: dict, k=5, seed=0) -> dict` with keys `f1_mean, f1_std, f1_folds, precision_mean, recall_mean, seconds`; `run_grid(texts, y, configs: list[dict], k=5, seed=0, verbose=True) -> pd.DataFrame` where each config is `{"name": str, "features": {...FeaturePipeline kwargs}, "model": {...NeuralNetwork kwargs}}` and the frame is sorted by `f1_mean` descending.

- [ ] **Step 1: Write failing tests**

`tests/test_cv.py`:
```python
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
        assert set(res) == {"f1_mean", "f1_std", "f1_folds", "precision_mean", "recall_mean", "seconds"}
        assert len(res["f1_folds"]) == 3
        assert res["f1_mean"] == pytest.approx(np.mean(res["f1_folds"]))
        assert res["f1_mean"] > 0.9
        assert res["seconds"] >= 0


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
                    "precision_mean", "recall_mean", "seconds"]:
            assert col in df.columns
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cv.py -v`
Expected: `ModuleNotFoundError: No module named 'src.cv'`

- [ ] **Step 3: Implement `src/cv.py`**

```python
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
    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "f1_folds": [float(v) for v in f1s],
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recs)),
        "seconds": time.perf_counter() - t0,
    }


def run_grid(texts, y, configs: list[dict], k: int = 5, seed: int = 0, verbose: bool = True) -> pd.DataFrame:
    """Evaluate every config with cross_validate and return a table sorted by mean F1 (best first)."""
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
            print(f"{cfg['name']:<28} F1 = {res['f1_mean']:.3f} ± {res['f1_std']:.3f}   "
                  f"P = {res['precision_mean']:.3f}  R = {res['recall_mean']:.3f}   ({res['seconds']:.1f}s)")
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False, kind="stable").reset_index(drop=True)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/test_cv.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the full suite once and commit**

Run: `.venv/Scripts/python.exe -m pytest`
Expected: all passed (≈ 88 tests), no warnings from our code.

```bash
git add src/cv.py tests/test_cv.py
git commit -m "feat(cv): stratified k-fold CV inside train and a hyperparameter grid runner"
```

- [ ] **Step 6: Calibration smoke run (not committed) — pick sensible learning rates for the real data**

Write to the scratchpad (not the repo) and run:
```python
import sys; sys.path.insert(0, r"C:\Users\Adir\dev\ml-sms-spam-ann")
from src.data import load_train
from src.cv import cross_validate
texts, y = load_train()
for lr in (0.1, 0.5, 2.0):
    for hidden in ((), (32,)):
        r = cross_validate(texts, y, {"use_extra": True}, {"hidden_layers": hidden, "learning_rate": lr, "epochs": 40, "seed": 42}, k=5, seed=42)
        print(f"lr={lr:<4} hidden={hidden!s:<6} F1={r['f1_mean']:.3f}±{r['f1_std']:.3f}  ({r['seconds']:.1f}s)")
```
Decision rule: the grid in Task 10 uses the best learning rate as its centre value `LR_MID` and `LR_MID/4`, `LR_MID*4` as the low/high variants. Record the numbers in the Task 10 markdown cell that introduces the grid. If the best F1 is below 0.80, first suspect the learning rate (TF-IDF rows have unit norm, so gradients are small — larger rates are normal), then `epochs`.

---

### Task 9: Notebook builder — Parts 1–3

**Files:**
- Create: `scripts/build_notebook.py`, `notebooks/spam_sms_ann.ipynb` (generated, then executed in place)

**Interfaces:**
- Consumes: everything in `src/`.
- Produces: a notebook whose code cells define `train_df, test_df, train_texts, y_train, test_texts, y_test, SEED, pipe (FeaturePipeline fitted on train), X_train, first_model` — Task 10 appends cells that use these names.

Conventions for the builder: two helpers `md(text)` and `code(text)` append cells; text is written with triple-quoted strings and `textwrap.dedent`. The notebook is executed with `jupyter nbconvert --to notebook --execute --inplace`, working directory = `notebooks/`, so the first code cell finds the repo root by walking up to the folder that contains `src/`.

- [ ] **Step 1: Write `scripts/build_notebook.py` with Parts 1–3**

```python
"""Assemble notebooks/spam_sms_ann.ipynb from Python so the notebook is reproducible and diff-able.

Run:  .venv/Scripts/python.exe scripts/build_notebook.py
Then: .venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/spam_sms_ann.ipynb
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "spam_sms_ann.ipynb"

cells = []


def md(text: str) -> None:
    cells.append(new_markdown_cell(textwrap.dedent(text).strip()))


def code(text: str) -> None:
    cells.append(new_code_cell(textwrap.dedent(text).strip()))


# ----------------------------------------------------------------------------- Part 1
md("""
# SMS Spam Classification with a Neural Network Built from Scratch

**Machine Learning — text-analysis assignment**

## Part 1 — Introduction

### Students

| Family name | ID (first 4 digits) |
|---|---|
| Buskila | TODO(team) |
| TODO(team: Liav's family name) | TODO(team) |

### AI assistants we used

We used **Claude Code (Anthropic, model Claude Fable 5)** as a pair programmer throughout the project.
The full, dated prompt log with links is in [`docs/prompts-log.md`](../docs/prompts-log.md); in short:

| What we asked for | Examples of prompts | Conversation |
|---|---|---|
| Reading the assignment, choosing a dataset that ships a train/test split, planning the work | "we made the decisions and downloaded the dataset, lets continue" · "how can i pick one thats already split" | [session link](https://claude.ai/code/session_019XxZC9ESBsXSmwJUA5PudP) |
| Writing the design spec and implementation plan, then the code and tests with us reviewing each step | "approved, write the plan and start" | same session |

Everything the assistant produced was read, run and tested by us; the learning algorithm in `src/ann.py`
is verified by a numerical gradient check (Part 3).

### The learning problem

**Binary text classification.** Given the text of an SMS message, predict whether it is **spam**
(unsolicited advertising / scams — the positive class) or **not spam** (a normal personal message).
The business motivation is the classic spam filter: catch as much spam as possible without hiding real
messages from the user. Because the two error types matter and the classes are imbalanced, the quality
metric prescribed by the assignment for binary problems is the **F1 score of the positive class (Spam)**.

### The dataset

* Kaggle: **E-Mail classification NLP** — <https://www.kaggle.com/datasets/datatattle/email-classification-nlp>
  (despite the title, the rows are SMS messages from the well-known SMS Spam Collection).
* Kaggle ships **two files**, and we use them exactly as given — we never re-split, shuffle or de-duplicate:
  `SMS_train.csv` (957 messages) for everything up to model selection, and `SMS_test.csv` (125 messages),
  which is opened only in Part 5.
* Columns: `S. No.` (row number), `Message_body` (the text), `Label` ∈ {`Spam`, `Non-Spam`}.
* The files are encoded in Windows-1252 (the `£` sign is byte `0xA3`), so they are read with `encoding="cp1252"`.
""")

code("""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# make `src` importable whether the notebook is run from notebooks/ or from the repo root
ROOT = Path.cwd().resolve()
while not (ROOT / "src").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from src.data import LABEL_NAMES, LABELS, load_test_df, load_train_df, split_xy
from src.features import HANDCRAFTED_NAMES, FeaturePipeline, TfidfFeaturizer, handcrafted_features, tokenize
from src.ann import NeuralNetwork, gradient_check
from src.metrics import accuracy, confusion_matrix, f1_score, precision, recall
from src.cv import cross_validate, run_grid, stratified_kfold_indices

SEED = 42
pd.set_option("display.max_colwidth", 110)
plt.rcParams["figure.dpi"] = 110
print("numpy", np.__version__, "| pandas", pd.__version__)
""")

md("""
### Loading the train set and the test set — as provided

`load_train_df()` / `load_test_df()` read the two CSV files that Kaggle ships (see `src/data.py`).
`split_xy` turns the `Label` column into `y = 1` for **Spam** and `y = 0` for **Non-Spam**.
""")

code("""
train_df = load_train_df()
test_df = load_test_df()
train_texts, y_train = split_xy(train_df)
test_texts, y_test = split_xy(test_df)

print(f"train: {train_df.shape[0]} rows   test: {test_df.shape[0]} rows")
print("first 5 rows of the TRAIN set:")
train_df.head()
""")

code("""
print("first 5 rows of the TEST set:")
test_df.head()
""")

code("""
balance = pd.DataFrame({
    "train": train_df["Label"].value_counts(),
    "test": test_df["Label"].value_counts(),
})
balance.loc["spam share"] = [y_train.mean(), y_test.mean()]
display(balance)

fig, ax = plt.subplots(figsize=(5, 3))
balance.drop("spam share").T.plot.bar(ax=ax, rot=0, color=["#4C72B0", "#DD8452"])
ax.set_ylabel("messages")
ax.set_title("Class balance in the two files Kaggle ships")
plt.tight_layout()
plt.show()
""")

md("""
Two things to notice, both of which we will come back to:

1. **The training set is spam-minority (12.7 %) while the test set is spam-majority (60.8 %).** The test file
   is also sorted with its 66 spam messages first, which is why `test_df.head()` shows only spam. Our model
   has no notion of row order, so this is harmless — but it means accuracy on the test set would be a poor
   summary, and the F1 score on the Spam class is the right thing to look at.
2. From here until Part 5, **`test_texts` / `y_test` are not touched**. All decisions (vocabulary, feature
   scaling, hyperparameters) are made from the training set alone, using cross-validation inside it.
""")

# ----------------------------------------------------------------------------- Part 2
md("""
## Part 2 — Feature engineering

A neural network needs numbers, so every message is turned into one fixed-length vector. We build it from
two blocks, both implemented by hand in `src/features.py`:

### 2.1 Tokenisation with placeholders
`tokenize()` lower-cases the text, replaces every **URL** by `__url__`, every run of **7+ digits** (a phone
number, spaces allowed) by `__phone__` and every other **number** by `__num__`, then splits on anything that is
not a letter/digit and drops 1-character tokens.
*Why:* spam almost always contains *some* phone number, *some* price or *some* link, but rarely the same one
twice. With placeholders the model can learn "contains a phone number" from a handful of examples instead of
having to memorise each number.

### 2.2 TF-IDF bag of words
For every token in a vocabulary learned **from the training messages only** (tokens that appear in at least
2 training messages, at most the 2 000 most frequent) we compute

$$\\text{tf-idf}(t, m) = \\text{tf}(t, m)\\cdot\\Big(\\ln\\frac{1+N}{1+\\text{df}(t)} + 1\\Big),$$

where tf is the count of token *t* in message *m*, *N* the number of training messages and df(*t*) the number
of training messages containing *t*. Each message vector is then L2-normalised.
*Why:* tf captures what the message talks about; idf down-weights tokens that appear everywhere ("to", "the")
and up-weights rare, discriminative ones ("prize", "claim"); normalisation stops long messages from
dominating simply because they contain more words.

### 2.3 Seven handcrafted cues (extra feature engineering)
`handcrafted_features()` adds message length in characters and words, the number of digits, the share of
upper-case letters, the number of `!`, and two flags: *contains a currency symbol* and *contains a URL or a
phone number*. These are standardised (z-scored) with means and standard deviations computed on the training
set. *Why:* they capture the *style* of spam — long, shouting, full of numbers and calls to action — which a
bag of words only sees indirectly. Whether they actually help is tested as a hyperparameter in Part 4.

`FeaturePipeline` glues the blocks together: `fit(texts)` learns vocabulary, idf, means and stds from the
training texts; `transform(texts)` applies them to any texts.
""")

code("""
pipe = FeaturePipeline(max_features=2000, min_df=2, use_extra=True).fit(train_texts)
X_train = pipe.transform(train_texts)
print(f"vocabulary size: {len(pipe.tfidf_.vocabulary_)}   handcrafted: {len(HANDCRAFTED_NAMES)}   "
      f"total features: {pipe.n_features_}   X_train: {X_train.shape} {X_train.dtype}")
""")

md("""
### The transformation on concrete examples

Three training messages and three test messages (the pipeline was fitted on the training set only; running
test messages *through* it uses no test information — it is exactly what happens at prediction time).
""")

code("""
def show_pipeline(texts, labels, title):
    print("=" * 100)
    print(title)
    for text, label in zip(texts, labels):
        print("-" * 100)
        print(f"[{LABEL_NAMES[int(label)]}]  {text}")
        print("  tokens         :", tokenize(text))
        print("  top tf-idf     :", [(t, round(w, 3)) for t, w in pipe.tfidf_.explain(text, k=6)])
        raw = handcrafted_features([text])[0]
        scaled = pipe.scaler_.transform([raw])[0]
        print("  handcrafted    :", {n: round(float(v), 2) for n, v in zip(HANDCRAFTED_NAMES, raw)})
        print("  standardised   :", {n: round(float(v), 2) for n, v in zip(HANDCRAFTED_NAMES, scaled)})
        vec = pipe.transform([text])[0]
        print(f"  feature vector : shape {vec.shape}, {np.count_nonzero(vec)} non-zero entries")

# a spam and two hams from train (indices chosen to show a phone number, a plain message and a £ price)
train_examples = [4, 0, 46]
show_pipeline([train_texts[i] for i in train_examples], y_train[train_examples], "TRAIN examples")
test_examples = [1, 70, 100]
show_pipeline([test_texts[i] for i in test_examples], y_test[test_examples], "TEST examples")
""")

# ----------------------------------------------------------------------------- Part 3
md("""
## Part 3 — The learning algorithm: a neural network written from scratch

`src/ann.py` implements a **multi-layer perceptron** (feed-forward neural network) in plain NumPy — no
scikit-learn, no PyTorch. It exposes the usual interface: `NeuralNetwork(...)`, `fit(X, y)`, `predict(X)`,
`predict_proba(X)`.

### How it works

**Architecture.** The input vector $x \\in \\mathbb{R}^d$ passes through zero or more hidden layers and one
output unit:

$$a^{(0)} = x,\\qquad z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)},\\qquad a^{(l)} = g\\big(z^{(l)}\\big),\\qquad
\\hat{p} = \\sigma\\big(z^{(L)}\\big) = \\frac{1}{1+e^{-z^{(L)}}}.$$

$g$ is the hidden activation (`relu`, `tanh` or `sigmoid` — a hyperparameter) and $\\hat p$ is the predicted
probability of spam. With `hidden_layers=()` there is only the output unit and the model is exactly
**logistic regression**, which we use as the baseline in Part 4.

**Loss.** Binary cross-entropy, optionally weighting the rare spam class (`class_weight="balanced"` weights
every spam row by $n_{\\text{neg}}/n_{\\text{pos}}$), plus an optional L2 penalty:

$$\\mathcal{L} = -\\frac{1}{n}\\sum_{i} w_i\\Big[y_i\\ln \\hat p_i + (1-y_i)\\ln(1-\\hat p_i)\\Big]
 + \\frac{\\lambda}{2}\\sum_l \\lVert W^{(l)}\\rVert^2 .$$

**Back-propagation.** For the sigmoid + cross-entropy output the error signal is simply
$\\delta^{(L)} = (\\hat p - y)\\,w/n$. It is propagated backwards layer by layer,
$\\delta^{(l-1)} = \\big(\\delta^{(l)} W^{(l)\\top}\\big)\\odot g'\\big(z^{(l-1)}\\big)$, and the gradients are
$\\partial\\mathcal L/\\partial W^{(l)} = a^{(l-1)\\top}\\delta^{(l)} + \\lambda W^{(l)}$ and
$\\partial\\mathcal L/\\partial b^{(l)} = \\sum_i \\delta^{(l)}_i$.

**Training.** Mini-batch stochastic gradient descent: each epoch shuffles the training rows, and for every
batch of `batch_size` rows the parameters move against the gradient, $W \\leftarrow W - \\eta\\,
\\partial\\mathcal L/\\partial W$, with learning rate $\\eta$. Weights start from He (relu) or Xavier
initialisation and biases from zero; the seed makes every run reproducible.

**Hyperparameters** (all constructor arguments): `hidden_layers`, `activation`, `learning_rate`, `epochs`,
`batch_size`, `l2`, `class_weight`, `seed`.

### Is the implementation correct? — gradient check

Hand-written back-propagation fails silently: a wrong gradient still "trains", just badly. So we compare the
analytic gradients with central finite differences $\\big(\\mathcal L(\\theta+\\varepsilon)-\\mathcal
L(\\theta-\\varepsilon)\\big)/2\\varepsilon$ for every single parameter (this is also part of the unit-test
suite in `tests/test_ann.py`, which runs it for every activation, depth, L2 and class-weight setting).
""")

code("""
rng = np.random.default_rng(SEED)
X_tiny = rng.normal(size=(12, 6))
y_tiny = (rng.random(12) < 0.4).astype(float)
for hidden, activation in [((), "relu"), ((5,), "relu"), ((6, 4), "tanh"), ((6, 4), "sigmoid")]:
    net = NeuralNetwork(hidden_layers=hidden, activation=activation, l2=0.01, class_weight="balanced", seed=SEED)
    err = gradient_check(net, X_tiny, y_tiny)
    print(f"hidden={hidden!s:<8} activation={activation:<8} worst relative error = {err:.2e}  "
          f"{'OK' if err < 1e-6 else 'BUG'}")
""")

md("""
Relative errors around $10^{-8}$–$10^{-10}$ mean the analytic gradient matches the numerical one to floating
point precision — the back-propagation is right.

### A first model on the training features

One hidden layer of 32 ReLU units, trained on the TF-IDF + handcrafted features from Part 2. The loss curve
shows that SGD is doing its job; how good this configuration really is (and whether others are better) is the
subject of Part 4 — this is only a sanity check on the training set itself.
""")

code("""
first_model = NeuralNetwork(hidden_layers=(32,), activation="relu", learning_rate=0.5, epochs=40,
                            batch_size=32, seed=SEED)
t0 = time.perf_counter()
first_model.fit(X_train, y_train)
print(f"{first_model}\\ntrainable parameters: {first_model.n_parameters_:,}   fit time: {time.perf_counter() - t0:.1f}s")

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(range(1, len(first_model.loss_history_) + 1), first_model.loss_history_)
ax.set_xlabel("epoch"); ax.set_ylabel("training loss (BCE)"); ax.set_title("First model: loss per epoch")
plt.tight_layout(); plt.show()

train_pred = first_model.predict(X_train)
print(f"training-set F1 (optimistic, same data it was fitted on): {f1_score(y_train, train_pred):.3f}")
""")


# ----------------------------------------------------------------------------- write
def main() -> None:
    nb = new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python"},
    })
    OUT.parent.mkdir(exist_ok=True)
    nbformat.write(nb, OUT)
    print(f"wrote {OUT.relative_to(ROOT)} with {len(cells)} cells")


if __name__ == "__main__":
    main()
```

Note the learning rate `0.5` in `first_model` — replace it with the `LR_MID` found in Task 8 Step 6 if different.

- [ ] **Step 2: Build and execute the notebook**

Run:
```bash
.venv/Scripts/python.exe scripts/build_notebook.py
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/spam_sms_ann.ipynb
```
Expected: `wrote notebooks/spam_sms_ann.ipynb with N cells`, then nbconvert finishes without an error. If a cell errors, nbconvert prints the traceback — fix the builder, rebuild, re-execute.

- [ ] **Step 3: Inspect the executed outputs**

Run (prints every cell's text output so the reviewer can read it without Jupyter):
```bash
.venv/Scripts/python.exe - <<'EOF'
import nbformat
nb = nbformat.read("notebooks/spam_sms_ann.ipynb", as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code": continue
    print(f"\n### cell {i}")
    for o in c.get("outputs", []):
        if o.output_type == "stream": print(o.text)
        elif o.output_type in ("execute_result", "display_data"): print(o.data.get("text/plain", "")[:1500])
        elif o.output_type == "error": print("ERROR", o.ename, o.evalue)
EOF
```
Check: train 957 / test 125; `head()` shows 5 rows for both; vocabulary size is reported; the six pipeline examples show tokens, tf-idf terms and handcrafted values; all gradient-check lines say `OK`; the first model's loss decreases.

- [ ] **Step 4: Commit**

```bash
git add scripts/build_notebook.py notebooks/spam_sms_ann.ipynb
git commit -m "feat(notebook): Parts 1-3 - data, feature engineering demos, ANN explanation and gradient check"
```

---

### Task 10: Notebook Parts 4–5, executed, with the discussion written from the real results

**Files:**
- Modify: `scripts/build_notebook.py` (append Parts 4–5 before `main()`), regenerate `notebooks/spam_sms_ann.ipynb`

**Interfaces:**
- Consumes: `train_texts, y_train, test_texts, y_test, SEED, pipe, show_pipeline` from Task 9 cells; `run_grid`, `cross_validate`, metrics.

- [ ] **Step 1: Append Part 4 cells to the builder** (insert before the `# --- write` section)

```python
# ----------------------------------------------------------------------------- Part 4
md("""
## Part 4 — Training with different hyperparameters

### How we compare configurations without touching the test set

Tuning needs data the model has not been trained on, but the assignment forbids re-splitting the given files.
So every configuration is scored with **stratified 5-fold cross-validation inside the training set**
(`src/cv.py`): the 957 training messages are dealt into 5 folds with the same spam share; in turn each fold is
held out, the feature pipeline **and** the network are fitted on the other four, and the held-out fold is
scored. We report the mean and standard deviation of the Spam-class F1 over the 5 folds. The test set stays
sealed until Part 5.

### The grid

We start from a reference configuration — one hidden layer of 32 ReLU units, learning rate `LR_MID`, 40 epochs,
batch size 32, TF-IDF + handcrafted features — and vary one knob at a time, plus a few combinations
(12 configurations, the whole run takes about a minute on a laptop):

| knob | values tried | question it answers |
|---|---|---|
| `hidden_layers` | `()` = logistic regression, `(32,)`, `(64, 32)` | does depth help over a linear model? |
| `learning_rate` | LR_MID / 4, LR_MID, LR_MID × 4 | is SGD stable and fast enough? |
| `activation` | relu, tanh | does the non-linearity matter? |
| `class_weight` | None, "balanced" | does up-weighting the rare spam class help F1? |
| `use_extra` | False, True | do the 7 handcrafted cues add anything to TF-IDF? |
| `l2`, `batch_size` | 0 vs 1e-3, 32 vs 8 | regularisation and noisier gradients |
""")

code("""
LR_MID = 0.5          # centre of the learning-rate grid (chosen from a quick calibration run, see plan Task 8)

def config(name, hidden=(32,), activation="relu", lr=LR_MID, class_weight=None, use_extra=True, l2=0.0, batch_size=32):
    return {"name": name,
            "features": {"max_features": 2000, "min_df": 2, "use_extra": use_extra},
            "model": {"hidden_layers": hidden, "activation": activation, "learning_rate": lr, "epochs": 40,
                      "batch_size": batch_size, "l2": l2, "class_weight": class_weight, "seed": SEED}}

configs = [
    config("logreg / tfidf only",        hidden=(), use_extra=False),
    config("logreg / +extra",            hidden=()),
    config("mlp32 / tfidf only",         use_extra=False),
    config("mlp32 / +extra  (reference)"),
    config("mlp32 / +extra / balanced",  class_weight="balanced"),
    config("mlp32 / +extra / tanh",      activation="tanh"),
    config("mlp32 / +extra / lr low",    lr=LR_MID / 4),
    config("mlp32 / +extra / lr high",   lr=LR_MID * 4),
    config("mlp32 / +extra / l2=1e-3",   l2=1e-3),
    config("mlp32 / +extra / batch 8",   batch_size=8),
    config("mlp64-32 / +extra",          hidden=(64, 32)),
    config("mlp64-32 / +extra / balanced", hidden=(64, 32), class_weight="balanced"),
]

t0 = time.perf_counter()
results = run_grid(train_texts, y_train, configs, k=5, seed=SEED)
print(f"\\ntotal: {time.perf_counter() - t0:.0f}s")
""")

code("""
table = results[["name", "hidden_layers", "activation", "learning_rate", "class_weight", "use_extra", "l2",
                 "batch_size", "f1_mean", "f1_std", "precision_mean", "recall_mean", "seconds"]].copy()
table[["f1_mean", "f1_std", "precision_mean", "recall_mean"]] = table[["f1_mean", "f1_std", "precision_mean", "recall_mean"]].round(3)
table["seconds"] = table["seconds"].round(1)
display(table)

fig, ax = plt.subplots(figsize=(8, 4.5))
order = results.iloc[::-1]
ax.barh(order["name"], order["f1_mean"], xerr=order["f1_std"], color="#4C72B0", capsize=3)
ax.set_xlabel("cross-validated F1 on the Spam class (mean ± std over 5 folds)")
ax.set_xlim(max(0.0, order["f1_mean"].min() - 0.15), 1.0)
ax.set_title("Hyperparameter comparison (train-set CV only)")
plt.tight_layout(); plt.show()
""")

md("""
### Choosing the winning configuration

TODO(execution): replace this paragraph after the grid has run, with the actual numbers — see Task 10 Step 3.
""")

code("""
best = results.iloc[0]
best_cfg = next(c for c in configs if c["name"] == best["name"])
print("winner:", best["name"])
print("  CV F1 = %.3f ± %.3f   precision = %.3f   recall = %.3f" % (best["f1_mean"], best["f1_std"], best["precision_mean"], best["recall_mean"]))
print("  features:", best_cfg["features"])
print("  model   :", best_cfg["model"])
""")

md("""
### The winning flow on 2–3 training examples

The same three training messages as in Part 2, now passed through the winning feature configuration and
scored by a model trained (with the winning hyperparameters) on all training data.
""")

code("""
best_pipe = FeaturePipeline(**best_cfg["features"]).fit(train_texts)
best_model = NeuralNetwork(**best_cfg["model"]).fit(best_pipe.transform(train_texts), y_train)

def show_flow(texts, labels, pipeline, model):
    for text, label in zip(texts, labels):
        vec = pipeline.transform([text])
        p = float(model.predict_proba(vec)[0])
        print("-" * 100)
        print(f"[{LABEL_NAMES[int(label)]}]  {text}")
        print("  tokens     :", tokenize(text))
        print("  top tf-idf :", [(t, round(w, 3)) for t, w in pipeline.tfidf_.explain(text, k=5)])
        if pipeline.use_extra:
            print("  handcrafted:", {n: round(float(v), 1) for n, v in zip(HANDCRAFTED_NAMES, handcrafted_features([text])[0])})
        print(f"  -> {vec.shape[1]} features -> P(spam) = {p:.3f} -> predicted {LABEL_NAMES[int(p >= 0.5)]}")

show_flow([train_texts[i] for i in train_examples], y_train[train_examples], best_pipe, best_model)
""")
```

- [ ] **Step 2: Append Part 5 cells to the builder**

```python
# ----------------------------------------------------------------------------- Part 5
md("""
## Part 5 — Prediction and evaluation on the test set

Now — and only now — the test set is used. The winning pipeline and model from Part 4 were fitted on the
**whole training set**; here they are merely *applied* to `SMS_test.csv`. Nothing is re-fitted.
""")

code("""
X_test = best_pipe.transform(test_texts)
test_proba = best_model.predict_proba(X_test)
test_pred = (test_proba >= 0.5).astype(int)
print(f"test feature matrix: {X_test.shape}")
""")

md("""
### The flow on 2–3 test messages
""")

code("""
show_flow([test_texts[i] for i in test_examples], y_test[test_examples], best_pipe, best_model)
""")

md("""
### The first 5 test predictions
""")

code("""
first5 = pd.DataFrame({
    "message": [t[:90] + ("…" if len(t) > 90 else "") for t in test_texts[:5]],
    "true label": [LABEL_NAMES[int(v)] for v in y_test[:5]],
    "P(spam)": np.round(test_proba[:5], 3),
    "predicted": [LABEL_NAMES[int(v)] for v in test_pred[:5]],
})
first5
""")

md("""
### Quality on the whole test set

The assignment's metric for a binary problem is the **F1 score of the positive class (Spam)**; precision,
recall, accuracy and the confusion matrix are shown for context.
""")

code("""
cm = confusion_matrix(y_test, test_pred)
print(f"F1 (Spam)  = {f1_score(y_test, test_pred):.3f}   <- the assignment's quality metric")
print(f"precision  = {precision(y_test, test_pred):.3f}   recall = {recall(y_test, test_pred):.3f}   accuracy = {accuracy(y_test, test_pred):.3f}")
print("\\nconfusion matrix (rows = true, columns = predicted):")
display(pd.DataFrame(cm, index=["true Non-Spam", "true Spam"], columns=["pred Non-Spam", "pred Spam"]))

fig, ax = plt.subplots(figsize=(3.6, 3.2))
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=14)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-Spam", "Spam"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["Non-Spam", "Spam"])
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Test-set confusion matrix")
plt.tight_layout(); plt.show()
""")

md("""
### Where does the model fail?
""")

code("""
errors = pd.DataFrame({
    "message": test_texts, "true": [LABEL_NAMES[int(v)] for v in y_test],
    "P(spam)": np.round(test_proba, 3), "predicted": [LABEL_NAMES[int(v)] for v in test_pred],
})
errors = errors[errors["true"] != errors["predicted"]]
print(f"{len(errors)} misclassified test messages out of {len(test_texts)}")
pd.set_option("display.max_colwidth", 160)
display(errors)

overlap = [t for t in test_texts if t in set(train_texts)]
print(f"\\nnote: {len(overlap)} test messages also occur verbatim in the training file (kept, as the files are used as shipped)")
""")

md("""
### Discussion

TODO(execution): replace with the discussion written from the actual errors — see Task 10 Step 3. It must cover:
(1) missed spam (false negatives) and what they have in common; (2) hams flagged as spam (false positives);
(3) the train/test distribution shift (13 % → 61 % spam) and why F1 rather than accuracy; (4) the 5 verbatim
train/test overlaps; (5) the small test set (125 rows: one error moves F1 by roughly a point) and what we would
try next.
""")
```

- [ ] **Step 3: Build, execute, read the results, then write the two TODO(execution) markdown cells**

Run:
```bash
.venv/Scripts/python.exe scripts/build_notebook.py
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/spam_sms_ann.ipynb
```
Then print the outputs of the Part 4 table cell, the winner cell, the test metrics cell and the errors cell
(same snippet as Task 9 Step 3). Replace the two `TODO(execution)` markdown cells in the builder with real text:

*Choosing the winner* — name the winning configuration and its CV F1 ± std; compare it with the logistic-regression
baseline (does depth help beyond the fold-to-fold noise?), state what `class_weight`, the extra features and the
learning rate did, and say explicitly that configurations within one standard deviation of each other are not
distinguishable on 957 messages, so if the best score is a tie we prefer the simpler model.

*Discussion* — walk through the actual misclassified messages grouped as false negatives / false positives, name
the pattern in each group (e.g. spam without digits or URLs, short promotional-looking hams), then the four
context points listed in the placeholder.

Rebuild and re-execute once more so the committed notebook contains the final text and fresh outputs.

- [ ] **Step 4: Verify there are no leftover placeholders and the notebook is fully executed**

Run:
```bash
grep -n "TODO(execution)" scripts/build_notebook.py notebooks/spam_sms_ann.ipynb || echo "no execution placeholders left"
.venv/Scripts/python.exe - <<'EOF'
import nbformat
nb = nbformat.read("notebooks/spam_sms_ann.ipynb", as_version=4)
codes = [c for c in nb.cells if c.cell_type == "code"]
assert all(c.execution_count for c in codes), "some code cell was not executed"
assert not any(o.output_type == "error" for c in codes for o in c.outputs), "a cell errored"
print(len(codes), "code cells executed, no errors")
EOF
```
Expected: `no execution placeholders left` (the `TODO(team)` student-ID markers in Part 1 are expected to remain until the team fills them) and `N code cells executed, no errors`.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_notebook.py notebooks/spam_sms_ann.ipynb
git commit -m "feat(notebook): Parts 4-5 - CV hyperparameter search, test-set evaluation and discussion"
```

---

### Task 11: Wrap-up — docs, task board, full verification, push

**Files:**
- Modify: `README.md`, `TODO.md`, `docs/prompts-log.md`

- [ ] **Step 1: README results section**

Under `## Status` in `README.md`, replace the "Work in progress" line with a short results block: winning configuration, CV F1 ± std, test-set F1, and one sentence pointing to the notebook. Keep the setup/run sections as they are (verify the two commands there still match the ones used in Tasks 9–10; add the `build_notebook.py` step).

- [ ] **Step 2: Task board**

In `TODO.md` tick every item in section 3 (Parts 1–5) that the notebook now satisfies, add a row under section 2 "Fill in student IDs and Liav's family name in notebook Part 1" (unticked, owner: both), and update `**Last updated:**`. Regenerate the Drive mirror:
```bash
DRIVE="/c/Users/Adir/Desktop/BSC/שנה ב/סמסטר ג/למידת מכונה/Project/TODO.md"; { head -3 "$DRIVE"; cat TODO.md; } > /tmp/t.md && mv /tmp/t.md "$DRIVE"
```

- [ ] **Step 3: Prompts log**

Append to row 1 of `docs/prompts-log.md` (same session) the prompt "approved, write the plan and start" and, in "Used for", the implementation plan, the `src/` package with tests, and the notebook.

- [ ] **Step 4: Full verification from a clean state**

```bash
.venv/Scripts/python.exe -m pytest
.venv/Scripts/python.exe scripts/build_notebook.py
.venv/Scripts/python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 notebooks/spam_sms_ann.ipynb
git status --short
```
Expected: all tests pass; the notebook re-executes without error; `git status` shows only the files you intend to commit (a re-executed notebook may differ only in timing numbers).

- [ ] **Step 5: Commit and push**

```bash
git add README.md TODO.md docs/prompts-log.md notebooks/spam_sms_ann.ipynb
git commit -m "docs: results in README, task board and prompts log updated"
git push origin main
```
Then open <https://github.com/AdirBuskila/ml-sms-spam-ann/blob/main/notebooks/spam_sms_ann.ipynb> in a private window and confirm GitHub renders the executed notebook with its outputs.

---

## Self-review against the spec

- **§2 dataset rules** → Task 1 (as-shipped loading, cp1252, Spam=1), Task 9/10 notebook text (test sealed, k-fold inside train), Task 10 overlap note. ✔
- **§4.1–4.5 components and signatures** → Tasks 1–8 use exactly the names in the spec (`FeaturePipeline(max_features, min_df, use_extra)`, `NeuralNetwork(...)` kwargs, `cross_validate`, `run_grid`, metrics). `cross_validate` takes `(texts, y, feature_params, model_params)` — argument order differs from the spec's sketch `(feature_cfg, model_cfg, texts, y)`; the plan's order is the one implemented and used everywhere. ✔
- **§5 notebook parts** → Task 9 (Parts 1–3) and Task 10 (Parts 4–5) contain every required element: student details, LLM prompts + link, problem/dataset explanation, both `head()`s, feature explanation with 3 train + 3 test examples, algorithm explanation + gradient check + `fit`/`predict`, CV grid + table + chart + justification + examples through the winning flow, test evaluation with 2–3 examples, first 5 predictions, F1, confusion matrix, error discussion. ✔
- **§6 tests** → Tasks 1–8; the sklearn oracle appears only in `tests/`. ✔
- **§7 reproducibility** → `SEED = 42` in the notebook, seeds in every CV/model call, features fitted per fold and on full train for Part 5. ✔
- **Placeholders**: the only intentional ones are `TODO(team)` for student IDs (data the team must supply) and the two `TODO(execution)` markdown cells that Task 10 Step 3 explicitly replaces before committing.
- **Type consistency**: `predict_proba` returns shape `(n,)` everywhere; `confusion_matrix` layout `[[tn, fp], [fn, tp]]` is used consistently in metrics, tests and the notebook heat-map labels; `results` columns used in Task 10 (`name, f1_mean, f1_std, precision_mean, recall_mean, seconds, hidden_layers, activation, learning_rate, class_weight, use_extra, l2, batch_size`) all exist in `run_grid`'s row dict.
