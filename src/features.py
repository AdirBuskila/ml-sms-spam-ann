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

    def _check_fitted(self) -> None:
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
        idx = np.argsort(-row, kind="stable")
        return [(self.feature_names_[j], float(row[j])) for j in idx[:k] if row[j] > 0]


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
