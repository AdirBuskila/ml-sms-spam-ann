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
