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
