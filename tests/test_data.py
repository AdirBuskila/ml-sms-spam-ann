import numpy as np
import pandas as pd
import pytest

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
    assert not any("�" in t for t in texts)      # no replacement characters


def test_row_order_is_preserved():
    texts, _ = load_train()
    assert texts[0].startswith("Rofl. Its true to its name")
    test_texts, _ = load_test()
    assert test_texts[0].startswith("UpgrdCentre Orange customer")


def test_split_xy_rejects_unknown_labels():
    df = pd.DataFrame({"S. No.": [1], "Message_body": ["hi"], "Label": ["Maybe"]})
    with pytest.raises(ValueError):
        split_xy(df)


def test_label_map_constant():
    assert LABELS == {"Non-Spam": 0, "Spam": 1}
