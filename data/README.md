# Data

## Source

- **Kaggle dataset:** "E-Mail classification NLP" by *datatattle* —
  <https://www.kaggle.com/datasets/datatattle/email-classification-nlp>
- Despite the title, the rows are **SMS messages** (the UCI SMS Spam Collection family).
- Downloaded **2026-09-01** through the Kaggle website (Download button → `archive.zip`), unzipped,
  and the two CSVs were placed in `data/raw/` **unchanged** (byte-identical to the download).
- License shown on Kaggle: "Data files © Original Authors". Used here for coursework only.

## Files (as shipped by Kaggle — this is the train/test split we use)

| file | bytes | rows | sha256 (first 16 hex) |
|---|---|---|---|
| `raw/SMS_train.csv` | 89,023 | 957 | `f19d4cb3e7161a4a` |
| `raw/SMS_test.csv`  | 16,502 | 125 | `4765e211fc5d3763` |

Columns (both files): `S. No.` (1-based row number, unused) · `Message_body` (the SMS text) ·
`Label` ∈ {`Spam`, `Non-Spam`}. Positive class for our metric: **Spam → 1**, Non-Spam → 0.

## Encoding

The files are **Windows-1252 (cp1252), not UTF-8**. Decoding as UTF-8 fails on the first `£`
(byte `0xA3`); other non-ASCII bytes are `ü` (`0xFC`), `Ü`, `é`, curly quotes (`0x91`/`0x92`) and
an en dash (`0x96`). Always load with `encoding="cp1252"`. Two training messages contain a line
break inside the quoted body; the `csv` module and `pandas.read_csv` handle it correctly.

## What the data looks like

| | train | test |
|---|---|---|
| rows | 957 | 125 |
| Non-Spam / Spam | 835 / 122 | 49 / 76 |
| spam share | 12.7 % | **60.8 %** |
| median message length (chars) | 59 | 142 |
| duplicate messages inside the file | 12 | 1 |

## Quirks we must disclose in the notebook (we do **not** "fix" them — files are used as shipped)

1. **Distribution shift.** Train is spam-minority (13 %), test is spam-majority (61 %). F1 on the
   Spam class remains the right metric; accuracy on the test set would be misleading.
2. **Test rows are sorted.** Rows 1–66 of `SMS_test.csv` are all Spam, the rest mostly Non-Spam.
   Our model has no order dependence, but it means "the first 5 rows / first 5 predictions" shown
   in the notebook are all spam.
3. **5 of the 125 test messages also occur verbatim in the training set** (3 Spam, 2 Non-Spam,
   labels agree). They slightly flatter the test score; we say so in Part 5 rather than delete rows.
4. Test messages are longer than training messages (median 142 vs 59 chars) — a consequence of 1.,
   since spam is long.

## Rules

- Never re-split, shuffle across the boundary, de-duplicate, or otherwise edit these files
  (assignment requirement: use the given train/test split as-is).
- The test set is read in Part 1 only to show its first rows, and is next touched in Part 5.
- Hyperparameter tuning uses k-fold cross-validation inside `SMS_train.csv` only.

## Re-downloading

Download from the URL above (Kaggle login required), unzip, copy `SMS_train.csv` and
`SMS_test.csv` into `data/raw/`, and confirm the hashes:
`certutil -hashfile data\raw\SMS_train.csv SHA256` (Windows) or `sha256sum data/raw/*.csv`.
