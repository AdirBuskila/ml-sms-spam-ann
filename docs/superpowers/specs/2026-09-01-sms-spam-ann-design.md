# SMS Spam Classification with a From-Scratch Neural Network — Design

**Date:** 2026-09-01 · **Team:** Adir + Liav · **Deadline:** 2026-09-10
**Assignment:** Machine Learning course, "ML with image processing / text analysis" (`docs/assignment.pdf`, kept local, not committed)

## 1. Goal

Deliver one Jupyter notebook (plus a small, tested Python package it imports) that follows the
assignment's five mandatory parts exactly, for a **binary text-classification** problem:
detect whether an SMS is **spam** (positive class) or **not spam**.

Quality metric (assignment rule for binary classification): **F1 on the positive class (Spam)**.

Everything the grader looks for, in one place:

| Requirement (PDF) | Where it is satisfied |
|---|---|
| Kaggle dataset with a **given** train/test split; never re-split | §2 — two files shipped by Kaggle, loaded as-is |
| Show first 5 rows of the dataset | Notebook Part 1 |
| Feature engineering + demo on 2–3 train and test examples | Notebook Part 2, `src/features.py` |
| Own implementation of a recent algorithm with `fit`/`predict` | `src/ann.py` (`NeuralNetwork`) |
| Configurable hyperparameters, several combinations compared | Notebook Part 4, `src/cv.py` |
| Test-set prediction, first 5 predictions, metric, discussion | Notebook Part 5 |
| LLM prompts logged with links | `docs/prompts-log.md` |
| Code + run outputs in the repo | executed notebook committed |

## 2. Dataset

**Kaggle:** <https://www.kaggle.com/datasets/datatattle/email-classification-nlp>
(title says "E-Mail", content is SMS; same family as the UCI SMS Spam Collection).

- Files shipped by Kaggle: `SMS_train.csv` (~950 rows) and `SMS_test.csv` (125 rows).
- Columns: `S. No.`, `Message_body`, `Label` with values `Spam` / `Non-Spam`.
- Known quirks (handled at load time, documented in the notebook): non-UTF-8 bytes
  (`£` appears as `�` in Kaggle's preview) → load with an explicit encoding and verify;
  the test set is ~61 % spam while train is spam-minority → mention as distribution shift in Part 5.

**Rules we impose on ourselves**

1. The two CSVs are loaded exactly as downloaded (`data/raw/`). No re-splitting, no shuffling
   across the boundary, no de-duplication across files.
2. The test set is opened **once**, in Part 5. All model selection uses **stratified k-fold
   cross-validation inside the training set only** (k = 5). The notebook says this explicitly.
3. Positive class is `Spam` → `y = 1`; `Non-Spam` → `y = 0`.
4. Both raw CSVs are committed (≈105 kB total) so graders can run the notebook without a Kaggle
   login. `data/README.md` records the source URL and download date.

## 3. Repository and environment

Local clone lives **outside Google Drive**: `C:\Users\Adir\dev\ml-sms-spam-ann`
(GitHub: `AdirBuskila/ml-sms-spam-ann`, public). The Drive folder keeps only `TODO.md` with a
pointer to the repo.

```
ml-sms-spam-ann/
├── README.md                  how to set up, test, and run
├── requirements.txt           numpy, pandas, matplotlib, jupyter; pytest + scikit-learn for tests only
├── TODO.md                    task board (moved from Drive)
├── data/
│   ├── README.md              provenance, URL, download date, column description
│   └── raw/SMS_train.csv, SMS_test.csv     exactly as downloaded from Kaggle
├── src/
│   ├── __init__.py
│   ├── data.py                load_train() / load_test(): read CSV, map labels, return (texts, y)
│   ├── features.py            tokenizer, TfidfFeaturizer, handcrafted features, Standardizer, FeaturePipeline
│   ├── ann.py                 NeuralNetwork — the hand-written algorithm (fit / predict / predict_proba)
│   ├── metrics.py             confusion_matrix, precision, recall, f1_score, accuracy
│   └── cv.py                  stratified_kfold_indices, cross_validate, run_grid
├── tests/                     pytest; sklearn used only as an oracle
├── notebooks/
│   └── spam_sms_ann.ipynb     THE graded deliverable — 5 parts, executed, outputs committed
└── docs/
    ├── prompts-log.md         LLM prompts, links, what they were used for (kept as we go)
    └── superpowers/specs|plans/
```

Environment: Python 3.13 in `.venv` (created with `uv`), `python -m pytest` from the repo root,
`jupyter nbconvert --execute` to produce the committed notebook run. All randomness is seeded.

## 4. Components

Each unit has one job, a small public surface, and is testable without the notebook.

### 4.1 `src/data.py`
- `load_train() -> (list[str], np.ndarray)` and `load_test()`: read the raw CSV with the verified
  encoding, keep original row order, map `Spam→1`, `Non-Spam→0`, drop the `S. No.` column.
- Also `load_train_df()` / `load_test_df()` returning the raw DataFrame for "first 5 rows".

### 4.2 `src/features.py` — Part 2 (all hand-written in NumPy)
- `tokenize(text) -> list[str]`: lowercase; URLs → `__url__`; runs of ≥7 digits → `__phone__`;
  other numbers → `__num__`; then split on non-alphanumerics; drop 1-char tokens.
  Rationale in notebook: spam is dominated by phone numbers, prices, URLs — normalising them lets the
  model learn "contains a phone number" rather than memorising one number.
- `TfidfFeaturizer(max_features=2000, min_df=2)`: `fit(texts)` builds `vocabulary_` and `idf_`
  **from training texts only** — drop tokens with document frequency < `min_df`, then keep the
  `max_features` tokens with the highest total count (sklearn's rule, so the oracle test is exact);
  `transform(texts) -> np.ndarray[float32] (n, V)` with raw-count `tf · idf`,
  `idf = ln((1+N)/(1+df)) + 1`, rows L2-normalised (sklearn's defaults: `smooth_idf=True`,
  `sublinear_tf=False`, `norm="l2"`), tokens outside the vocabulary ignored.
- `handcrafted_features(texts) -> np.ndarray (n, 7)`: char length, word count, digit count,
  uppercase ratio, `!` count, has currency symbol (£/$/€), has URL-or-phone. These are the classic
  spam cues and are easy to explain on 2–3 examples. (Assignment's optional "additional feature
  engineering" bonus.)
- `Standardizer`: `fit`/`transform` z-scoring for the handcrafted block (fitted on train only).
- `FeaturePipeline(max_features, min_df, use_extra: bool)`: `fit(texts)`, `transform(texts)` →
  `hstack([tfidf, scaled_extra])` when `use_extra`. This is the single object the CV loop and the
  notebook use, so "the flow" is identical everywhere. `use_extra` is itself a Part 4 hyperparameter.

### 4.3 `src/ann.py` — Part 3, the algorithm we implement ourselves
```python
NeuralNetwork(hidden_layers=(32,), activation="relu",   # "relu" | "tanh" | "sigmoid"
              learning_rate=0.1, epochs=30, batch_size=32,
              l2=0.0, class_weight=None,                # None | "balanced"
              seed=0, verbose=False)
    .fit(X, y) -> self          # X: (n, d) float, y: (n,) in {0,1}; records loss_history_
    .predict_proba(X) -> (n,)   # P(spam)
    .predict(X, threshold=0.5) -> (n,) int
```
- Architecture: fully connected MLP, `d → h1 → … → 1`, sigmoid output, binary cross-entropy loss
  (+ optional L2). `hidden_layers=()` is exactly **logistic regression** → our baseline and the ANN
  bonus come from one class and appear side by side in Part 4.
- Init: He for relu, Xavier for tanh/sigmoid, zero biases. Optimiser: mini-batch SGD with
  reshuffling each epoch (`np.random.default_rng(seed)`).
- `class_weight="balanced"` multiplies the positive-class loss term by `n_neg / n_pos`
  (train is spam-minority; F1 on spam is the metric).
- Numerics: stable sigmoid via `np.logaddexp`; probabilities clipped before `log`.
- Internals kept separate and testable: `_forward(X) -> activations, pre_activations`,
  `_backward(...) -> grads`, `_loss(X, y)`.
- The notebook explains forward pass, loss, backprop and SGD in words + the update equations.

### 4.4 `src/metrics.py`
`confusion_matrix`, `precision`, `recall`, `f1_score`, `accuracy` — positive class = 1, zero-division
returns 0.0 (documented). Written by us, validated against sklearn in tests.

### 4.5 `src/cv.py` — Part 4 machinery
- `stratified_kfold_indices(y, k=5, seed=0) -> list[(train_idx, val_idx)]`.
- `cross_validate(feature_cfg, model_cfg, texts, y, k=5, seed=0) -> dict` — for **each fold**:
  fit `FeaturePipeline` on the fold's training texts only, transform both parts, fit
  `NeuralNetwork`, compute F1/precision/recall on the validation part. Returns per-fold scores,
  mean and std, and wall time. Fitting features inside the fold avoids vocabulary/IDF leakage.
- `run_grid(configs, texts, y, k=5) -> pandas.DataFrame`, one row per config, sorted by mean F1.

## 5. The notebook (`notebooks/spam_sms_ann.ipynb`)

Markdown headers use the assignment's own part names so the grader can navigate.

1. **Part 1 — Introduction**: student details (family name + first 4 ID digits, filled by the team);
   LLM usage summary with links (mirrors `docs/prompts-log.md`); the learning problem and dataset;
   load train and test **as provided**; `head()` of both; class balance bar chart; a note that the test
   set is now sealed until Part 5.
2. **Part 2 — Feature engineering**: explain tokenisation, TF-IDF and the handcrafted cues, *why*
   each suits spam detection; fit the pipeline on train; walk 3 train + 3 test messages through
   tokenizer → non-zero TF-IDF entries → handcrafted vector (a small table each).
3. **Part 3 — The algorithm**: plain-language + equations for forward pass, BCE, backprop, SGD;
   show the `NeuralNetwork` API; run the gradient-check cell live (relative error printed); train a
   first model and plot `loss_history_`.
4. **Part 4 — Training with different hyperparameters**: state the k-fold-inside-train policy;
   run `run_grid` over ~12 configs (below); results table; bar chart of mean F1 ± std; discuss;
   pick the winner and justify; show 2–3 examples passing through the winning pipeline.
5. **Part 5 — Test-set evaluation**: refit the winning config on the full train set; transform the
   test set with the train-fitted pipeline; show 2–3 test examples through the pipeline; first 5
   predictions next to true labels; F1 (headline), plus precision/recall/accuracy and the confusion
   matrix; list misclassified messages and discuss failure modes (short hams that look promotional,
   spam without digits/URLs, the 61 % spam test distribution).

Hyperparameter grid (small enough to run in a few minutes on CPU):

| knob | values |
|---|---|
| `hidden_layers` | `()` (logistic regression), `(32,)`, `(64, 32)` |
| `learning_rate` | 0.05, 0.3 |
| `activation` | relu, tanh |
| `class_weight` | None, "balanced" |
| `use_extra` (handcrafted features) | False, True |

Not a full Cartesian product: a base config plus one-factor-at-a-time variations plus a few
combinations, ≈ 12 rows. Fixed: `epochs=30`, `batch_size=32`, `max_features=2000`, `min_df=2`.

## 6. Testing (pytest, TDD per module)

- `test_data.py`: shapes, label mapping, no NaN, encoding round-trips `£`.
- `test_features.py`: tokenizer cases (URL, phone, number, punctuation); `TfidfFeaturizer` equals
  sklearn `TfidfVectorizer(tokenizer=tokenize, ...)` to 1e-6 on a toy corpus; vocabulary built from
  train only (unseen test tokens ignored); handcrafted features on crafted strings; pipeline shapes.
- `test_ann.py`: **gradient check** — analytic vs central-difference gradients on a tiny random
  problem for every activation, with and without hidden layers / L2 / class weights, relative error
  < 1e-6 in float64; can overfit 20 separable points to 100 % accuracy; `predict_proba ∈ [0,1]`;
  same seed ⇒ identical weights; loss decreases over epochs on a toy problem; `hidden_layers=()`
  matches a closed-form logistic-regression gradient.
- `test_metrics.py`: equality with sklearn on random labels, including all-zero edge cases.
- `test_cv.py`: folds are disjoint, cover everything, keep class ratios within ±1 sample; runs on a toy
  problem end to end.
- Final check: the notebook executes top-to-bottom headlessly (`nbconvert --execute`) before every
  push that touches it.

## 7. Reproducibility and anti-leakage rules

- Every RNG seeded; the notebook sets a global `SEED = 42` used everywhere.
- Feature pipeline fitted on training data only — per fold in CV, on the full train set for Part 5.
- The test set variables are created in Part 1 and next touched in Part 5; no cell in between reads them.
- Executed notebook outputs are committed; `README.md` documents the exact commands.

## 8. Out of scope (YAGNI)

Sparse matrices, Adam/momentum, early stopping, word embeddings, lemmatisation/stop-word lists,
threshold tuning on the test set, any use of sklearn outside tests.

## 9. Timeline

| by | milestone |
|---|---|
| Sep 1 | this spec, repo on GitHub, dataset in `data/raw/` |
| Sep 2 | `data.py`, `features.py`, `metrics.py` with tests |
| Sep 3 | `ann.py` with gradient check |
| Sep 4 | `cv.py`; notebook Parts 1–3 |
| Sep 6 | notebook Parts 4–5, executed, committed |
| Sep 7 | review pass, prompts log, README, registration sheet |
| Sep 8–9 | video (if chosen), incognito link test, submission; Sep 10 kept free |

## 10. Open questions for the lecturer (not blocking)

1. Is an ANN acceptable as "an algorithm from recent lectures"?
2. Is k-fold CV inside the train set acceptable for hyperparameter tuning?
3. May sklearn be used for *metrics/tests* when the learning algorithm is our own?
