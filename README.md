# SMS Spam Classification with a From-Scratch Neural Network

Course project (Machine Learning, text-analysis track) by **Adir Buskila** and **Liav**.

We classify SMS messages as **spam / not spam** using hand-written TF-IDF features and a
**neural network implemented from scratch in NumPy** (`fit` / `predict`), tuned with k-fold
cross-validation on the training set and evaluated once on the test set that Kaggle ships.

- Dataset: <https://www.kaggle.com/datasets/datatattle/email-classification-nlp>
  (`SMS_train.csv` / `SMS_test.csv`, used exactly as provided — never re-split)
- Metric: **F1 on the positive class (Spam)**
- The graded notebook: [`notebooks/spam_sms_ann.ipynb`](notebooks/spam_sms_ann.ipynb)
- Design: [`docs/superpowers/specs/2026-09-01-sms-spam-ann-design.md`](docs/superpowers/specs/2026-09-01-sms-spam-ann-design.md)
- LLM prompts used: [`docs/prompts-log.md`](docs/prompts-log.md)

## Results

| | |
|---|---|
| Features | hand-written TF-IDF (1 142 tokens, fitted on train only) + 7 handcrafted spam cues |
| Model selection | 12 configurations, stratified 5-fold CV inside `SMS_train.csv`, one-standard-error rule |
| Winning configuration | `NeuralNetwork(hidden_layers=(32,), activation="relu", learning_rate=2.0, epochs=40, batch_size=32)` |
| Cross-validated F1 (Spam) | **0.941 ± 0.025** |
| **Test-set F1 (Spam)** — the assignment's metric | **0.945** (precision 0.986, recall 0.908, accuracy 0.936; 8 errors out of 125) |

The full walk-through, plots, error analysis and discussion are in the executed notebook
[`notebooks/spam_sms_ann.ipynb`](notebooks/spam_sms_ann.ipynb). Open tasks live in [`TODO.md`](TODO.md).
Deadline: 10 September 2026.

## Setup

Python 3.13. From the repo root:

```bash
# with uv (fast)
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# or with plain pip
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`scikit-learn` is a **test-only** dependency (an oracle for our own TF-IDF and metrics);
the learning algorithm never uses it.

## Run

```bash
python -m pytest                                   # unit tests, incl. the gradient check
python scripts/build_notebook.py                   # regenerate the notebook from scripts/build_notebook.py
python -m nbconvert --to notebook --execute --inplace notebooks/spam_sms_ann.ipynb
```

Use `python -m nbconvert` (not the `jupyter nbconvert` launcher): on a machine with several
Python installs the `jupyter` launcher can pick a `jupyter-nbconvert.exe` from another install
and then execute the notebook with that interpreter instead of the venv.

## Layout

```
src/        data loading, features, the NeuralNetwork, metrics, cross-validation
tests/      pytest suite
notebooks/  the five-part assignment notebook (executed outputs are committed)
data/raw/   the two CSV files exactly as downloaded from Kaggle
docs/       prompts log, design spec, implementation plan
```
