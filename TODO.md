# ML Project — Task Board

**Team:** Adir + Liav
**Course:** Machine Learning — image processing / text analysis assignment
**Source of truth:** `assignment_ml_w_image_or_text_instructions-wo_points_and_bonuses.pdf`
**Last updated:** 2026-09-01
**Repo (source of truth):** `C:\Users\Adir\dev\ml-sms-spam-ann` -> https://github.com/AdirBuskila/ml-sms-spam-ann
**🔥 DEADLINE: 10 September 2026**

> How to use this file: tick boxes as you go, put your name in the **Owner** slot when you take something,
> and resolve everything in section 1 before writing real code — the rest depends on it.

---

## 1. Decisions we need to make FIRST 🔴

Nothing below section 2 can start until these are settled. Write the answer directly under each one.

### D1 — Text (NLP) or Images (Computer Vision)?

Drives the dataset, the feature engineering, and how heavy the compute gets.

| | Text / NLP | Images / CV |
|---|---|---|
| Example task | spam / sentiment / topic | digit or image classification |
| Features | TF-IDF, bag-of-words, n-grams | raw pixels, HOG, edges, PCA |
| Compute | light, runs anywhere | heavier, may want Colab GPU |
| Risk | low | medium |

- **Recommendation:** Text — a hand-written ANN over TF-IDF vectors is far easier to debug than one over image data, and the assignment rewards a *working* model over an ambitious broken one.
- **DECISION:** Text
- [x] Decided

### D2 — Which Kaggle dataset?

Hard requirements from the PDF:

- Must come from Kaggle, and the assignment says pick something **relatively simple**
- **Must already ship a separate train and test split** — we are explicitly forbidden from splitting it ourselves
- Needs a public URL for the registration sheet

Starting points given in the PDF:

- NLP: <https://www.kaggle.com/datasets?tags=13204-NLP> · example: [spam text message classification](https://www.kaggle.com/datasets/team-ai/spam-text-message-classification)
- CV: <https://www.kaggle.com/datasets?tags=13207-Computer+Vision> · example: [MNIST](https://www.kaggle.com/datasets/hoijatk/mnist-dataset)

- ~~First pick: `mariumfaheem666/spam-sms-classification-using-nlp`~~ - **dropped on 2026-09-01**: it is a single
  `Spam_SMS.csv` with no train/test split (the PDF's own NLP example is unsplit too, and no SMS-spam dataset on
  Kaggle ships one - we searched). Re-splitting it ourselves would risk trap T1.
- **DECISION:** <https://www.kaggle.com/datasets/datatattle/email-classification-nlp> - "E-Mail classification NLP"
  (content is SMS spam). Ships `SMS_train.csv` (~950 rows) + `SMS_test.csv` (125 rows), labels `Spam` / `Non-Spam`.
  Caveats: small test set; test is ~61% spam; some `£` bytes are mangled - handle at load time.
- [x] Dataset chosen
- [x] Verified it has a ready-made train/test split (2 files in Kaggle's Data Explorer)
- [ ] Both of us can download it (Adir: done 2026-09-01; Liav: pending)

### D3 — Learning type: classification / regression / clustering?

Follows from D2, but it determines our quality metric, so write it down explicitly.

| Task type | Required metric (from the PDF) |
|---|---|
| Regression | R² |
| Multi-class classification | macro-average F1 |
| Binary classification | F1 on the positive class only |

- **DECISION:** Binary classification
- **Therefore our metric is:** F1
- [x] Decided

### D4 — Which algorithm do we hand-implement?

The PDF is strict here: it must be an algorithm from the **recent** lectures. KNN on its own is explicitly called out as not enough. ANN is named as bonus-worthy.

- **Leaning:** Neural network (ANN) from scratch in NumPy — it is the named bonus, and its knobs (hidden layers, learning rate, epochs, batch size, activation) map perfectly onto the Part 4 "run the flow with different hyperparameters" requirement.
- [ ] Confirm with the lecturer / syllabus that the ANN counts as "recently covered"
- **DECISION:** ANN (multi-layer perceptron) from scratch in NumPy - `NeuralNetwork(hidden_layers, activation,
  learning_rate, epochs, batch_size, l2, class_weight)` with `fit` / `predict`. `hidden_layers=()` is plain logistic
  regression, so the baseline and the ANN bonus come from the same class and are compared side by side in Part 4.
- [x] Decided (2026-09-01)

### D5 — Presentation format: recorded video or live in class?

- Video: ≤ 5 minutes, uploaded to YouTube or anywhere playable **without requiring a download**, link viewable by anyone who has it
- Live: presented in class instead — no video needed, but no second chances
- **DECISION:** _______________ (deferred on 2026-09-01 - does not block the code)
- [ ] Decided

### D6 — Third team member? ✅ RESOLVED

- **DECISION: No. Team is Adir + Liav only (2 people).**
- [x] Decided

Consequence: both of us present in the video, and only 2 IDs go in the submission line.

### D7 — What is the actual submission deadline? ✅ RESOLVED

- **DEADLINE: 10 September 2026** (late = penalty of half a symbolic point per day)
- [x] Confirmed

Working back from that, roughly: dataset + setup locked by **Aug 30**, ANN working and gradient-checked by
**Sep 4**, experiments + test evaluation by **Sep 7**, video + submission **Sep 8–9**. Leave the last day free —
something always breaks.

---

## 2. Project setup 🛠️

| # | Task | Owner | Done |
|---|---|---|---|
| 2.1 | Create the repo skeleton (`src/`, `notebooks/`, `tests/`, `data/`, `docs/`) | Adir | [x] |
| 2.2 | `.gitignore` — covers `desktop.ini`, `.venv/`, `__pycache__`, `.ipynb_checkpoints`, the assignment PDF (`data/` is **committed** on purpose: ~105 kB, lets graders run the notebook) | Adir | [x] |
| 2.3 | `requirements.txt` — numpy, pandas, matplotlib, jupyter, pytest (+ sklearn for tests only) | Adir | [x] |
| 2.4 | Create a virtualenv **outside** this Google Drive folder — `.venv` inside the repo at `C:\Users\Adir\dev\ml-sms-spam-ann` (uv, Python 3.13) | Adir | [x] |
| 2.5 | Create the GitHub repo and push | | [ ] |
| 2.6 | Make the repo **public** (or confirm graders can access it) | | [ ] |
| 2.7 | Download `SMS_train.csv` + `SMS_test.csv` into `data/raw/` and document the source in `data/README.md` | Adir | [x] |
| 2.8 | Start `docs/prompts-log.md` and log LLM prompts **as we go** (see trap T4) | Adir | [x] |

⚠️ **Google Drive + git warning:** if we both have this folder syncing while either of us runs git, the `.git`
directory can corrupt. Once GitHub is up, treat GitHub as the source of truth and work from a clone that
lives **outside** Drive.

---

## 3. The notebook — 5 mandatory parts 📓

The graded deliverable is one notebook following exactly this structure.

### Part 1 — Introduction

- [ ] Student details: **family name + first 4 digits of ID** for each of us
- [ ] The LLM / chatbot prompts we used, with links to the conversations, and what we used them for
- [ ] Explanation of the learning problem and the dataset
- [ ] Load the train set and the test set — **as provided, no re-splitting**
- [ ] Display the first 5 rows

### Part 2 — Feature engineering

- [ ] Implement the feature extraction for our chosen track (text or image)
- [ ] Explain *why* each feature makes sense for this problem
- [ ] Demonstrate the transformation on **2–3 concrete examples from train AND from test**
- [ ] *(optional bonus)* additional feature engineering

### Part 3 — Implement the learning algorithm

- [ ] Write the algorithm **ourselves** — not a library call
- [ ] Expose a `fit(X, y)` function and a `predict(X)` function
- [ ] Make the hyperparameters configurable
- [ ] Explain how the algorithm works, in the notebook and in the presentation
- [ ] Unit-test it — for an ANN a **gradient check** is essential, because hand-written backprop fails silently

### Part 4 — Training with different hyperparameters

- [ ] Run the full flow across several hyperparameter combinations
- [ ] Compare them using our quality metric from D3
- [ ] Show the pipeline on **2–3 examples** as they pass through feature engineering
- [ ] Pick and justify the winning combination

⚠️ Tuning needs validation data, but we may not re-split the given train/test. Use **k-fold
cross-validation inside the train set only** and say so explicitly in the notebook — the test set stays
sealed until Part 5.

### Part 5 — Prediction and evaluation on the test set

- [ ] Apply the winning combination to the whole test set
- [ ] Show the pipeline on **2–3 test examples**
- [ ] Show the **first 5 predictions**
- [ ] Report the quality metric from D3
- [ ] Brief discussion: where does the model fail, and why?

---

## 4. Submission checklist 📤

### Registration sheet

One of us registers the group; **all fields must be filled**.

- [ ] Assignment type (text analysis / image processing)
- [ ] Learning type (classification / regression / clustering)
- [ ] Implemented learning algorithms
- [ ] Dataset name
- [ ] Dataset URL (Kaggle)
- [ ] Video URL
- [ ] Repository URL
- [ ] Contact emails — **only if they differ from the email used to submit the form**
- [ ] Student name 1..5 (Hebrew, one per column)
- [ ] Email student 1..5 (one per column)

### The four links, submitted space-separated with no extra text

```
<video-url> <repo-url> <kaggle-dataset-url> <id1> <id2> <id3>
```

- [ ] Video link — playable in-browser, **no download required**
- [ ] Repository link — contains the code **and the output of every run**
- [ ] Kaggle dataset link
- [ ] All ID numbers
- [ ] **Open every link in a private / incognito window and confirm it works** — the PDF demands this twice
- [ ] Submitted once only (one submission per group)

### Video (if we chose video in D5)

- [ ] Under 5 minutes
- [ ] Says our names at the start
- [ ] Covers every part of the assignment **and our outputs** — show the code and the results, don't just narrate
- [ ] Everyone in the group presents a roughly equal share
- [ ] Uploaded, link tested from a logged-out browser

---

## 5. Traps that cost points 🚨

| | Trap | Why it hurts |
|---|---|---|
| T1 | Re-splitting the dataset into train/test ourselves | Explicitly forbidden (PDF p.4, twice). Use the two files Kaggle ships; the test set is opened once, in Part 5. Tuning = k-fold CV inside train. |
| T2 | Using `sklearn` for the learning algorithm | Part 3 requires **our own** implementation with fit/predict. |
| T3 | Implementing only KNN | Called out in the PDF as not advanced enough. |
| T4 | Reconstructing the LLM prompt log at the end | Impossible after the fact. Log prompts **as we go**. |
| T5 | Wrong quality metric | Binary → F1 on the positive class. Multi-class → macro-F1. Regression → R². |
| T6 | A repo with code but no run output | The PDF requires outputs to be committed. Run all cells before the final push. |
| T7 | A dead or permission-locked link | No video / no working code link = significant penalty. Test incognito. |
| T8 | Video over 5 minutes, or missing | Penalty. |
| T9 | Notebook missing or not working | "A task without a notebook will not be checked." |

---

## 6. Division of work

| Area | Owner | Notes |
|---|---|---|
| Dataset research & selection | | |
| Feature engineering (Part 2) | | |
| ANN implementation (Part 3) | | |
| Hyperparameter experiments (Part 4) | | |
| Evaluation & write-up (Part 5) | | |
| Video editing | | |
| Sheet registration & final submission | | |

Both of us must present, so whoever writes a part should be the one to explain it on camera.

---

## 7. Open questions for the lecturer ❓

- [ ] Does the ANN count as "an algorithm from the recent lectures"?
- [ ] Is k-fold CV inside the train set acceptable for hyperparameter tuning?
- [ ] Are we allowed to use `sklearn` for *metrics* if we implement the learning algorithm ourselves?
