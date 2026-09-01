# LLM / chatbot prompts log

The assignment requires listing the prompts we gave to AI assistants, links to the conversations,
and what we used them for. This file is kept **as we go** — every session gets an entry the same day.

> Before submission: make sure every link below opens for someone who is not logged in as us
> (export or share the transcript if the platform requires it), and copy this table into
> Part 1 of the notebook.

| # | Date | Tool | Prompt(s) (verbatim or lightly shortened) | Used for | Link |
|---|---|---|---|---|---|
| 0 | before 2026-09-01 | _(fill in)_ | Drafting the initial `TODO.md` task board from the assignment PDF | Planning | _(add link)_ |
| 1 | 2026-09-01 | Claude Code (Claude Fable 5) | "alright claude as you can see in @TODO.md, we made the decisions and downloaded the dataset, lets continue" · "okay about the split to train/test, we will handle that, no? no need to search for an already split dataset" · "okay im now seeing this, so we need to change the dataset, how can i pick one thats already split" · "give me the url to the dataset" · "approved, write the plan and start" · "I'm counting on you to proceed with the project, implement everything correctly, test it, see that the model that we trained gets the spam correctly and everything is within the requirements of the PDF" · "merge it to main" · (student names/IDs for Part 1) · "about the video we need to make, I want you to create a transcript for me and Liav… tell us what to say, on which file to stand, and which lines of code to show… in Hebrew" | Re-reading the PDF's train/test rule; discovering the first dataset shipped no split; finding a Kaggle SMS-spam dataset with a given split (`datatattle/email-classification-nlp`); writing the design spec and the implementation plan (`docs/superpowers/`); creating the repo, `.gitignore`, `requirements.txt`, venv; implementing `src/` (data loading, tokenizer + TF-IDF, handcrafted features, the `NeuralNetwork`, metrics, k-fold CV) test-first with pytest (104 tests incl. gradient check); generating and executing the 5-part notebook; writing the results discussion from the actual outputs; merging to `main`; drafting the Hebrew presentation script for the video (kept outside the repo) | https://claude.ai/code/session_019XxZC9ESBsXSmwJUA5PudP |

## How we use the assistant (summary for the notebook)

- Planning and project structure (task board, design spec, implementation plan).
- Writing and reviewing code **that we then read, test and explain ourselves** — the learning
  algorithm in `src/ann.py` is our own NumPy implementation and is unit-tested (gradient check).
- Drafting explanatory Markdown cells, which we edit.
