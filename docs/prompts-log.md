# LLM / chatbot prompts log

The assignment requires listing the prompts we gave to AI assistants, links to the conversations,
and what we used them for. This file is kept **as we go** — every session gets an entry the same day.

> Before submission: make sure every link below opens for someone who is not logged in as us
> (export or share the transcript if the platform requires it), and copy this table into
> Part 1 of the notebook.

| # | Date | Tool | Prompt(s) (verbatim or lightly shortened) | Used for | Link |
|---|---|---|---|---|---|
| 0 | before 2026-09-01 | _(fill in)_ | Drafting the initial `TODO.md` task board from the assignment PDF | Planning | _(add link)_ |
| 1 | 2026-09-01 | Claude Code (Claude Fable 5) | "alright claude as you can see in @TODO.md, we made the decisions and downloaded the dataset, lets continue" · "okay about the split to train/test, we will handle that, no? no need to search for an already split dataset" · "okay im now seeing this, so we need to change the dataset, how can i pick one thats already split" · "give me the url to the dataset" | Re-reading the PDF's train/test rule; discovering the first dataset shipped no split; finding a Kaggle SMS-spam dataset with a given split (`datatattle/email-classification-nlp`); writing the design spec (`docs/superpowers/specs/2026-09-01-sms-spam-ann-design.md`); creating the repo, `.gitignore`, `requirements.txt`, venv | https://claude.ai/code/session_019XxZC9ESBsXSmwJUA5PudP |

## How we use the assistant (summary for the notebook)

- Planning and project structure (task board, design spec, implementation plan).
- Writing and reviewing code **that we then read, test and explain ourselves** — the learning
  algorithm in `src/ann.py` is our own NumPy implementation and is unit-tested (gradient check).
- Drafting explanatory Markdown cells, which we edit.
