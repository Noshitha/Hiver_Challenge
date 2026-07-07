# Hiver 100-Minute Open Challenge

Build an AI email suggested-response system that:
- takes customer email threads
- decides whether a reply should be drafted
- generates a suggested reply grounded in past examples
- measures how good that reply actually is, per email and overall

This repository is intentionally evaluation-first. The generator is retrieval-grounded and runnable end to end, but the main focus is the scoring system: defining what a "good" support reply means and measuring it in a way that is stricter than exact match and less hand-wavy than pure LLM judging.

## What I built

The pipeline has four stages:

1. `triage`
   Decides whether the email is `needs_reply`, `no_reply`, or `human_review`.

2. `retrieve`
   Finds the most similar historical email/reply examples from the training set using TF-IDF over subject, body, thread context, category, and policy tags.

3. `generate`
   Produces a suggested reply using a generative model when an API key is present, or a deterministic fallback when running fully offline.

4. `evaluate`
   Scores each generated reply with a hybrid judge and writes per-response and aggregate metrics.

## Repository structure

- `scripts/build_dataset.py`
  Builds the synthetic dataset used by the benchmark.
- `data/train.jsonl`
  Retrieval bank of past support emails and replies.
- `data/test.jsonl`
  Held-out evaluation set.
- `src/triage.py`
  Rules-first triage with optional LLM classification.
- `src/retrieve.py`
  TF-IDF retrieval over historical examples.
- `src/generate.py`
  Reply generation logic.
- `src/evaluate.py`
  Production-style hybrid judge with deterministic guardrails and optional LLM cross-check.
- `src/pipeline.py`
  End-to-end runner.
- `outputs/`
  Per-email predictions and aggregate reports.

## 1. Dataset: where it came from and why it is representative

I did not use a public customer support email corpus directly because most public datasets are either:
- not actually email/reply pairs
- not shareable enough for a public repo
- too noisy for a credible benchmark inside a 100-minute challenge

Instead, I built a synthetic-but-realistic dataset by hand in [build_dataset.py](/Users/noshitha/Documents/Hiver_Challenge/scripts/build_dataset.py:1).

The dataset includes:
- 32 training examples
- 12 held-out test examples
- support categories such as billing, technical support, deliverability, plan management, security, escalation, acknowledgements, and FYI/no-reply cases
- richer fields than just raw email text:
  - `subject`
  - `incoming_email`
  - `thread_history`
  - `gold_reply`
  - `category`
  - `needs_reply`
  - `policy_tags`

Why this is representative:
- it covers the most common patterns a shared support inbox sees
- it includes both "answer directly" and "route to human review" cases
- it includes cases where no reply should be drafted at all
- it preserves the kinds of specifics support replies need to handle well:
  invoice IDs, exports, seat counts, SSO setup, bounce domains, suspicious logins, and escalation language

Why synthetic data was the right tradeoff here:
- real support email data is private
- synthetic data makes the repo public and reproducible
- a small, clean benchmark is more useful than a noisy corpus with unclear provenance

## 2. Response generation: how the suggested reply is grounded

The generation system is retrieval-augmented rather than purely free-form.

For each incoming email:
- triage decides whether it deserves a draft
- retrieval finds the top similar historical examples from `train.jsonl`
- generation uses those examples as grounding context

Why this approach:
- it is much cheaper and faster than fine-tuning
- it gives the model examples of the expected support style
- it reduces hallucination risk compared with prompting from scratch
- it keeps the system auditable because every generated draft can be traced back to retrieved examples

Why TF-IDF instead of embeddings:
- zero external dependencies
- fast enough for a small benchmark
- easy to inspect and explain
- good tradeoff for a 100-minute challenge

This is not the best possible retrieval system, but it is a sensible and runnable one.

## 3. Accuracy: what "good" means for a suggested reply

Exact-match accuracy is too strict for support email generation.

Two replies can both be good even if they use different wording. What matters is whether the reply:
- addresses the user’s actual issue
- includes the needed next step
- stays factually safe
- preserves important entities like invoice IDs or requested artifacts
- uses an appropriate support tone
- is concise without being empty

So in this project, "accuracy" means:
- **relevance**
  Did the reply answer the real ask?
- **completeness**
  Did it include the key instruction, next step, or request for missing info?
- **correctness**
  Did it avoid false promises, wrong policy, or unsafe claims?
- **tone**
  Is it professional and empathetic where needed?
- **brevity**
  Is it useful without being bloated?

## 4. Evaluation system: how replies are scored

The evaluator lives in [evaluate.py](/Users/noshitha/Documents/Hiver_Challenge/src/evaluate.py:1).

It uses a **hybrid judge**:

### A. Deterministic rubric with guardrails

This is the backbone of the scoring system.

The evaluator checks:
- keyword coverage from the customer email
- keyword coverage from the gold reply
- entity preservation, such as invoice IDs, email addresses, `PDF`, `CSV`, `SSO`, and webhook-related terms
- whether the draft includes a concrete action step
- whether it asks for details when the reference reply does
- whether it includes navigation guidance like `Settings > Billing` when expected
- whether it escalates risky cases when expected
- whether it makes risky unsupported claims like "already refunded" or "100% fixed"

This makes the judge much less likely to over-score vague replies that are merely polite.

### B. Optional LLM judge cross-check

If `OPENAI_API_KEY` is set, the evaluator also asks an LLM to score:
- relevance
- completeness
- correctness
- tone
- brevity

The LLM is a secondary signal, not the only judge.

Important design choice:
- deterministic guardrails can cap weak replies even if the LLM is overly generous

That makes the system more production-trustworthy than a pure LLM-as-judge setup.

### C. Reference similarity

The evaluator also compares the generated draft against the gold reply using:
- unigram overlap F1
- bigram recall
- keyword coverage
- entity overlap

This gives a grounded "did we say roughly the right things?" signal without requiring exact text match.

### D. Length sanity

The evaluator penalizes replies that are:
- too short to be useful
- much longer than the reference
- obviously bloated for a support email

### Final weighted score

Each generated reply gets a final score from:
- `65%` rubric score
- `25%` reference similarity
- `10%` length sanity

The system writes:
- a per-response score with rationale
- an aggregate score across the benchmark

## Why I think this metric is the right one

It avoids the main failure modes of simpler metrics:

- **Exact match**
  Too brittle for generative email replies.

- **BLEU / ROUGE only**
  Too lexical; they miss whether the reply is actually helpful and safe.

- **LLM judge only**
  Too subjective and too easy to inflate with polished but generic text.

This metric is a better compromise because:
- rubric dimensions align with how humans judge support quality
- deterministic checks make the metric harder to game
- reference similarity adds grounding
- the final score is explainable, not just a magic number

## How I validated that the metric reflects real quality

I validated the metric in two ways:

1. **Sanity-check on weak drafts**
   The first evaluator version was too generous and gave generic fallback drafts scores in the high 70s and low 80s. After strengthening the judge, the same drafts now score much lower unless they include the expected concrete action or entity details.

2. **Per-response rationale**
   Each prediction includes a rationale showing why a score was assigned. That makes it easier to inspect whether low scores correspond to real issues like missing escalation, poor specificity, or drift from the gold reply.

This does not replace human evaluation, but it does make the metric much more honest and auditable.

## Current results

After upgrading the evaluator, the current offline benchmark reports:

- triage accuracy: `1.00`
- triage F1 for `needs_reply`: `1.00`
- average hybrid generation score: `64.03`
- pass rate at threshold `70`: `14.29%`

This lower score is intentional and useful: it shows the evaluator is no longer over-rewarding generic fallback replies.

## Outputs

Running the pipeline produces:

- `outputs/predictions.jsonl`
  Per-email triage result, generated draft, retrieved example IDs, component scores, and rationale.

- `outputs/evaluation.json`
  Aggregate metrics including triage accuracy, precision/recall/F1, average generation score, pass rate, category breakdown, and confusion matrix.

- `outputs/report.csv`
  Compact table for quick review.

## How to run

Build the dataset and run the full benchmark:

```bash
python3 scripts/build_dataset.py
python3 src/pipeline.py --train data/train.jsonl --test data/test.jsonl --top-k 3
```

Optional environment variables for live LLM generation/judging:

```bash
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4.1-mini
export OPENAI_API_URL=https://api.openai.com/v1/responses
```

Then rerun:

```bash
python3 src/pipeline.py --train data/train.jsonl --test data/test.jsonl --top-k 3
```

## AI tools used

I used AI development tools during implementation and am documenting that here explicitly, as requested:

- ChatGPT / Codex-style coding assistance for architecture, refactoring, and README iteration
- optional OpenAI API for live generation and judging when an API key is present

The dataset itself is hand-authored synthetic data in code so its provenance is transparent.

## Limitations

- the dataset is small and synthetic
- retrieval is lexical TF-IDF, not semantic embeddings
- the offline fallback generator is much weaker than the evaluator
- there is no external knowledge base or policy backend
- there is no human-labeled agreement study for the metric

## If I had more time

- improve the generator so it produces concrete action steps offline
- add semantic embedding retrieval
- validate evaluator agreement against human ratings
- add policy/knowledge-base grounding
- support richer multi-turn thread state

## Summary

This repo ships:
- a public dataset
- a runnable end-to-end suggested-reply system
- a reply-needed triage step
- a grounded generation pipeline
- a per-response and overall evaluation system

The core idea is simple:
**a support reply is "accurate" when it is relevant, complete, safe, specific, and concise, not when it exactly matches a reference sentence-for-sentence.**
