# Hiver 100-Minute Open Challenge

**Retrieval-Augmented Generation (RAG) email reply system with hybrid evaluation.**

## What this builds

A fully offline, end-to-end pipeline that:

1. **Triages** incoming support emails into `needs_reply`, `no_reply`, or `human_review`
2. **Retrieves** similar historical examples from a training set
3. **Generates** grounded draft replies using retrieved context
4. **Evaluates** each reply with hybrid scoring (LLM rubric + reference similarity + length sanity)
5. **Reports** per-email predictions and aggregate metrics

**Key innovation:** Hybrid evaluation combines subjective quality judgment (rubric) with objective reference checks (similarity, length) for credible, explainable scoring.

## Project structure

- `data/train.jsonl` — 20 historical email/reply pairs for retrieval
- `data/test.jsonl` — 10 held-out test cases with gold labels
- `src/triage.py` — rules-first + optional LLM triage
- `src/retrieve.py` — TF-IDF-based similarity retrieval (no external deps)
- `src/generate.py` — RAG prompt-based reply generation
- `src/evaluate.py` — hybrid scoring: rubric (70%) + similarity (20%) + length (10%)
- `src/pipeline.py` — end-to-end CLI runner
- `outputs/` — `predictions.jsonl`, `evaluation.json`, `report.csv`

## Approach and tradeoffs

**Why retrieval-augmented generation?**
- Grounds replies in real historical examples rather than pure LLM creativity
- Reduces hallucination risk for support responses
- Simpler and faster than fine-tuning for 100-minute constraint
- Easy to audit: every generated reply cites retrieved examples

**Why TF-IDF retrieval instead of embeddings?**
- Zero external dependencies (runs with Python stdlib)
- Fast indexing and query (<1ms per retrieval)
- Good enough for small training sets (20 examples)
- Transparent: you can inspect similarity scores

**Triage design:**
- Rules catch obvious cases (newsletters, auto-replies)
- LLM classifier handles borderline cases
- `human_review` route for risky scenarios (refunds, security, legal threats)
- Avoids wasting generation and evaluation budget on no-action emails

## Why this metric is appropriate

**The challenge asks:** *"How do we know this reply is good?"*

**Single-score approaches fail:**
- Pure LLM-as-judge is subjective and opaque
- Exact match is too strict for generative tasks
- BLEU/ROUGE miss semantic quality

**Our hybrid metric combines three signals:**

1. **Rubric-based LLM judge (70% weight):** Scores relevance, completeness, correctness, tone, brevity on 1-5 scale. Provides structured rationale.
   
2. **Reference similarity (20% weight):** Token overlap with gold reply. Catches when generated text diverges wildly from correct answer.

3. **Length sanity check (10% weight):** Penalizes replies that are way too short (<10 words) or too verbose (>200 words).

**Why this works for support email:**
- Rubric captures subjective quality (empathy, professionalism)
- Similarity ensures factual grounding
- Length check prevents trivial or bloated responses
- Weighted combination is explainable: you can see all three components in output

**Validation:**
- Tested on 10 diverse support scenarios
- Average hybrid score: 78.28/100
- 100% pass rate above 70-point threshold
- Category breakdown shows consistent performance across billing, technical, escalation cases

## AI tools and libraries used

- **Python 3** standard library only (no pip installs required for base run)
- **Optional OpenAI API** for LLM judge and generation:
  - Set `OPENAI_API_KEY` environment variable
  - Set `OPENAI_MODEL` (default: `gpt-4.1-mini`)
  - Set `OPENAI_API_URL` (default: `https://api.openai.com/v1/responses`)
- **Fallback heuristics:** Pipeline runs fully offline if API key not set

**Development tools:**
- GitHub Copilot CLI for code scaffolding and refactoring
- Manual dataset curation for realistic support scenarios

## How to run

```bash
python3 src/pipeline.py --train data/train.jsonl --test data/test.jsonl --top-k 3
```

**Optional flags:**
- `--top-k N` — retrieve N similar examples (default: 3)
- `--output-dir DIR` — output directory (default: `outputs/`)

**Runtime:** ~5-10 seconds for 10 test emails (offline mode)

## Outputs

- **`outputs/predictions.jsonl`** — per-email results with:
  - Triage decision and confidence
  - Generated draft text
  - Retrieved example IDs
  - Rubric scores (5 dimensions)
  - Hybrid score components
  - Judge rationale

- **`outputs/evaluation.json`** — aggregate metrics:
  - Triage accuracy, precision, recall, F1
  - Average hybrid score
  - Pass rate above threshold
  - Scores by category
  - Confusion matrix

- **`outputs/report.csv`** — compact summary for quick inspection

## Dataset

**Training set (`data/train.jsonl`):** 20 synthetic but realistic support email/reply pairs covering:
- Billing (invoices, charges, refunds)
- Technical (password resets, integrations)
- Account management (upgrades, downgrades, user additions)
- How-to questions (exports, templates, workflows)
- Security incidents
- Escalations

**Test set (`data/test.jsonl`):** 10 held-out examples with same categories, intentionally varied phrasing to test retrieval robustness.

**Dataset schema:**
```json
{
  "id": "train_001",
  "incoming_email": "Customer's question or request",
  "gold_reply": "Reference human-written reply",
  "category": "billing",
  "needs_reply": "needs_reply"
}
```

**Why synthetic?**
- Real customer support data is private/sensitive
- Synthetic data allows transparent sharing and reproducible evaluation
- Covers representative scenarios across common support categories
- Documented in submission (not hidden or proprietary)

## Limitations and future work

**Current limitations:**
- Small training set (20 examples) — real deployment would need 100s-1000s
- TF-IDF retrieval is basic — embeddings would improve semantic matching
- LLM judge can be subjective — real validation needs human ratings
- No multi-turn conversation handling
- No policy/KB integration beyond retrieved examples

**Future improvements:**
- Add semantic embeddings for retrieval (e.g., sentence-transformers)
- Expand training set with real anonymized examples
- Fine-tune small model on support data
- Add policy/knowledge base grounding
- Implement confidence-based escalation thresholds
- A/B test generated replies against human baseline

## Submission checklist

✅ Public GitHub repository  
✅ Dataset (train/test split with documented generation)  
✅ Runnable end-to-end Gen-AI response generator  
✅ Per-response and overall evaluation system  
✅ README with approach, metric rationale, and run instructions  
✅ All code is original work completed during challenge window  

**Repository:** [https://github.com/Noshitha/Hiver_Challenge](https://github.com/Noshitha/Hiver_Challenge)

