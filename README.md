# Hiver 100-Minute Open Challenge

Offline-first AI email assistant pipeline for support inboxes.

## What this builds

End-to-end runnable system with four stages:
1. **Ingest** support emails from a local dataset.
2. **Triage** each email into `needs_reply`, `no_reply`, or `human_review`.
3. **Generate** draft responses only for `needs_reply`.
4. **Evaluate** per-item quality and aggregate metrics.

It also includes a thin optional adapter (`src/adapter_mcp.py`) that can reuse the same pipeline for live inbox payloads.

## Project structure

- `data/synthetic_support_emails.jsonl` — public synthetic support dataset
- `src/triage.py` — rules-first + optional LLM triage
- `src/generate.py` — reply generation (LLM or fallback heuristic)
- `src/evaluate.py` — triage + generation scoring
- `src/pipeline.py` — orchestrates full run and writes outputs
- `src/adapter_mcp.py` — optional live inbox adapter
- `outputs/` — generated on run (`predictions.jsonl`, `evaluation.json`, `report.csv`)

## Approach and tradeoffs

- **Offline reproducibility first**: primary scoring does not depend on live mailbox APIs.
- **Safety-aware triage**: risky emails (refund/legal/security/escalation) route to `human_review`.
- **Token-efficient flow**: one triage pass + one generation pass only when needed.
- **Evaluation fit for support**: combines triage classification metrics and rubric-based response quality.

## Why this metric is appropriate

This challenge needs both **decision quality** and **reply quality**:

- Triage quality: precision/recall/F1 (for `needs_reply`) and confusion matrix across all three labels.
- Reply quality: per-response rubric (relevance, completeness, correctness, tone, brevity), overall average, and pass rate.

This mirrors real support operations: first decide whether to respond automatically, then ensure generated replies are useful and safe.

## AI tools and libraries used

- Python standard library (`json`, `csv`, `argparse`, `urllib`).
- Optional OpenAI Responses API integration via environment variables:
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL` (optional, default `gpt-4.1-mini`)
  - `OPENAI_API_URL` (optional, default `https://api.openai.com/v1/responses`)

If API key is not set, pipeline still runs fully offline using deterministic heuristics.

## How to run

```bash
python3 src/pipeline.py --input data/synthetic_support_emails.jsonl --output-dir outputs
```

## Outputs

- `outputs/predictions.jsonl` — per-email triage decision, optional draft, and per-item generation score
- `outputs/evaluation.json` — aggregate metrics summary
- `outputs/report.csv` — compact report for quick inspection

## Dataset

`data/synthetic_support_emails.jsonl` is synthetic support-style data with fields:

- `email_id`
- `subject`
- `body`
- `thread_history`
- `gold_needs_reply`
- `gold_reply`
- optional `category`

