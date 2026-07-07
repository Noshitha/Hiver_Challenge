from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate import Evaluator
from generate import ReplyGenerator
from llm_client import LLMClient
from models import EmailRecord
from triage import TriageEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hiver challenge offline email pipeline")
    parser.add_argument("--input", default="data/synthetic_support_emails.jsonl", help="Path to input JSONL dataset")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[EmailRecord]:
    records: list[EmailRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                EmailRecord(
                    email_id=obj["email_id"],
                    subject=obj.get("subject", ""),
                    body=obj.get("body", ""),
                    thread_history=obj.get("thread_history", ""),
                    gold_needs_reply=obj["gold_needs_reply"],
                    gold_reply=obj.get("gold_reply", ""),
                    category=obj.get("category", ""),
                )
            )
    return records


def run_pipeline(records: list[EmailRecord]) -> list[dict]:
    llm = LLMClient()
    triage = TriageEngine(llm)
    generator = ReplyGenerator(llm)
    evaluator = Evaluator(llm)

    outputs: list[dict] = []
    for email in records:
        triage_result = triage.predict(email)
        generation = None
        generation_score = None

        if triage_result.label == "needs_reply":
            generation = generator.generate(email)
            generation_score = evaluator.score_generation(email.body, generation.draft_text, email.gold_reply)

        outputs.append(
            {
                "email_id": email.email_id,
                "category": email.category,
                "triage": {
                    "gold": email.gold_needs_reply,
                    "predicted": triage_result.label,
                    "confidence": round(triage_result.confidence, 4),
                    "reason": triage_result.reason,
                },
                "generation": (
                    {
                        "draft_text": generation.draft_text,
                        "confidence": round(generation.confidence, 4),
                        "flags": generation.flags,
                        "mode": generation.mode,
                    }
                    if generation
                    else None
                ),
                "generation_score": (
                    {
                        "relevance": generation_score.relevance,
                        "completeness": generation_score.completeness,
                        "correctness": generation_score.correctness,
                        "tone": generation_score.tone,
                        "brevity": generation_score.brevity,
                        "total_0_100": generation_score.total_0_100,
                        "rationale": generation_score.rationale,
                    }
                    if generation_score
                    else None
                ),
            }
        )
    return outputs


def write_outputs(items: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    report_path = output_dir / "report.csv"
    evaluation_path = output_dir / "evaluation.json"

    llm = LLMClient()
    evaluator = Evaluator(llm)
    summary = evaluator.aggregate(items)

    with predictions_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "email_id",
                "category",
                "gold_label",
                "predicted_label",
                "triage_confidence",
                "generation_mode",
                "generation_score_0_100",
            ]
        )
        for item in items:
            writer.writerow(
                [
                    item["email_id"],
                    item["category"],
                    item["triage"]["gold"],
                    item["triage"]["predicted"],
                    item["triage"]["confidence"],
                    (item["generation"] or {}).get("mode", ""),
                    (item["generation_score"] or {}).get("total_0_100", ""),
                ]
            )

    with evaluation_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    records = load_jsonl(input_path)
    items = run_pipeline(records)
    write_outputs(items, output_dir)
    print(f"Processed {len(items)} emails.")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()

