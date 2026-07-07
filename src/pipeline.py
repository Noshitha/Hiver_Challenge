from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate import Evaluator
from generate import ReplyGenerator
from llm_client import LLMClient
from models import EmailRecord
from retrieve import SimpleRetriever
from triage import TriageEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hiver challenge offline email pipeline with RAG")
    parser.add_argument("--train", default="data/train.jsonl", help="Training data for retrieval")
    parser.add_argument("--test", default="data/test.jsonl", help="Test data for evaluation")
    parser.add_argument("--output-dir", default="outputs", help="Output directory")
    parser.add_argument("--top-k", type=int, default=3, help="Number of examples to retrieve")
    return parser.parse_args()


def load_test_jsonl(path: Path) -> list[EmailRecord]:
    """Load test examples into EmailRecord format."""
    records: list[EmailRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                EmailRecord(
                    email_id=obj.get("id", obj.get("email_id", "")),
                    subject="",  # test data uses incoming_email only
                    body=obj["incoming_email"],
                    thread_history="",
                    gold_needs_reply=obj["needs_reply"],
                    gold_reply=obj.get("gold_reply", ""),
                    category=obj.get("category", ""),
                )
            )
    return records


def run_pipeline(records: list[EmailRecord], retriever: SimpleRetriever, top_k: int) -> list[dict]:
    """Run end-to-end pipeline with retrieval-augmented generation."""
    llm = LLMClient()
    triage = TriageEngine(llm)
    generator = ReplyGenerator(llm)
    evaluator = Evaluator(llm)

    outputs: list[dict] = []
    for email in records:
        triage_result = triage.predict(email)
        generation = None
        generation_score = None
        retrieved_examples = None

        if triage_result.label == "needs_reply":
            retrieved_examples = retriever.retrieve(email.body, top_k=top_k)
            generation = generator.generate(email, retrieved_examples=retrieved_examples)
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
                        "retrieved_example_ids": generation.retrieved_example_ids,
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
                        "rubric_score_0_100": generation_score.rubric_score_0_100,
                        "reference_similarity_0_100": generation_score.reference_similarity_0_100,
                        "length_penalty_0_100": generation_score.length_penalty_0_100,
                        "hybrid_score_0_100": generation_score.hybrid_score_0_100,
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
                "rubric_score",
                "similarity_score",
                "hybrid_score",
            ]
        )
        for item in items:
            g_score = item.get("generation_score") or {}
            writer.writerow(
                [
                    item["email_id"],
                    item["category"],
                    item["triage"]["gold"],
                    item["triage"]["predicted"],
                    item["triage"]["confidence"],
                    (item.get("generation") or {}).get("mode", ""),
                    g_score.get("rubric_score_0_100", ""),
                    g_score.get("reference_similarity_0_100", ""),
                    g_score.get("hybrid_score_0_100", ""),
                ]
            )

    with evaluation_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)


def main() -> None:
    args = parse_args()
    train_path = Path(args.train)
    test_path = Path(args.test)
    output_dir = Path(args.output_dir)
    
    retriever = SimpleRetriever(train_path)
    records = load_test_jsonl(test_path)
    items = run_pipeline(records, retriever, args.top_k)
    write_outputs(items, output_dir)
    
    print(f"Processed {len(items)} test emails with RAG (top-k={args.top_k}).")
    print(f"Wrote outputs to: {output_dir}")


if __name__ == "__main__":
    main()

