from __future__ import annotations

from dataclasses import dataclass

from llm_client import LLMClient


LABELS = ["needs_reply", "no_reply", "human_review"]


@dataclass
class GenerationScore:
    relevance: int
    completeness: int
    correctness: int
    tone: int
    brevity: int
    total_0_100: float
    rationale: str


class Evaluator:
    def __init__(self, llm_client: LLMClient, pass_threshold: float = 70.0) -> None:
        self.llm_client = llm_client
        self.pass_threshold = pass_threshold

    def score_generation(self, customer_email: str, draft_reply: str, gold_reply: str) -> GenerationScore:
        if self.llm_client.enabled:
            llm_scored = self._llm_judge(customer_email, draft_reply, gold_reply)
            if llm_scored:
                return llm_scored
        return self._heuristic_judge(customer_email, draft_reply, gold_reply)

    def aggregate(self, items: list[dict]) -> dict:
        total = len(items)
        correct = sum(1 for item in items if item["triage"]["predicted"] == item["triage"]["gold"])
        triage_accuracy = (correct / total) if total else 0.0

        confusion = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
        tp = fp = fn = 0
        generation_scores: list[float] = []

        for item in items:
            gold = item["triage"]["gold"]
            pred = item["triage"]["predicted"]
            confusion[gold][pred] += 1

            if pred == "needs_reply" and gold == "needs_reply":
                tp += 1
            elif pred == "needs_reply" and gold != "needs_reply":
                fp += 1
            elif pred != "needs_reply" and gold == "needs_reply":
                fn += 1

            g = item.get("generation_score")
            if g and isinstance(g.get("total_0_100"), (int, float)):
                generation_scores.append(float(g["total_0_100"]))

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        avg_score = sum(generation_scores) / len(generation_scores) if generation_scores else 0.0
        pass_rate = (
            sum(1 for score in generation_scores if score >= self.pass_threshold) / len(generation_scores)
            if generation_scores
            else 0.0
        )

        return {
            "triage": {
                "accuracy": round(triage_accuracy, 4),
                "precision_needs_reply": round(precision, 4),
                "recall_needs_reply": round(recall, 4),
                "f1_needs_reply": round(f1, 4),
                "confusion_matrix": confusion,
            },
            "generation": {
                "count_scored": len(generation_scores),
                "average_score_0_100": round(avg_score, 2),
                "pass_threshold": self.pass_threshold,
                "pass_rate": round(pass_rate, 4),
            },
            "overall": {
                "total_emails": total,
                "llm_judge_enabled": self.llm_client.enabled,
            },
        }

    def _llm_judge(self, customer_email: str, draft_reply: str, gold_reply: str) -> GenerationScore | None:
        system_prompt = (
            "You are a strict support-reply judge. "
            "Return JSON with integer fields relevance, completeness, correctness, tone, brevity in range 1-5, "
            "plus a short rationale."
        )
        user_prompt = (
            f"Customer email:\n{customer_email}\n\n"
            f"Draft reply:\n{draft_reply}\n\n"
            f"Reference reply:\n{gold_reply}\n\n"
            "Scoring rubric:\n"
            "- relevance: addresses user's ask\n"
            "- completeness: covers needed steps/info\n"
            "- correctness: avoids false promises/facts\n"
            "- tone: empathetic/professional\n"
            "- brevity: concise and readable\n"
        )
        scored = self.llm_client.complete_json(system_prompt, user_prompt, temperature=0.0)
        if not isinstance(scored, dict):
            return None
        try:
            relevance = int(scored.get("relevance", 3))
            completeness = int(scored.get("completeness", 3))
            correctness = int(scored.get("correctness", 3))
            tone = int(scored.get("tone", 3))
            brevity = int(scored.get("brevity", 3))
        except (TypeError, ValueError):
            return None
        dims = [max(1, min(5, x)) for x in [relevance, completeness, correctness, tone, brevity]]
        total_0_100 = (sum(dims) / 25.0) * 100.0
        return GenerationScore(
            relevance=dims[0],
            completeness=dims[1],
            correctness=dims[2],
            tone=dims[3],
            brevity=dims[4],
            total_0_100=round(total_0_100, 2),
            rationale=str(scored.get("rationale", "LLM rubric evaluation")),
        )

    def _heuristic_judge(self, customer_email: str, draft_reply: str, gold_reply: str) -> GenerationScore:
        c_lower = customer_email.lower()
        d_lower = draft_reply.lower()

        relevance = 4 if ("?" in customer_email and "help" in d_lower) or any(k in d_lower for k in ["please", "can", "we"]) else 3
        completeness = 4 if any(k in d_lower for k in ["next step", "please share", "confirm"]) else 3
        correctness = 4 if not any(k in d_lower for k in ["guarantee", "immediately refunded"]) else 2
        tone = 5 if any(k in d_lower for k in ["thanks", "sorry", "happy to help"]) else 3
        brevity = 5 if len(draft_reply.split()) <= 120 else 3

        if "refund" in c_lower and "specialist" in d_lower:
            correctness = min(5, correctness + 1)
        if gold_reply and len(set(d_lower.split()) & set(gold_reply.lower().split())) > 6:
            relevance = min(5, relevance + 1)

        total_0_100 = ((relevance + completeness + correctness + tone + brevity) / 25.0) * 100.0
        return GenerationScore(
            relevance=relevance,
            completeness=completeness,
            correctness=correctness,
            tone=tone,
            brevity=brevity,
            total_0_100=round(total_0_100, 2),
            rationale="Heuristic rubric evaluation (offline mode).",
        )

