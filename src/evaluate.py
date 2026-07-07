from __future__ import annotations

import math
from collections import Counter
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
    rubric_score_0_100: float
    reference_similarity_0_100: float
    length_penalty_0_100: float
    hybrid_score_0_100: float
    rationale: str


class Evaluator:
    def __init__(self, llm_client: LLMClient, pass_threshold: float = 70.0) -> None:
        self.llm_client = llm_client
        self.pass_threshold = pass_threshold
        self.rubric_weight = 0.7
        self.similarity_weight = 0.2
        self.length_weight = 0.1

    def score_generation(self, customer_email: str, draft_reply: str, gold_reply: str) -> GenerationScore:
        """
        Hybrid evaluation with three components:
        1. Rubric-based LLM judge (or heuristic fallback)
        2. Reference similarity (token overlap + simple semantic)
        3. Length sanity check
        """
        if self.llm_client.enabled:
            rubric = self._llm_judge_rubric(customer_email, draft_reply, gold_reply)
        else:
            rubric = self._heuristic_rubric(customer_email, draft_reply, gold_reply)
        
        similarity_score = self._reference_similarity(draft_reply, gold_reply)
        length_score = self._length_sanity(draft_reply, gold_reply)
        
        hybrid = (
            self.rubric_weight * rubric["score_0_100"] +
            self.similarity_weight * similarity_score +
            self.length_weight * length_score
        )
        
        return GenerationScore(
            relevance=rubric["relevance"],
            completeness=rubric["completeness"],
            correctness=rubric["correctness"],
            tone=rubric["tone"],
            brevity=rubric["brevity"],
            rubric_score_0_100=rubric["score_0_100"],
            reference_similarity_0_100=round(similarity_score, 2),
            length_penalty_0_100=round(length_score, 2),
            hybrid_score_0_100=round(hybrid, 2),
            rationale=rubric["rationale"]
        )

    def aggregate(self, items: list[dict]) -> dict:
        total = len(items)
        correct = sum(1 for item in items if item["triage"]["predicted"] == item["triage"]["gold"])
        triage_accuracy = (correct / total) if total else 0.0

        confusion = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}
        tp = fp = fn = 0
        generation_scores: list[float] = []
        category_scores: dict[str, list[float]] = {}

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
            if g and isinstance(g.get("hybrid_score_0_100"), (int, float)):
                score = float(g["hybrid_score_0_100"])
                generation_scores.append(score)
                
                category = item.get("category", "unknown")
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(score)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        avg_score = sum(generation_scores) / len(generation_scores) if generation_scores else 0.0
        pass_rate = (
            sum(1 for score in generation_scores if score >= self.pass_threshold) / len(generation_scores)
            if generation_scores
            else 0.0
        )

        category_averages = {
            cat: round(sum(scores) / len(scores), 2)
            for cat, scores in category_scores.items()
        }

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
                "average_hybrid_score_0_100": round(avg_score, 2),
                "pass_threshold": self.pass_threshold,
                "pass_rate": round(pass_rate, 4),
                "by_category": category_averages,
            },
            "overall": {
                "total_emails": total,
                "llm_judge_enabled": self.llm_client.enabled,
            },
        }

    def _llm_judge_rubric(self, customer_email: str, draft_reply: str, gold_reply: str) -> dict:
        """LLM-based rubric scoring."""
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
            return self._heuristic_rubric(customer_email, draft_reply, gold_reply)
        
        try:
            relevance = int(scored.get("relevance", 3))
            completeness = int(scored.get("completeness", 3))
            correctness = int(scored.get("correctness", 3))
            tone = int(scored.get("tone", 3))
            brevity = int(scored.get("brevity", 3))
        except (TypeError, ValueError):
            return self._heuristic_rubric(customer_email, draft_reply, gold_reply)
        
        dims = [max(1, min(5, x)) for x in [relevance, completeness, correctness, tone, brevity]]
        score_0_100 = (sum(dims) / 25.0) * 100.0
        
        return {
            "relevance": dims[0],
            "completeness": dims[1],
            "correctness": dims[2],
            "tone": dims[3],
            "brevity": dims[4],
            "score_0_100": round(score_0_100, 2),
            "rationale": str(scored.get("rationale", "LLM rubric evaluation")),
        }

    def _heuristic_rubric(self, customer_email: str, draft_reply: str, gold_reply: str) -> dict:
        """Fallback heuristic rubric scoring."""
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

        score_0_100 = ((relevance + completeness + correctness + tone + brevity) / 25.0) * 100.0
        
        return {
            "relevance": relevance,
            "completeness": completeness,
            "correctness": correctness,
            "tone": tone,
            "brevity": brevity,
            "score_0_100": round(score_0_100, 2),
            "rationale": "Heuristic rubric evaluation (offline mode).",
        }

    def _reference_similarity(self, draft_reply: str, gold_reply: str) -> float:
        """Token overlap similarity with gold reference."""
        if not gold_reply:
            return 75.0  # neutral score when no reference available
        
        draft_tokens = set(draft_reply.lower().split())
        gold_tokens = set(gold_reply.lower().split())
        
        if not draft_tokens or not gold_tokens:
            return 50.0
        
        overlap = len(draft_tokens & gold_tokens)
        union = len(draft_tokens | gold_tokens)
        jaccard = overlap / union if union > 0 else 0.0
        
        # Scale to 0-100 range
        return min(100.0, jaccard * 150.0)  # boost to make it meaningful

    def _length_sanity(self, draft_reply: str, gold_reply: str) -> float:
        """Penalize drafts that are way too short or verbose."""
        draft_words = len(draft_reply.split())
        
        if draft_words < 10:
            return 40.0  # too terse
        elif draft_words > 200:
            return 60.0  # too verbose
        elif 20 <= draft_words <= 100:
            return 100.0  # ideal range
        else:
            return 85.0  # acceptable


