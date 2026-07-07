from __future__ import annotations

import re
from dataclasses import dataclass

from llm_client import LLMClient


LABELS = ["needs_reply", "no_reply", "human_review"]

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "for",
    "from",
    "get",
    "have",
    "help",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "our",
    "please",
    "so",
    "that",
    "the",
    "their",
    "this",
    "to",
    "us",
    "we",
    "what",
    "when",
    "where",
    "with",
    "you",
    "your",
}

COURTESY_TERMS = {"thanks", "thank", "sorry", "appreciate", "happy"}
COMPLAINT_TERMS = {"frustrating", "urgent", "issue", "problem", "fraud", "unacceptable", "broke", "error"}
RISKY_CLAIMS = {
    "guarantee",
    "definitely refunded",
    "already refunded",
    "legal advice",
    "100% fixed",
}
ACTION_TERMS = {
    "confirm",
    "share",
    "send",
    "reply",
    "check",
    "download",
    "export",
    "reconnect",
    "reset",
    "update",
    "deactivate",
    "review",
    "investigate",
    "follow",
    "select",
    "open",
}
ESCALATION_TERMS = {"escalate", "escalating", "specialist", "security team", "billing team", "senior support"}


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
        self.rubric_weight = 0.65
        self.similarity_weight = 0.25
        self.length_weight = 0.10

    def score_generation(self, customer_email: str, draft_reply: str, gold_reply: str) -> GenerationScore:
        """
        Hybrid evaluation:
        1. Deterministic rubric built from issue coverage, actionability, correctness, and tone
        2. Optional LLM judge blended with deterministic rubric
        3. Reference similarity using lexical, phrase, and entity overlap
        4. Length sanity relative to expected support-reply length
        """
        deterministic = self._deterministic_rubric(customer_email, draft_reply, gold_reply)
        rubric = deterministic

        if self.llm_client.enabled:
            llm_rubric = self._llm_judge_rubric(customer_email, draft_reply, gold_reply)
            if llm_rubric:
                rubric = self._merge_rubrics(deterministic, llm_rubric)

        similarity_score = self._reference_similarity(draft_reply, gold_reply)
        length_score = self._length_sanity(draft_reply, gold_reply)

        hybrid = (
            self.rubric_weight * rubric["score_0_100"]
            + self.similarity_weight * similarity_score
            + self.length_weight * length_score
        )

        return GenerationScore(
            relevance=rubric["relevance"],
            completeness=rubric["completeness"],
            correctness=rubric["correctness"],
            tone=rubric["tone"],
            brevity=rubric["brevity"],
            rubric_score_0_100=round(rubric["score_0_100"], 2),
            reference_similarity_0_100=round(similarity_score, 2),
            length_penalty_0_100=round(length_score, 2),
            hybrid_score_0_100=round(hybrid, 2),
            rationale=rubric["rationale"],
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
                category_scores.setdefault(category, []).append(score)

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
                "judge_version": "deterministic_guardrailed_v2",
            },
        }

    def _llm_judge_rubric(self, customer_email: str, draft_reply: str, gold_reply: str) -> dict | None:
        system_prompt = (
            "You are a strict support-reply judge. "
            "Return JSON with integer fields relevance, completeness, correctness, tone, brevity in range 1-5, "
            "plus a short rationale and a list field failure_modes."
        )
        user_prompt = (
            f"Customer email:\n{customer_email}\n\n"
            f"Draft reply:\n{draft_reply}\n\n"
            f"Reference reply:\n{gold_reply}\n\n"
            "Score harshly when the draft is generic, misses the requested action, "
            "omits a specific next step, or makes ungrounded promises.\n"
            "Scoring rubric:\n"
            "- relevance: addresses the exact ask\n"
            "- completeness: includes the key next step or key missing information\n"
            "- correctness: no false promises, wrong policy, or factual drift\n"
            "- tone: professional and empathetic when needed\n"
            "- brevity: concise but still useful\n"
        )
        scored = self.llm_client.complete_json(system_prompt, user_prompt, temperature=0.0)
        if not isinstance(scored, dict):
            return None

        try:
            dims = [
                int(scored.get("relevance", 3)),
                int(scored.get("completeness", 3)),
                int(scored.get("correctness", 3)),
                int(scored.get("tone", 3)),
                int(scored.get("brevity", 3)),
            ]
        except (TypeError, ValueError):
            return None

        dims = [max(1, min(5, value)) for value in dims]
        score_0_100 = (sum(dims) / 25.0) * 100.0
        failure_modes = scored.get("failure_modes", [])
        if not isinstance(failure_modes, list):
            failure_modes = []
        failure_modes_text = ", ".join(str(item) for item in failure_modes[:4] if item)
        rationale = str(scored.get("rationale", "LLM rubric evaluation")).strip()
        if failure_modes_text:
            rationale = f"{rationale} Failure modes: {failure_modes_text}."

        return {
            "relevance": dims[0],
            "completeness": dims[1],
            "correctness": dims[2],
            "tone": dims[3],
            "brevity": dims[4],
            "score_0_100": round(score_0_100, 2),
            "rationale": rationale,
        }

    def _merge_rubrics(self, deterministic: dict, llm_rubric: dict) -> dict:
        merged_dims: list[int] = []
        for key in ["relevance", "completeness", "correctness", "tone", "brevity"]:
            det_value = deterministic[key]
            llm_value = llm_rubric[key]
            merged = int(round((0.65 * det_value) + (0.35 * llm_value)))
            merged = max(1, min(5, merged))

            if key == "correctness" and deterministic[key] <= 2:
                merged = min(merged, 2)
            if key == "completeness" and deterministic[key] <= 2:
                merged = min(merged, 3)

            merged_dims.append(merged)

        score_0_100 = (sum(merged_dims) / 25.0) * 100.0
        rationale = (
            f"Deterministic checks: {deterministic['rationale']} "
            f"LLM cross-check: {llm_rubric['rationale']}"
        ).strip()

        return {
            "relevance": merged_dims[0],
            "completeness": merged_dims[1],
            "correctness": merged_dims[2],
            "tone": merged_dims[3],
            "brevity": merged_dims[4],
            "score_0_100": round(score_0_100, 2),
            "rationale": rationale,
        }

    def _deterministic_rubric(self, customer_email: str, draft_reply: str, gold_reply: str) -> dict:
        customer_tokens = self._tokenize(customer_email)
        draft_tokens = self._tokenize(draft_reply)
        gold_tokens = self._tokenize(gold_reply)

        customer_keywords = self._extract_keywords(customer_email, limit=8)
        gold_keywords = self._extract_keywords(gold_reply, limit=10)
        draft_token_set = set(draft_tokens)

        customer_overlap = self._coverage_ratio(customer_keywords, draft_token_set)
        gold_overlap = self._coverage_ratio(gold_keywords, draft_token_set)

        customer_entities = self._extract_entities(customer_email)
        gold_entities = self._extract_entities(gold_reply)
        entity_overlap = self._entity_overlap(customer_entities | gold_entities, draft_reply)

        expected = self._expected_capabilities(customer_email, gold_reply)
        has_action_step = self._has_action_step(draft_reply)
        asks_for_details = self._contains_any(draft_reply, {"confirm", "share", "send", "reply", "provide"})
        includes_navigation = ">" in draft_reply or self._contains_any(
            draft_reply,
            {"settings", "reports", "billing", "security", "team", "integrations"},
        )
        mentions_escalation = self._contains_phrase(draft_reply, ESCALATION_TERMS)
        risky_claim_hits = [claim for claim in RISKY_CLAIMS if claim in draft_reply.lower()]
        complaint_context = self._contains_phrase(customer_email, COMPLAINT_TERMS)
        polite = self._contains_phrase(draft_reply, COURTESY_TERMS)

        relevance_score = self._band_score((0.55 * customer_overlap) + (0.25 * gold_overlap) + (0.20 * entity_overlap))
        completeness_signal = 0.45 * gold_overlap + 0.20 * entity_overlap + 0.20 * float(has_action_step) + 0.15 * float(
            self._meets_expected_behaviors(expected, asks_for_details, includes_navigation, mentions_escalation)
        )
        completeness_score = self._band_score(completeness_signal)

        correctness_score = 4
        correctness_notes: list[str] = []
        if risky_claim_hits:
            correctness_score = 1
            correctness_notes.append(f"contains risky unsupported claim(s): {', '.join(risky_claim_hits)}")
        if expected["needs_escalation"] and not mentions_escalation:
            correctness_score = min(correctness_score, 2)
            correctness_notes.append("misses required escalation language")
        if expected["needs_navigation"] and not includes_navigation:
            correctness_score = min(correctness_score, 3)
            correctness_notes.append("does not provide the expected product path or navigation hint")
        if expected["needs_specific_artifact"] and entity_overlap < 0.5:
            correctness_score = min(correctness_score, 2)
            correctness_notes.append("fails to preserve key invoice/account/export details")
        if gold_reply and gold_overlap < 0.25:
            correctness_score = min(correctness_score, 3)
            correctness_notes.append("drifts from the reference reply's core instructions")
        if not correctness_notes:
            correctness_score = 5 if gold_overlap >= 0.45 and entity_overlap >= 0.5 else 4

        tone_score = 4
        tone_notes: list[str] = []
        if complaint_context and not self._contains_phrase(draft_reply, {"sorry", "understand"}):
            tone_score = 2
            tone_notes.append("complaint context without empathy")
        elif not polite:
            tone_score = 3
            tone_notes.append("professional enough but lacks courtesy markers")
        elif complaint_context:
            tone_score = 5
        else:
            tone_score = 4

        brevity_score = self._brevity_score(draft_tokens, gold_tokens)

        dims = [relevance_score, completeness_score, correctness_score, tone_score, brevity_score]
        score_0_100 = (sum(dims) / 25.0) * 100.0

        rationale_parts = [
            f"keyword coverage customer={customer_overlap:.2f}, gold={gold_overlap:.2f}",
            f"entity overlap={entity_overlap:.2f}",
            f"action_step={'yes' if has_action_step else 'no'}",
        ]
        if correctness_notes:
            rationale_parts.append("correctness: " + "; ".join(correctness_notes))
        if tone_notes:
            rationale_parts.append("tone: " + "; ".join(tone_notes))

        return {
            "relevance": relevance_score,
            "completeness": completeness_score,
            "correctness": correctness_score,
            "tone": tone_score,
            "brevity": brevity_score,
            "score_0_100": round(score_0_100, 2),
            "rationale": ". ".join(rationale_parts) + ".",
        }

    def _reference_similarity(self, draft_reply: str, gold_reply: str) -> float:
        if not gold_reply:
            return 60.0

        draft_tokens = self._tokenize(draft_reply)
        gold_tokens = self._tokenize(gold_reply)
        if not draft_tokens or not gold_tokens:
            return 30.0

        unigram_f1 = self._f1_overlap(draft_tokens, gold_tokens)
        draft_bigrams = self._ngrams(draft_tokens, 2)
        gold_bigrams = self._ngrams(gold_tokens, 2)
        bigram_recall = self._overlap_ratio(gold_bigrams, draft_bigrams)

        gold_keywords = self._extract_keywords(gold_reply, limit=10)
        keyword_coverage = self._coverage_ratio(gold_keywords, set(draft_tokens))

        gold_entities = self._extract_entities(gold_reply)
        entity_overlap = self._entity_overlap(gold_entities, draft_reply) if gold_entities else keyword_coverage

        score = (
            0.40 * unigram_f1
            + 0.20 * bigram_recall
            + 0.25 * keyword_coverage
            + 0.15 * entity_overlap
        ) * 100.0
        return max(0.0, min(100.0, score))

    def _length_sanity(self, draft_reply: str, gold_reply: str) -> float:
        draft_words = len(self._tokenize(draft_reply))
        gold_words = len(self._tokenize(gold_reply))

        if draft_words < 12:
            return 20.0
        if draft_words > 180:
            return 45.0

        if gold_words:
            ratio = draft_words / gold_words if gold_words else 1.0
            if 0.7 <= ratio <= 1.35:
                return 100.0
            if 0.5 <= ratio <= 1.6:
                return 82.0
            if 0.35 <= ratio <= 2.0:
                return 65.0
            return 45.0

        if 18 <= draft_words <= 90:
            return 95.0
        if 12 <= draft_words <= 130:
            return 80.0
        return 55.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _extract_keywords(self, text: str, limit: int = 8) -> list[str]:
        tokens = [token for token in self._tokenize(text) if token not in STOPWORDS and len(token) > 2]
        ordered: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if token not in seen:
                ordered.append(token)
                seen.add(token)
            if len(ordered) >= limit:
                break
        return ordered

    @staticmethod
    def _coverage_ratio(keywords: list[str], draft_token_set: set[str]) -> float:
        if not keywords:
            return 0.5
        covered = sum(1 for token in keywords if token in draft_token_set)
        return covered / len(keywords)

    @staticmethod
    def _extract_entities(text: str) -> set[str]:
        entities = set(re.findall(r"inv-\d+", text.lower()))
        entities.update(re.findall(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", text.lower()))
        entities.update(re.findall(r"\b(?:okta|sso|pdf|csv|slack|api|webhook)\b", text.lower()))
        return entities

    @staticmethod
    def _entity_overlap(entities: set[str], draft_reply: str) -> float:
        if not entities:
            return 0.5
        draft_lower = draft_reply.lower()
        matched = sum(1 for entity in entities if entity in draft_lower)
        return matched / len(entities)

    @staticmethod
    def _contains_any(text: str, terms: set[str]) -> bool:
        token_set = set(re.findall(r"[a-z0-9]+", text.lower()))
        return any(term in token_set for term in terms)

    @staticmethod
    def _contains_phrase(text: str, phrases: set[str]) -> bool:
        lower = text.lower()
        return any(phrase in lower for phrase in phrases)

    def _has_action_step(self, text: str) -> bool:
        return ">" in text or self._contains_any(text, ACTION_TERMS)

    @staticmethod
    def _expected_capabilities(customer_email: str, gold_reply: str) -> dict[str, bool]:
        combined = f"{customer_email}\n{gold_reply}".lower()
        return {
            "needs_escalation": any(term in combined for term in ["refund", "legal", "security", "fraud", "unauthorized"]),
            "needs_navigation": any(term in combined for term in ["settings >", "reports >", "billing >", "security >", "team >"]),
            "needs_details_request": any(term in combined for term in ["confirm", "share", "send us", "reply with", "provide"]),
            "needs_specific_artifact": any(term in combined for term in ["invoice", "inv-", "export", "receipt", "pdf", "csv"]),
        }

    @staticmethod
    def _meets_expected_behaviors(expected: dict[str, bool], asks_for_details: bool, includes_navigation: bool, mentions_escalation: bool) -> bool:
        checks: list[bool] = []
        if expected["needs_details_request"]:
            checks.append(asks_for_details)
        if expected["needs_navigation"]:
            checks.append(includes_navigation)
        if expected["needs_escalation"]:
            checks.append(mentions_escalation)
        if not checks:
            return True
        return all(checks)

    @staticmethod
    def _brevity_score(draft_tokens: list[str], gold_tokens: list[str]) -> int:
        draft_len = len(draft_tokens)
        gold_len = len(gold_tokens)
        if draft_len < 12 or draft_len > 180:
            return 1
        if not gold_len:
            return 5 if 18 <= draft_len <= 90 else 4
        ratio = draft_len / gold_len if gold_len else 1.0
        if 0.75 <= ratio <= 1.25:
            return 5
        if 0.55 <= ratio <= 1.5:
            return 4
        if 0.4 <= ratio <= 1.8:
            return 3
        return 2

    @staticmethod
    def _band_score(signal: float) -> int:
        if signal >= 0.85:
            return 5
        if signal >= 0.65:
            return 4
        if signal >= 0.45:
            return 3
        if signal >= 0.25:
            return 2
        return 1

    @staticmethod
    def _f1_overlap(draft_tokens: list[str], gold_tokens: list[str]) -> float:
        draft_counts: dict[str, int] = {}
        gold_counts: dict[str, int] = {}
        for token in draft_tokens:
            draft_counts[token] = draft_counts.get(token, 0) + 1
        for token in gold_tokens:
            gold_counts[token] = gold_counts.get(token, 0) + 1

        overlap = 0
        for token, count in draft_counts.items():
            overlap += min(count, gold_counts.get(token, 0))

        if overlap == 0:
            return 0.0

        precision = overlap / len(draft_tokens)
        recall = overlap / len(gold_tokens)
        return (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    @staticmethod
    def _ngrams(tokens: list[str], size: int) -> set[tuple[str, ...]]:
        if len(tokens) < size:
            return set()
        return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}

    @staticmethod
    def _overlap_ratio(expected: set[tuple[str, ...]], actual: set[tuple[str, ...]]) -> float:
        if not expected:
            return 0.5
        return len(expected & actual) / len(expected)
