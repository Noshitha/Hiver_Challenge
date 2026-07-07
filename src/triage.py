from __future__ import annotations

import json
import re

from models import EmailRecord, TriageResult
from llm_client import LLMClient


NO_REPLY_PATTERNS = [
    r"newsletter",
    r"no action needed",
    r"fyi",
    r"out of office",
    r"auto[- ]?reply",
    r"\bseen\b",
]

HUMAN_REVIEW_PATTERNS = [
    r"refund",
    r"chargeback",
    r"legal",
    r"breach",
    r"security",
    r"unacceptable",
    r"escalat",
    r"lawyer",
    r"threat",
]


class TriageEngine:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def predict(self, email: EmailRecord) -> TriageResult:
        normalized = f"{email.subject}\n{email.body}\n{email.thread_history}".lower()

        if self._matches_any(normalized, HUMAN_REVIEW_PATTERNS):
            return TriageResult(label="human_review", confidence=0.92, reason="Matched high-risk escalation/security pattern.")

        if self._matches_any(normalized, NO_REPLY_PATTERNS):
            return TriageResult(label="no_reply", confidence=0.9, reason="Matched informational/auto-message pattern.")

        if "?" in email.body or "please" in normalized or "help" in normalized:
            heuristic_guess = TriageResult(label="needs_reply", confidence=0.72, reason="Contains direct question/request.")
        else:
            heuristic_guess = TriageResult(label="no_reply", confidence=0.62, reason="No clear question or request.")

        if heuristic_guess.confidence >= 0.8 or not self.llm_client.enabled:
            return heuristic_guess

        llm = self._llm_classify(email)
        return llm if llm is not None else heuristic_guess

    def _llm_classify(self, email: EmailRecord) -> TriageResult | None:
        system_prompt = (
            "You classify support emails into exactly one label: needs_reply, no_reply, human_review. "
            "Return strict JSON with keys: label, confidence, reason."
        )
        user_prompt = (
            f"Subject: {email.subject}\n"
            f"Body: {email.body}\n"
            f"Thread: {email.thread_history}\n"
            "Rules:\n"
            "- human_review: refunds, legal/security issues, aggressive escalations, unclear risky actions.\n"
            "- no_reply: newsletters, FYI, auto-replies, acknowledgements without ask.\n"
            "- needs_reply: customer asks a question or requests action.\n"
        )
        result = self.llm_client.complete_json(system_prompt, user_prompt, temperature=0.0)
        if not result:
            return None

        parsed = self._parse_json_payload(result)
        if not parsed:
            return None
        label = parsed.get("label")
        confidence = parsed.get("confidence")
        reason = parsed.get("reason")
        if label not in {"needs_reply", "no_reply", "human_review"}:
            return None
        try:
            c = float(confidence)
        except (TypeError, ValueError):
            c = 0.55
        return TriageResult(label=label, confidence=max(0.0, min(1.0, c)), reason=str(reason or "LLM classification"))

    @staticmethod
    def _matches_any(text: str, patterns: list[str]) -> bool:
        return any(re.search(p, text) for p in patterns)

    @staticmethod
    def _parse_json_payload(result: dict) -> dict | None:
        if isinstance(result, dict) and {"label", "confidence", "reason"}.issubset(result.keys()):
            return result

        raw_text = result.get("output_text")
        if isinstance(raw_text, str):
            try:
                return json.loads(raw_text)
            except json.JSONDecodeError:
                return None
        return None

