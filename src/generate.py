from __future__ import annotations

import re

from llm_client import LLMClient
from models import EmailRecord, GenerationResult


class ReplyGenerator:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def generate(self, email: EmailRecord) -> GenerationResult:
        if self.llm_client.enabled:
            drafted = self._generate_with_llm(email)
            if drafted:
                flags = self._safety_flags(drafted)
                confidence = 0.78 if not flags else 0.64
                return GenerationResult(draft_text=drafted, confidence=confidence, flags=flags, mode="llm")
        drafted = self._generate_heuristic(email)
        flags = self._safety_flags(drafted)
        confidence = 0.66 if not flags else 0.58
        return GenerationResult(draft_text=drafted, confidence=confidence, flags=flags, mode="heuristic")

    def _generate_with_llm(self, email: EmailRecord) -> str | None:
        system_prompt = (
            "You are a customer support email assistant. "
            "Write concise, empathetic, action-oriented replies. "
            "Do not invent policies, timelines, discounts, or promises."
        )
        user_prompt = (
            f"Subject: {email.subject}\n"
            f"Customer email: {email.body}\n"
            f"Thread context: {email.thread_history}\n\n"
            "Requirements:\n"
            "- 3-6 sentences\n"
            "- directly answer the ask\n"
            "- include clear next step\n"
            "- avoid hallucinated commitments\n"
        )
        return self.llm_client.complete_text(system_prompt, user_prompt, temperature=0.2)

    def _generate_heuristic(self, email: EmailRecord) -> str:
        ask = self._extract_ask(email.body)
        return (
            "Hi there,\n\n"
            "Thanks for reaching out — I’m happy to help. "
            f"I understand you need support with: {ask}. "
            "Please share any relevant account details or screenshots, and we’ll take the next step right away.\n\n"
            "Best,\nSupport Team"
        )

    @staticmethod
    def _extract_ask(body: str) -> str:
        first_sentence = re.split(r"[.!?]\s+", body.strip())[0]
        if not first_sentence:
            return "your request"
        return first_sentence[:180]

    @staticmethod
    def _safety_flags(draft_text: str) -> list[str]:
        text = draft_text.lower()
        risky = []
        if "guarantee" in text:
            risky.append("contains_guarantee")
        if "we already refunded" in text:
            risky.append("unverified_refund_claim")
        if "legal advice" in text:
            risky.append("legal_advice_language")
        return risky

