from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TriageLabel = Literal["needs_reply", "no_reply", "human_review"]


@dataclass
class EmailRecord:
    email_id: str
    subject: str
    body: str
    thread_history: str
    gold_needs_reply: TriageLabel
    gold_reply: str
    category: str = ""


@dataclass
class TriageResult:
    label: TriageLabel
    confidence: float
    reason: str


@dataclass
class GenerationResult:
    draft_text: str
    confidence: float
    flags: list[str]
    mode: str
    retrieved_example_ids: list[str] = None
    
    def __post_init__(self):
        if self.retrieved_example_ids is None:
            self.retrieved_example_ids = []

