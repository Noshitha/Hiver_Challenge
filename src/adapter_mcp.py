from __future__ import annotations

from models import EmailRecord
from pipeline import run_pipeline


def run_from_inbox_entries(entries: list[dict]) -> list[dict]:
    """
    Thin adapter for live inbox integrations.
    It converts external email payloads into EmailRecord shape and reuses the same pipeline.
    """
    records: list[EmailRecord] = []
    for entry in entries:
        records.append(
            EmailRecord(
                email_id=str(entry.get("email_id", "")),
                subject=str(entry.get("subject", "")),
                body=str(entry.get("body", "")),
                thread_history=str(entry.get("thread_history", "")),
                # Live inbox items typically don't include gold labels; use safe defaults.
                gold_needs_reply="no_reply",
                gold_reply="",
                category=str(entry.get("category", "live_inbox")),
            )
        )
    return run_pipeline(records)


def to_adapter_payload(items: list[dict]) -> list[dict]:
    """
    Optional helper for tools that require a compact response schema.
    """
    compact = []
    for item in items:
        compact.append(
            {
                "email_id": item["email_id"],
                "triage_label": item["triage"]["predicted"],
                "triage_confidence": item["triage"]["confidence"],
                "draft_text": (item.get("generation") or {}).get("draft_text", ""),
                "flags": (item.get("generation") or {}).get("flags", []),
            }
        )
    return compact
