from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


class LLMClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
        self.api_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/responses").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def complete_json(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        payload = {
            "model": self.model,
            "temperature": temperature,
            "input": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
            "text": {"format": {"type": "json_object"}},
        }
        return self._post(payload)

    def complete_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str | None:
        if not self.enabled:
            return None

        payload = {
            "model": self.model,
            "temperature": temperature,
            "input": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
        }
        response = self._post(payload)
        if not response:
            return None
        return self._extract_text(response)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.api_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=45) as resp:
                raw = resp.read().decode("utf-8")
        except (error.HTTPError, error.URLError, TimeoutError):
            return None

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed

    def _extract_text(self, response_payload: dict[str, Any]) -> str | None:
        if isinstance(response_payload.get("output_text"), str):
            return response_payload["output_text"].strip()

        output = response_payload.get("output")
        if not isinstance(output, list):
            return None

        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for piece in content:
                if not isinstance(piece, dict):
                    continue
                text = piece.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        final = "\n".join(chunks).strip()
        return final or None

