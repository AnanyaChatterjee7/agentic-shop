"""Agent memory.

- Short-term memory (Session): this conversation's messages. Trimmed and summarised to
  fit the context window, and cleared after a timeout (like an OCI Agents session timeout).
- Long-term memory (LongTermMemory): customer preferences persisted across sessions
  (a JSON file here; an Oracle table or vector store in production). Store only what you
  need and let customers delete it (privacy / Responsible AI).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .llm import count_tokens


class Session:
    def __init__(self, customer_id: str, max_tokens: int = 600, timeout_s: int = 1800):
        self.customer_id = customer_id
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.messages: list[dict] = []
        self.summary: list[str] = []
        self.last_active = time.time()

    def add(self, msg: dict):
        if time.time() - self.last_active > self.timeout_s:
            self.messages, self.summary = [], []          # session expired
        self.last_active = time.time()
        self.messages.append(msg)
        while self.tokens() > self.max_tokens and len(self.messages) > 4:
            old = self.messages.pop(0)
            if old["role"] == "user":                      # a real agent would ask the LLM to summarise
                self.summary.append(" ".join(old["content"].split()[:8]))

    def tokens(self) -> int:
        return sum(count_tokens(m["content"]) for m in self.messages)

    def window(self, system_prompt: str) -> list[dict]:
        sys = system_prompt + (f"\nEarlier in this chat the user asked about: {' / '.join(self.summary)}"
                               if self.summary else "")
        return [{"role": "system", "content": sys}] + self.messages


class LongTermMemory:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.data: dict[str, dict] = {}
        if self.path and self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def remember(self, customer: str, key: str, value: str):
        self.data.setdefault(customer, {})[key] = value
        self._save()

    def get(self, customer: str) -> dict:
        return dict(self.data.get(customer, {}))

    def forget(self, customer: str):                       # "right to be forgotten"
        self.data.pop(customer, None)
        self._save()

    def _save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
