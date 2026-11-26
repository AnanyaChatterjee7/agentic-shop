"""Guardrails around the agents (Responsible AI in practice).

Input: prompt-injection detection, harmful/off-limits requests, size limits, PII masking.
Output: PII masking, no secrets, no fabricated discounts.
Plus (elsewhere): human approval for high-value refunds, authorization checks in tools,
step limits, audit trace of every action (accountability and transparency).
"""
from __future__ import annotations

import re

INJECTION = [r"ignore (all |any )?(the )?(previous|prior|above|your) (instructions|rules)",
             r"(reveal|show|print) (your|the) (system )?(prompt|instructions)",
             r"you are now", r"act as (an? )?(admin|developer|system)",
             r"(give|make) (me )?(everything|it|all) (for )?free", r"set (the )?price to"]
BLOCKED = [r"stolen card", r"hack", r"other customer'?s? (data|address|orders?)"]
PII = {"CARD": r"\b(?:\d[ -]?){12,15}\d\b",
       "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.]+",
       "PHONE": r"\+\d{1,3}[\s-]?\d{4,5}[\s-]?\d{4,6}\b|\b\d{10}\b"}
MAX_INPUT_CHARS = 1000


def mask_pii(text: str) -> tuple[str, list[str]]:
    found = []
    for label, pat in PII.items():
        if re.search(pat, text):
            found.append(label)
            text = re.sub(pat, f"<{label}>", text)
    return text, found


class Guardrails:
    def check_input(self, text: str) -> tuple[bool, str, str]:
        """Returns (allowed, reason, sanitised_text)."""
        if len(text) > MAX_INPUT_CHARS:
            return False, "message too long", text
        low = text.lower()
        if any(re.search(p, low) for p in INJECTION):
            return False, "possible prompt injection / policy override attempt", text
        if any(re.search(p, low) for p in BLOCKED):
            return False, "request not allowed", text
        clean, found = mask_pii(text)
        return True, ("masked " + ", ".join(found)) if found else "", clean

    def check_output(self, text: str) -> str:
        return mask_pii(text)[0]
