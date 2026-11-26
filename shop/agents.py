"""Multi-agent system: a Supervisor routes each request to specialist agents.

Specialists (each with ONLY the tools it needs: least privilege, better tool choice):
  sales    - product discovery, recommendations, cart, coupons, checkout
  support  - store policies via RAG (grounded answers with sources)
  orders   - order tracking
  refunds  - refunds, with human approval above the limit
Pattern: hierarchical / supervisor. Several intents in one message run sequentially,
and the supervisor merges the answers. Compare OCI Fusion AI Agent Studio supervisor
agents and OCI Generative AI Agents with multiple tools.
"""
from __future__ import annotations

import re

from .agent import Agent
from .guardrails import Guardrails
from .llm import MockLLM
from .memory import Session
from .tools import ToolContext, registry

SPECIALISTS = {
    "sales": ("You are a friendly sales assistant. Recommend only products returned by tools. "
              "Never invent prices or discounts.",
              ["search_products", "recommend_for_me", "add_to_cart", "view_cart", "apply_coupon",
               "checkout", "save_preference"]),
    "support": ("You answer policy questions ONLY from retrieved documents and cite the source. "
                "If nothing is found, say so and offer a human.", ["search_policies"]),
    "orders": ("You help customers track their own orders.", ["track_order"]),
    "refunds": ("You process refunds following policy. High-value refunds need manager approval.",
                ["request_refund", "track_order"]),
}

ROUTES = {  # the supervisor's intent classifier (a real supervisor asks the LLM to classify)
    "sales": r"find|search|show me|looking for|do you have|recommend|suggest|for me\b|\badd\b|cart|basket|coupon|"
             r"code|promo|check ?out|buy|place (?:the |my )?order|my (?:shoe size|size|interest|budget) is|i (?:love|like|enjoy)",
    "refunds": r"order\s*#?\s*\d{4}.*(refund|return|money back|damaged|broken|faulty)|"
               r"(refund|return|money back).*order\s*#?\s*\d{4}",
    "orders": r"order\s*#?\s*\d{4}",
    "support": r"polic|return|refund|shipping|delivery|warranty|exchange|privacy|how long",
}


class Supervisor:
    def __init__(self, llm=None, guardrails: Guardrails | None = None):
        self.guardrails = guardrails or Guardrails()
        llm = llm or MockLLM()
        self.agents = {name: Agent(f"{name}-agent", instr, registry(*tools), llm=llm)
                       for name, (instr, tools) in SPECIALISTS.items()}

    def route(self, text: str) -> list[str]:
        q = text.lower()
        hits = [n for n, pat in ROUTES.items() if re.search(pat, q)]
        if "refunds" in hits:                       # refund intent wins over tracking / generic policy
            hits = [h for h in hits if h not in ("orders", "support")]
        if "orders" in hits and "support" in hits:
            hits.remove("support")
        return hits

    def handle(self, text: str, session: Session, ctx: ToolContext, trace: list | None = None) -> str:
        trace = trace if trace is not None else []
        ok, reason, clean = self.guardrails.check_input(text)    # guardrail once, at the front door
        if not ok:
            trace.append(("supervisor", "guardrail", f"BLOCKED: {reason}"))
            return f"Sorry, I can't help with that ({reason})."
        if reason:
            trace.append(("supervisor", "guardrail", reason))
        targets = self.route(clean)
        trace.append(("supervisor", "route", ", ".join(targets) or "none"))
        if not targets:
            return ("I can help with products, your cart and checkout, order tracking, refunds and "
                    "store policies. What would you like to do?")
        answers = [self.agents[t].run(clean, session, ctx, trace) for t in targets]
        return self.guardrails.check_output("\n".join(answers))
