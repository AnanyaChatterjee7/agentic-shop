"""The 'model'. MockLLM imitates a tool-calling chat model offline and deterministically.

A real LLM reads the tool descriptions and decides which tools to call. MockLLM does that
with pattern rules so you can see each decision. OCIGenAIChat shows the OCI Generative AI
call you would swap in (needs `pip install oci` and an OCI account).

Exam concepts: chat roles (system/user/assistant/tool), tool calling, temperature,
tokens/context window, the model proposes and the runtime executes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    type: str                               # "tool_call" | "final"
    content: str = ""
    thought: str = ""
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)


def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)            # ~4 characters per token (rough rule of thumb)


def plan_tools(question: str, tool_names: set[str]) -> list[tuple[str, dict]]:
    """What a tool-calling model would infer from the request + tool descriptions."""
    q = question.lower()
    plan: list[tuple[str, dict]] = []

    def want(name, args):
        if name in tool_names:
            plan.append((name, args))

    m = re.search(r"my (shoe size|size|favou?rite colou?r|interest|budget) is ([\w$ ]+?)(?:[.,!]| and |$)", q)
    if m:
        want("save_preference", {"key": m.group(1).replace(" ", "_"), "value": m.group(2).strip()})
    m = re.search(r"i (?:love|like|enjoy) ([a-z ]+?)(?:[.,!]| and |$)", q)
    if m:
        want("save_preference", {"key": "interest", "value": m.group(1).strip()})

    order = re.search(r"order\s*#?\s*(\d{4})", q)
    if re.search(r"recommend|suggest|for me\b", q):
        want("recommend_for_me", {})
    elif re.search(r"\b(find|search|show me|looking for|do you have|need)\b", q) and not order:
        args = {"query": question}
        price = re.search(r"(?:under|below|less than)\s*\$?(\d+(?:\.\d+)?)", q)
        if price:
            args["max_price"] = float(price.group(1))
        want("search_products", args)

    m = re.search(r"\badd (?:(\d+) (?:x )?)?(?:the |a |an )?(.+?)(?: to (?:my |the )?(?:cart|basket)|,| and |$)", q)
    if m:
        args = {"product": m.group(2).strip()}
        if m.group(1):
            args["quantity"] = int(m.group(1))
        want("add_to_cart", args)

    m = re.search(r"\b(?:coupon|code|promo)\s+([a-z0-9]+)", q)
    if m:
        want("apply_coupon", {"code": m.group(1).upper()})
    if re.search(r"\b(check ?out|place (?:the |my )?order|buy (?:it|them|now))\b", q):
        want("checkout", {})
    elif re.search(r"\b(my cart|the cart|basket|what.?s in)", q) and not re.search(r"\badd\b", q):
        want("view_cart", {})

    if order:
        if re.search(r"refund|return|money back|damaged|broken|faulty", q):
            reason = "damaged" if re.search(r"damaged|broken|faulty", q) else "customer request"
            want("request_refund", {"order_id": order.group(1), "reason": reason})
        else:
            want("track_order", {"order_id": order.group(1)})
    elif re.search(r"polic|return|refund|shipping|delivery|warranty|exchange|privacy|how long", q):
        want("search_policies", {"query": question})
    return plan


class MockLLM:
    def __init__(self, temperature: float = 0.0):
        self.temperature = temperature

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        names = {t["name"] for t in tools}
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        question = messages[last_user]["content"]
        observations = [m for m in messages[last_user + 1:] if m["role"] == "tool"]
        done = [(m["name"], m.get("args")) for m in observations]

        for name, args in plan_tools(question, names):
            if (name, args) not in done:
                return LLMResponse("tool_call", tool_name=name, tool_args=args,
                                   thought=f"The user needs '{name}'; I will call it.")
        if observations:
            lines = [f"- {m['content']}" for m in observations]
            return LLMResponse("final", thought="I have everything I need.",
                               content="Here's what I found/did:\n" + "\n".join(lines))
        return LLMResponse("final", thought="No tool fits; answer directly without inventing facts.",
                           content="I can help with finding products, your cart and checkout, order tracking, "
                                   "refunds and store policies. What would you like to do?")


class OCIGenAIChat:   # pragma: no cover - reference only, not used by the demos
    """Swap-in for MockLLM using OCI Generative AI (on-demand serving mode)."""

    def __init__(self, compartment_id: str, model_id: str = "cohere.command-r-plus-08-2024",
                 endpoint: str = "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"):
        import oci
        from oci.generative_ai_inference import GenerativeAiInferenceClient
        self.client = GenerativeAiInferenceClient(oci.config.from_file(), service_endpoint=endpoint)
        self.compartment_id, self.model_id = compartment_id, model_id

    def chat(self, messages, tools):
        from oci.generative_ai_inference import models as m
        req = m.CohereChatRequest(
            message=messages[-1]["content"], temperature=0.0, max_tokens=600,
            preamble_override=next((x["content"] for x in messages if x["role"] == "system"), None),
            tools=[m.CohereTool(name=t["name"], description=t["description"],
                                parameter_definitions={k: m.CohereParameterDefinition(type=v["type"], is_required=k in t["parameters"]["required"])
                                                       for k, v in t["parameters"]["properties"].items()})
                   for t in tools])
        resp = self.client.chat(m.ChatDetails(compartment_id=self.compartment_id, chat_request=req,
                                              serving_mode=m.OnDemandServingMode(model_id=self.model_id)))
        r = resp.data.chat_response
        if getattr(r, "tool_calls", None):
            call = r.tool_calls[0]
            return LLMResponse("tool_call", tool_name=call.name, tool_args=call.parameters)
        return LLMResponse("final", content=r.text)
