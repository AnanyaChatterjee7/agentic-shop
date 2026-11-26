"""A ReAct agent runtime: Thought -> Action (tool call) -> Observation -> ... -> Final answer.

The RUNTIME (this class, or OCI Generative AI Agents in the cloud) owns the loop:
it sends messages + tool schemas to the model, executes tool calls, feeds results back,
applies guardrails, and stops on a final answer or when max_steps is hit.
Every step is written to a trace (observability / audit log).
"""
from __future__ import annotations

from .guardrails import Guardrails
from .llm import MockLLM
from .memory import Session
from .tools import ToolContext, ToolRegistry


class Agent:
    def __init__(self, name: str, instructions: str, tools: ToolRegistry, llm=None,
                 max_steps: int = 8, guardrails: Guardrails | None = None):
        self.name, self.instructions, self.tools = name, instructions, tools
        self.llm = llm or MockLLM()
        self.max_steps = max_steps
        self.guardrails = guardrails

    def run(self, user_input: str, session: Session, ctx: ToolContext, trace: list | None = None) -> str:
        trace = trace if trace is not None else []
        log = lambda kind, text: trace.append((self.name, kind, text))

        if self.guardrails:
            ok, reason, user_input = self.guardrails.check_input(user_input)
            if not ok:
                log("guardrail", f"BLOCKED: {reason}")
                return f"Sorry, I can't help with that ({reason})."
            if reason:
                log("guardrail", reason)

        session.add({"role": "user", "content": user_input})
        for step in range(1, self.max_steps + 1):
            resp = self.llm.chat(session.window(self.instructions), self.tools.schemas())
            log("thought", resp.thought)
            if resp.type == "final":
                answer = self.guardrails.check_output(resp.content) if self.guardrails else resp.content
                session.add({"role": "assistant", "content": answer})
                return answer
            log("action", f"{resp.tool_name}({resp.tool_args})")
            obs = self.tools.call(ctx, resp.tool_name, resp.tool_args)
            log("observation", obs)
            session.add({"role": "assistant", "content": f"calling {resp.tool_name}"})
            session.add({"role": "tool", "name": resp.tool_name, "args": resp.tool_args, "content": obs})
        log("stop", "max_steps reached")
        return "Sorry, that took too many steps. A human agent will follow up."
