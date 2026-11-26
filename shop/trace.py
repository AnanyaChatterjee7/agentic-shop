"""Pretty-print an agent trace (observability: what the agents thought and did)."""

ICONS = {"route": "ROUTE", "guardrail": "GUARD", "thought": "THINK", "action": "ACT  ",
         "observation": "OBS  ", "stop": "STOP "}


def print_trace(trace: list) -> None:
    for who, kind, text in trace:
        print(f"    {ICONS.get(kind, kind):5} [{who}] {text}")
