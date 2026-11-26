"""MCP client: starts mcp_server.py as a subprocess and talks JSON-RPC over stdio.

This is what an MCP host (an IDE, a desktop AI app, or another agent) does under the hood.
"""
import json
import subprocess
import sys
from pathlib import Path

SERVER = Path(__file__).with_name("mcp_server.py")


def main():
    proc = subprocess.Popen([sys.executable, str(SERVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            text=True, encoding="utf-8", cwd=SERVER.parent)
    msg_id = 0

    def send(method, params=None, notify=False):
        nonlocal msg_id
        req = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            msg_id += 1
            req["id"] = msg_id
        print(f"\n-> {json.dumps(req)}")
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        if notify:
            return None
        resp = json.loads(proc.stdout.readline())
        text = json.dumps(resp)
        print(f"<- {text[:300]}{'...' if len(text) > 300 else ''}")
        return resp

    send("initialize", {"protocolVersion": "2025-06-18", "clientInfo": {"name": "demo-client", "version": "1"},
                        "capabilities": {}})
    send("notifications/initialized", notify=True)
    tools = send("tools/list")["result"]["tools"]
    print("\nDiscovered tools:", [t["name"] for t in tools])
    send("tools/call", {"name": "search_products", "arguments": {"query": "wireless earbuds for running"}})
    send("tools/call", {"name": "add_to_cart", "arguments": {"product": "EL-302", "quantity": 1}})
    send("tools/call", {"name": "view_cart", "arguments": {}})
    send("tools/call", {"name": "request_refund", "arguments": {"order_id": "5002", "reason": "x"}})
    send("resources/list")
    send("resources/read", {"uri": "policy://shipping"})
    proc.stdin.close()
    proc.wait(timeout=5)


if __name__ == "__main__":
    main()
