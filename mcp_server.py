"""AgenticShop as an MCP (Model Context Protocol) server over stdio.

MCP = open standard (JSON-RPC 2.0) so ANY MCP-capable agent or app (host/client) can discover
and call this shop's tools without custom integration code.
  - initialize     : handshake, protocol version, capabilities
  - tools/list     : advertise tools with name, description, inputSchema
  - tools/call     : run a tool, return content blocks (+ isError)
  - resources/list : read-only data the client may load as context (here: policy documents)
  - resources/read : fetch one resource
Transports: stdio (this file, a local subprocess) or Streamable HTTP (remote servers).

Run it directly to type JSON lines yourself, or run mcp_client_demo.py.
Note: a real server would authenticate the caller (e.g. OAuth). Here the session is fixed to C001.
"""
import json
import sys

from shop.memory import LongTermMemory
from shop.store import POLICIES
from shop.tools import ALL_TOOLS, ToolContext, registry

PROTOCOL_VERSION = "2025-06-18"
EXPOSED = ["search_products", "add_to_cart", "view_cart", "track_order", "search_policies"]  # no refunds over MCP


def handle(req: dict, ctx: ToolContext, reg) -> dict | None:
    method, rid, params = req.get("method"), req.get("id"), req.get("params") or {}
    if rid is None:                                   # a notification: no response
        return None
    if method == "initialize":
        result = {"protocolVersion": PROTOCOL_VERSION,
                  "capabilities": {"tools": {}, "resources": {}},
                  "serverInfo": {"name": "agentic-shop", "version": "1.0.0"}}
    elif method == "tools/list":
        result = {"tools": [{"name": n, "description": ALL_TOOLS[n].description,
                             "inputSchema": ALL_TOOLS[n].parameters} for n in EXPOSED]}
    elif method == "tools/call":
        out = reg.call(ctx, params.get("name", ""), params.get("arguments") or {})
        result = {"content": [{"type": "text", "text": out}], "isError": out.startswith("ERROR")}
    elif method == "resources/list":
        result = {"resources": [{"uri": f"policy://{k}", "name": f"{k} policy", "mimeType": "text/plain"}
                                for k in POLICIES]}
    elif method == "resources/read":
        key = params.get("uri", "").removeprefix("policy://")
        if key not in POLICIES:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": "unknown resource"}}
        result = {"contents": [{"uri": params["uri"], "mimeType": "text/plain", "text": POLICIES[key]}]}
    else:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"Method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def main():
    ctx = ToolContext.create("C001", profiles=LongTermMemory())
    reg = registry(*EXPOSED)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            resp = handle(json.loads(line), ctx, reg)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
