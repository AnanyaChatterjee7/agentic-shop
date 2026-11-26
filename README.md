# AgenticShop: an e-commerce store run by AI agents

A working online sports store where AI agents help customers find products, manage their
cart, check out, track orders, get refunds and ask about store policies. Each feature is built
to show one topic from **Oracle Agentic AI Foundations Associate (1Z0-1157-26)**.

It is pure Python 3.10+ with no packages to install and no API keys. A deterministic mock model
(`shop/llm.py`) makes the agent decisions, so every Thought → Action → Observation step is
visible and repeatable. `OCIGenAIChat` in the same file shows how to swap in OCI Generative AI.

## Run it (PyCharm or terminal)

Open the **`agentic-shop`** folder itself as the PyCharm project, pick a Python 3.10+ interpreter,
then right-click a file and choose **Run**.

| File | What it does |
|---|---|
| `chat.py` | **Shop with the agents interactively.** Log in as C001 (Priya) or C002 (Alex). You also act as the manager who approves big refunds. Commands: `/cart` `/prefs` `/forget` `/switch` `/trace off` `/quit` |
| `demo.py` | A 14-step scripted shopping journey that touches every exam topic, with the full agent trace |
| `mcp_client_demo.py` | Starts `mcp_server.py` and talks to it over MCP (JSON-RPC on stdio) |
| `evals.py` | Agent evaluation suite: routing, tool selection and guardrails (exits non-zero on failure) |
| `sql/schema.sql` | The store on Oracle Database 23ai with AI Vector Search. Run it on an Always Free Autonomous DB |

## Architecture

```
customer ──► Guardrails ──► Supervisor agent (routes by intent)
                               ├── sales-agent    search_products, recommend_for_me, add_to_cart,
                               │                  view_cart, apply_coupon, checkout, save_preference
                               ├── support-agent  search_policies (RAG over policy docs)
                               ├── orders-agent   track_order
                               └── refunds-agent  request_refund (manager approval > $100)
each agent:  ReAct loop ── LLM (mock / OCI GenAI) ── tools ── Store + VectorIndex
memory:      Session (short-term, trimmed + summarised)  |  LongTermMemory (data/profiles.json)
MCP server:  the same tools exposed to any MCP client
```

## Where each exam topic lives

| Exam topic | Where to look | What to notice |
|---|---|---|
| **AI agent fundamentals**: agent vs. chatbot vs. workflow | `shop/agent.py`, demo step 5 | The model chooses its own steps at run time; a workflow would hard-code them |
| **LLM basics**: roles, system prompts, tokens, temperature, hallucination | `shop/llm.py`, `SPECIALISTS` in `shop/agents.py` | System prompts limit answers to tool results; demo step 14 declines instead of making things up |
| **ReAct / agent loop**, stop conditions | `Agent.run` in `shop/agent.py` | Thought → Action → Observation, `max_steps`, errors fed back as observations (step 4) |
| **Tools / function calling** | `shop/tools.py` | JSON-schema tools; the model *requests* a call and the runtime *executes* it; argument checks |
| **Tool security**, least privilege | `track_order`, `ToolContext` | The customer's identity comes from the login session, never from the model, so step 7 can't read Alex's order |
| **RAG**: chunking, embeddings, top-k, citations | `shop/vectors.py`, `search_policies` | Answers grounded in policy text with `[source: …]` |
| **Vector search / hybrid search** | `search_products`, `sql/schema.sql` | Similarity search plus a price filter (step 2); `VECTOR`, `VECTOR_DISTANCE`, HNSW/IVF, `FETCH APPROX` |
| **Memory**: short-term vs. long-term | `shop/memory.py` | Session trimming, summary and timeout vs. saved preferences; `/forget` deletes them (privacy) |
| **Personalisation** | `recommend_for_me` | Uses long-term memory and purchase history (step 13) |
| **Multi-agent orchestration** | `shop/agents.py` | Supervisor routes to specialists; one message can use two agents (step 9) |
| **MCP** | `mcp_server.py`, `mcp_client_demo.py` | `initialize`, `tools/list`, `tools/call`, `resources/*`; refunds are deliberately *not* exposed |
| **Guardrails / Responsible AI** | `shop/guardrails.py` | Blocks prompt injection (step 11), masks PII (step 12), limits input size, masks output |
| **Human-in-the-loop** | `request_refund` | Refunds over $100 wait for a manager's approval (step 10; `chat.py` asks you) |
| **Observability and evaluation** | `shop/trace.py`, `evals.py` | Every step is traced for audit; regression evals measure routing and tool accuracy |
| **OCI services** | `OCIGenAIChat`, comments in `tools.py` | OCI Generative AI (on-demand vs. dedicated), OCI GenAI Agents tool types (RAG, SQL, function, API), Select AI |

## Exercises (do these to really learn it)

1. Add a `cancel_order` tool for orders still `processing`, give it to the orders agent, and add an eval case.
2. Add a `wishlist` long-term memory and use it in `recommend_for_me`.
3. Make the supervisor send warranty claims for electronics to a new `warranty-agent`.
4. Add an output guardrail that blocks any reply promising a discount that wasn't applied by a tool.
5. Expose `checkout` over MCP, but require an approval step first. Why is that risky without one?
6. Run `sql/schema.sql` on Autonomous Database, then compare exact and `FETCH APPROX` results.
7. Swap `MockLLM` for `OCIGenAIChat` and run `evals.py`. Real models don't always choose tools the same way, which is why evals matter.

> Check the official exam topics on Oracle MyLearn and take Oracle's free course. Avoid
> "exam dump" sites: they break Oracle's candidate agreement and are often wrong.
