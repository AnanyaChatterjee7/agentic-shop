"""Agent tools for the shop: the only way an agent can read or change store data.

Exam concepts:
- A tool = name + description + JSON-schema parameters + code. The model reads the
  description and schema to decide when and how to call it; the RUNTIME executes it.
- Identity comes from the authenticated session (ToolContext), NEVER from model arguments,
  so a prompt cannot make the agent act for another customer (least privilege).
- Tools enforce business rules (stock, quantity limits, ownership) even if the model is wrong.
- Risky actions (refunds above a limit) require human-in-the-loop approval.
- Tool types in OCI Generative AI Agents: RAG tool (search_policies), SQL tool
  (could replace search_products with NL2SQL), function-calling tools (cart, checkout),
  API endpoint tools (track_order could call a carrier API).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .store import COUPONS, MAX_QTY_PER_ITEM, POLICIES, REFUND_APPROVAL_LIMIT, Store
from .vectors import VectorIndex, build_policy_index, build_product_index


@dataclass
class ToolContext:
    customer_id: str
    store: Store
    products: VectorIndex
    policies: VectorIndex
    profiles: object = None                                   # LongTermMemory
    approve: Callable[[str], bool] = lambda request: False    # human-in-the-loop hook

    @classmethod
    def create(cls, customer_id: str, store: Store | None = None, **kw) -> "ToolContext":
        store = store or Store()
        return cls(customer_id, store, build_product_index(store), build_policy_index(POLICIES), **kw)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., str]

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _params(required: list[str] | None = None, **props) -> dict:
    return {"type": "object", "properties": {k: {"type": v} for k, v in props.items()},
            "required": required or []}


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {t.name: t for t in tools}

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def call(self, ctx: ToolContext, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool '{name}'"
        props, req = tool.parameters["properties"], tool.parameters["required"]
        for r in req:
            if r not in args:
                return f"ERROR: missing argument '{r}'"
        for k, v in args.items():
            if k not in props:
                return f"ERROR: unexpected argument '{k}'"
            t = props[k]["type"]
            if (t == "integer" and not isinstance(v, int)) or (t == "number" and not isinstance(v, (int, float))) \
                    or (t == "string" and not isinstance(v, str)):
                return f"ERROR: argument '{k}' must be {t}"
        try:
            return tool.func(ctx, **args)
        except Exception as exc:   # errors become observations the agent can react to
            return f"ERROR: {exc}"


# ------------------------------------------------------------------ tool functions
def _fmt(p: dict) -> str:
    stock = "in stock" if p["stock"] > 0 else "OUT OF STOCK"
    return f"{p['sku']} {p['name']} ${p['price']:.2f} ({stock})"


def search_products(ctx, query: str, max_price: float | None = None) -> str:
    where = (lambda m: m["price"] <= max_price) if max_price else None   # hybrid: vector + SQL-style filter
    hits = ctx.products.search(query, k=3, where=where)
    if not hits:
        return "No matching products found."
    return "; ".join(f"{_fmt(h['meta'])} [match {h['score']}]" for h in hits)


def resolve_product(ctx, product: str) -> dict | None:
    p = ctx.store.products.get(product.upper())
    if p:
        return p
    hits = ctx.products.search(product, k=1, min_score=0.3)
    return hits[0]["meta"] if hits else None


def add_to_cart(ctx, product: str, quantity: int = 1) -> str:
    p = resolve_product(ctx, product)
    if not p:
        return f"ERROR: no product matches '{product}'"
    cart = ctx.store.cart(ctx.customer_id)["items"]
    new_qty = cart.get(p["sku"], 0) + quantity
    if quantity < 1 or new_qty > MAX_QTY_PER_ITEM:
        return f"ERROR: quantity must be 1 to {MAX_QTY_PER_ITEM} per item"
    if new_qty > p["stock"]:
        return f"ERROR: only {p['stock']} of {p['name']} in stock"
    cart[p["sku"]] = new_qty
    return f"Added {quantity} x {p['name']} to cart"


def view_cart(ctx) -> str:
    c = ctx.store.cart(ctx.customer_id)
    if not c["items"]:
        return "Cart is empty"
    lines = [f"{q} x {ctx.store.products[s]['name']}" for s, q in c["items"].items()]
    t = ctx.store.cart_totals(ctx.customer_id)
    return (f"{', '.join(lines)} | subtotal ${t['subtotal']:.2f}, discount ${t['discount']:.2f}, "
            f"shipping ${t['shipping']:.2f}, total ${t['total']:.2f}")


def apply_coupon(ctx, code: str) -> str:
    code = code.upper()
    if code not in COUPONS:
        return f"ERROR: coupon {code} is not valid"
    ctx.store.cart(ctx.customer_id)["coupon"] = code
    return f"Coupon {code} applied ({int(COUPONS[code] * 100)}% off)"


def checkout(ctx) -> str:
    if not ctx.store.cart(ctx.customer_id)["items"]:
        return "ERROR: cart is empty"
    r = ctx.store.place_order(ctx.customer_id)
    return f"Order {r['order_id']} placed, total ${r['total']:.2f} (payment via secure provider)"


def track_order(ctx, order_id: str) -> str:
    o = ctx.store.orders.get(order_id)
    if not o or o["customer"] != ctx.customer_id:       # authorization check inside the tool
        return f"ERROR: order {order_id} not found for this customer"
    return f"Order {order_id}: {o['status']}, placed {o['date']}, ETA {o['eta']}, total ${o['total']:.2f}"


def request_refund(ctx, order_id: str, reason: str) -> str:
    o = ctx.store.orders.get(order_id)
    if not o or o["customer"] != ctx.customer_id:
        return f"ERROR: order {order_id} not found for this customer"
    if o["status"] == "refunded":
        return f"ERROR: order {order_id} was already refunded"
    if o["status"] != "delivered":
        return f"ERROR: order {order_id} is {o['status']}; only delivered orders can be refunded"
    if o["total"] > REFUND_APPROVAL_LIMIT:
        if not ctx.approve(f"Refund ${o['total']:.2f} for order {order_id} (reason: {reason})"):
            return f"Refund for order {order_id} was NOT approved by a manager; a support ticket was opened"
        note = " (manager approved)"
    else:
        note = ""
    o["status"] = "refunded"
    return f"Refund of ${o['total']:.2f} for order {order_id} issued{note}"


def search_policies(ctx, query: str) -> str:
    hits = ctx.policies.search(query, k=2, min_score=0.1)
    if not hits:
        return "No policy found. Say you don't know and offer a human agent."   # don't hallucinate
    return " | ".join(f"[source: {h['key']}] {h['text']}" for h in hits)


def save_preference(ctx, key: str, value: str) -> str:
    ctx.profiles.remember(ctx.customer_id, key, value)
    return f"Saved preference {key} = {value}"


def recommend_for_me(ctx) -> str:
    prefs = ctx.profiles.get(ctx.customer_id) if ctx.profiles else {}
    bought = {s for o in ctx.store.orders.values() if o["customer"] == ctx.customer_id for s in o["items"]}
    interest = " ".join(prefs.values()) or "running"
    hits = [h for h in ctx.products.search(interest, k=6, min_score=0.05,
                                            where=lambda m: m["stock"] > 0)
            if h["key"] not in bought][:3]
    based_on = ", ".join(f"{k}={v}" for k, v in prefs.items()) or "popular items"
    return f"Based on {based_on}: " + "; ".join(_fmt(h["meta"]) for h in hits)


ALL_TOOLS = {t.name: t for t in [
    Tool("search_products", "Semantic search of the product catalog. Optional max_price filter.",
         _params(["query"], query="string", max_price="number"), search_products),
    Tool("add_to_cart", "Add a product (SKU or product name) to the customer's cart.",
         _params(["product"], product="string", quantity="integer"), add_to_cart),
    Tool("view_cart", "Show cart contents and totals.", _params(), view_cart),
    Tool("apply_coupon", "Apply a discount coupon code to the cart.", _params(["code"], code="string"), apply_coupon),
    Tool("checkout", "Place an order for everything in the cart.", _params(), checkout),
    Tool("track_order", "Get status and delivery ETA of one of the customer's orders.",
         _params(["order_id"], order_id="string"), track_order),
    Tool("request_refund", "Refund a delivered order. Refunds above $100 need manager approval.",
         _params(["order_id", "reason"], order_id="string", reason="string"), request_refund),
    Tool("search_policies", "Search store policy documents: returns, shipping, warranty, exchanges, privacy.",
         _params(["query"], query="string"), search_policies),
    Tool("save_preference", "Remember a customer preference (e.g. shoe_size, interest) for future sessions.",
         _params(["key", "value"], key="string", value="string"), save_preference),
    Tool("recommend_for_me", "Personalised product recommendations from saved preferences and history.",
         _params(), recommend_for_me),
]}


def registry(*names: str) -> ToolRegistry:
    return ToolRegistry([ALL_TOOLS[n] for n in (names or ALL_TOOLS)])
