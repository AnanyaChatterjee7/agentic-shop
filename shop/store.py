"""The e-commerce back end: catalog, customers, carts, orders, coupons, policies.

In production this would be Oracle Database tables (see sql/schema.sql). Agents never
touch this data directly. They go through TOOLS, which enforce the business rules.
"""
from __future__ import annotations

import copy
import itertools
from datetime import date

PRODUCTS = [
    {"sku": "SH-101", "name": "TrailRunner Waterproof Running Shoes", "category": "shoes", "price": 129.99, "stock": 12,
     "description": "Waterproof trail running shoe with grippy sole for rain and mud."},
    {"sku": "SH-102", "name": "CityLite Running Shoes", "category": "shoes", "price": 79.99, "stock": 30,
     "description": "Lightweight breathable road running shoe for daily jogging."},
    {"sku": "SH-103", "name": "Summit Hiking Boots", "category": "shoes", "price": 179.00, "stock": 5,
     "description": "Waterproof leather hiking boot with ankle support for mountains."},
    {"sku": "AP-201", "name": "StormShell Rain Jacket", "category": "apparel", "price": 99.50, "stock": 8,
     "description": "Packable waterproof rain jacket, windproof, for running and hiking."},
    {"sku": "AP-202", "name": "DryFit Running T-Shirt", "category": "apparel", "price": 24.99, "stock": 50,
     "description": "Quick-dry breathable sports t-shirt for running and gym."},
    {"sku": "AP-203", "name": "Merino Wool Socks (3 pack)", "category": "apparel", "price": 19.99, "stock": 40,
     "description": "Warm cushioned wool socks for hiking and running."},
    {"sku": "EL-301", "name": "PulseWatch GPS Smartwatch", "category": "electronics", "price": 249.00, "stock": 6,
     "description": "GPS running watch with heart rate monitor, waterproof, 7 day battery."},
    {"sku": "EL-302", "name": "BeatBuds Wireless Earbuds", "category": "electronics", "price": 59.99, "stock": 25,
     "description": "Sweat resistant wireless earbuds for running and workouts."},
    {"sku": "EL-303", "name": "PowerCore Portable Charger", "category": "electronics", "price": 34.99, "stock": 20,
     "description": "10000mAh power bank to charge phone and watch while travelling."},
    {"sku": "HM-401", "name": "Steel Insulated Water Bottle", "category": "accessories", "price": 21.00, "stock": 60,
     "description": "Keeps drinks cold for 24 hours, leak proof, for gym and hiking."},
    {"sku": "HM-402", "name": "Yoga Mat Pro", "category": "fitness", "price": 45.00, "stock": 15,
     "description": "Non slip cushioned yoga and exercise mat."},
    {"sku": "HM-403", "name": "Adjustable Dumbbell Set", "category": "fitness", "price": 149.00, "stock": 0,
     "description": "Adjustable weights 2 to 24 kg for home gym strength training."},
]

POLICIES = {
    "returns": "Returns: unused items in original packaging can be returned within 30 days of delivery for a "
               "full refund. Shoes must be unworn outdoors. Refunds go to the original payment method in 5 "
               "business days. Refunds above 100 USD need manager approval.",
    "shipping": "Shipping: standard delivery takes 3 to 5 business days and is free on orders over 75 USD, "
                "otherwise 6.99 USD. Express delivery takes 1 to 2 days and costs 14.99 USD.",
    "warranty": "Warranty: electronics have a 1 year manufacturer warranty covering defects. Shoes and "
                "apparel have a 90 day quality guarantee against stitching or sole defects.",
    "exchanges": "Exchanges: size exchanges are free within 30 days. Start an exchange from your orders page "
                 "or ask the support assistant.",
    "privacy": "Privacy: we never store full card numbers. Payment is handled by a PCI compliant provider. "
               "You can request deletion of your personal data at any time.",
}

COUPONS = {"WELCOME10": 0.10, "RUN20": 0.20}
FREE_SHIPPING_OVER = 75.0
SHIPPING_FEE = 6.99
MAX_QTY_PER_ITEM = 5
REFUND_APPROVAL_LIMIT = 100.0

CUSTOMERS = {
    "C001": {"name": "Priya", "email": "priya@example.com", "tier": "gold"},
    "C002": {"name": "Alex", "email": "alex@example.com", "tier": "standard"},
}

ORDERS = {
    "5001": {"customer": "C001", "items": {"AP-202": 2}, "total": 49.98, "status": "shipped",
             "date": "2026-11-22", "eta": "2026-11-28"},
    "5002": {"customer": "C001", "items": {"SH-101": 1}, "total": 129.99, "status": "delivered",
             "date": "2026-11-10", "eta": "2026-11-14"},
    "5003": {"customer": "C002", "items": {"EL-301": 1}, "total": 249.00, "status": "processing",
             "date": "2026-11-25", "eta": "2026-12-01"},
}


class Store:
    def __init__(self):
        self.products = {p["sku"]: copy.deepcopy(p) for p in PRODUCTS}
        self.customers = copy.deepcopy(CUSTOMERS)
        self.orders = copy.deepcopy(ORDERS)
        self.carts: dict[str, dict] = {}          # customer -> {"items": {sku: qty}, "coupon": str|None}
        self._ids = itertools.count(6001)

    # ---- cart
    def cart(self, customer: str) -> dict:
        return self.carts.setdefault(customer, {"items": {}, "coupon": None})

    def cart_totals(self, customer: str) -> dict:
        c = self.cart(customer)
        subtotal = sum(self.products[s]["price"] * q for s, q in c["items"].items())
        discount = round(subtotal * COUPONS.get(c["coupon"] or "", 0.0), 2)
        after = subtotal - discount
        shipping = 0.0 if after >= FREE_SHIPPING_OVER or not c["items"] else SHIPPING_FEE
        return {"subtotal": round(subtotal, 2), "discount": discount, "shipping": shipping,
                "total": round(after + shipping, 2)}

    def place_order(self, customer: str) -> dict:
        c = self.cart(customer)
        totals = self.cart_totals(customer)
        oid = str(next(self._ids))
        for sku, qty in c["items"].items():
            self.products[sku]["stock"] -= qty
        self.orders[oid] = {"customer": customer, "items": dict(c["items"]), "total": totals["total"],
                            "status": "processing", "date": date.today().isoformat(), "eta": "in 3-5 business days"}
        self.carts[customer] = {"items": {}, "coupon": None}
        return {"order_id": oid, **totals}
