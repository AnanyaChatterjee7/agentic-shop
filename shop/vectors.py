"""Semantic search (RAG) over products and policy documents.

embed() is a toy offline embedding: hashed bag-of-words plus a small synonym map, so
"sneakers" finds "shoes". Real systems use an embedding model (OCI Generative AI Cohere
embed, or an ONNX model loaded into Oracle Database 23ai with VECTOR_EMBEDDING).

Exam concepts: chunking, embeddings, cosine similarity, top-k, similarity threshold,
hybrid search (vector similarity + relational filters such as price/stock), grounding + citations.
"""
from __future__ import annotations

import hashlib
import math
import re

DIM = 2048
STOP = {"the", "a", "an", "is", "are", "of", "to", "for", "and", "in", "on", "my", "what", "how", "do",
        "i", "can", "it", "be", "with", "me", "find", "show", "some", "any", "under", "want", "need",
        "looking", "you", "your", "have", "something", "good", "please", "usd", "price"}
SYNONYMS = {"sneaker": "shoe", "trainer": "shoe", "boot": "shoe", "footwear": "shoe", "jog": "running",
            "jogging": "running", "run": "running", "rain": "waterproof", "wet": "waterproof",
            "headphone": "earbud", "earphone": "earbud", "tee": "t-shirt", "shirt": "t-shirt",
            "refund": "return", "money": "refund", "delivery": "shipping", "ship": "shipping",
            "guarantee": "warranty", "broken": "defect", "faulty": "defect", "damaged": "defect",
            "workout": "gym", "exercise": "gym", "trek": "hiking", "hike": "hiking"}


def _tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9\-]+", text.lower()):
        if w in STOP or w.isdigit():
            continue
        if w.endswith("s") and len(w) > 3 and not w.endswith("ss"):
            w = w[:-1]
        out.append(SYNONYMS.get(w, w))
    return out


def embed(text: str) -> list[float]:
    vec = [0.0] * DIM
    for w in _tokens(text):
        vec[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


def chunk(text: str, size: int = 30, overlap: int = 8) -> list[str]:
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]


class VectorIndex:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, key: str, text: str, meta: dict | None = None):
        for piece in chunk(text):
            self.rows.append({"key": key, "text": piece, "meta": meta or {}, "vec": embed(piece)})

    def search(self, query: str, k: int = 3, min_score: float = 0.15, where=None) -> list[dict]:
        """`where` is a relational filter applied together with similarity, i.e. hybrid search."""
        q = embed(query)
        best: dict[str, dict] = {}
        for r in self.rows:
            if where and not where(r["meta"]):
                continue
            s = cosine(q, r["vec"])
            if s >= min_score and (r["key"] not in best or s > best[r["key"]]["score"]):
                best[r["key"]] = {**r, "score": round(s, 3)}
        return sorted(best.values(), key=lambda r: r["score"], reverse=True)[:k]


def build_product_index(store) -> VectorIndex:
    idx = VectorIndex()
    for p in store.products.values():
        idx.add(p["sku"], f"{p['name']} {p['category']} {p['description']}", p)   # meta = live product row
    return idx


def build_policy_index(policies: dict) -> VectorIndex:
    idx = VectorIndex()
    for name, text in policies.items():
        idx.add(name, text)
    return idx
