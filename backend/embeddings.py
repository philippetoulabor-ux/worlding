from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

from openai import OpenAI

from backend.config import (
    DATA_JSON_PATH,
    EMBEDDING_MODEL,
    MAX_CONTEXT_CHARS,
    OPENAI_API_KEY,
    RAG_INDEX_PATH,
    TOP_K,
)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY ist nicht gesetzt.")
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


@lru_cache(maxsize=1)
def load_rag_index() -> list[dict[str, Any]]:
    if not RAG_INDEX_PATH.is_file():
        raise FileNotFoundError(
            f"RAG-Index nicht gefunden: {RAG_INDEX_PATH}. "
            "Bitte process_notes.py ausführen."
        )
    return json.loads(RAG_INDEX_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_note_bodies() -> dict[str, str]:
    if not DATA_JSON_PATH.is_file():
        return {}
    data = json.loads(DATA_JSON_PATH.read_text(encoding="utf-8"))
    return {node["id"]: node.get("body", "") for node in data.get("nodes", [])}


def embed_query(text: str) -> list[float]:
    response = get_client().embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    k = top_k or TOP_K
    index = load_rag_index()
    if not index:
        return []

    query_vector = embed_query(query)
    bodies = load_note_bodies()

    scored = []
    for entry in index:
        score = cosine_similarity(query_vector, entry["vector"])
        note_id = entry["id"]
        body = bodies.get(note_id, "")
        scored.append(
            {
                "id": note_id,
                "title": entry["title"],
                "score": score,
                "body": body[:MAX_CONTEXT_CHARS],
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:k]
