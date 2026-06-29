from __future__ import annotations

import json
from collections import defaultdict
from time import time
from typing import Any, Generator

from backend.config import OPENAI_MODEL, RATE_LIMIT, RATE_WINDOW_SECONDS, SYSTEM_PROMPT
from backend.embeddings import get_client, search

_rate_buckets: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(client_ip: str) -> None:
    now = time()
    window_start = now - RATE_WINDOW_SECONDS
    bucket = _rate_buckets[client_ip]
    _rate_buckets[client_ip] = [t for t in bucket if t > window_start]
    if len(_rate_buckets[client_ip]) >= RATE_LIMIT:
        raise PermissionError("Zu viele Anfragen. Bitte kurz warten.")
    _rate_buckets[client_ip].append(now)


def build_context(results: list[dict[str, Any]]) -> str:
    if not results:
        return "Keine relevanten Notizen gefunden."

    blocks = []
    for item in results:
        blocks.append(
            f'--- Notiz: "{item["title"]}" ---\n{item["body"] or "(Kein Inhalt)"}'
        )
    return "\n\n".join(blocks)


def stream_chat(message: str, client_ip: str = "unknown") -> Generator[dict[str, Any], None, None]:
    check_rate_limit(client_ip)

    results = search(message)
    context = build_context(results)
    source_ids = [item["id"] for item in results]

    user_content = f"""Kontext aus dem Zettelkasten:

{context}

---

Frage: {message}"""

    client = get_client()
    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        stream=True,
        temperature=0.2,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield {"type": "token", "content": delta}

    yield {"type": "sources", "ids": source_ids, "titles": [r["title"] for r in results]}


def format_sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
