#!/usr/bin/env python3
"""Local dev server: serves frontend/ and /api/chat with SSE streaming."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from backend.chat import format_sse, stream_chat

app = FastAPI()


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    message = (body.get("message") or "").strip()
    if not message:
        return {"error": "Nachricht darf nicht leer sein."}

    client_ip = request.client.host if request.client else "unknown"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    def generate():
        try:
            for event in stream_chat(message, client_ip=client_ip):
                yield format_sse(event)
        except PermissionError as exc:
            yield format_sse({"type": "error", "content": str(exc)})
        except Exception as exc:
            yield format_sse({"type": "error", "content": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")


app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
