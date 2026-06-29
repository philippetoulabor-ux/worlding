from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.chat import format_sse, stream_chat


class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_POST(self) -> None:
        client_ip = self.headers.get("X-Forwarded-For", self.client_address[0])
        if client_ip and "," in client_ip:
            client_ip = client_ip.split(",")[0].strip()

        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            message = (body.get("message") or "").strip()
            if not message:
                self._json_error(400, "Nachricht darf nicht leer sein.")
                return
            if len(message) > 2000:
                self._json_error(400, "Nachricht zu lang (max. 2000 Zeichen).")
                return

            self.send_response(200)
            self._cors_headers()
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            for event in stream_chat(message, client_ip=client_ip):
                self.wfile.write(format_sse(event).encode("utf-8"))
                self.wfile.flush()

        except PermissionError as exc:
            self._json_error(429, str(exc))
        except FileNotFoundError as exc:
            self._json_error(503, str(exc))
        except RuntimeError as exc:
            self._json_error(500, str(exc))
        except Exception:
            self._json_error(500, "Interner Serverfehler.")

    def _cors_headers(self) -> None:
        origin = self.headers.get("Origin", "*")
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_error(self, code: int, message: str) -> None:
        self.send_response(code)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format: str, *args) -> None:
        pass
