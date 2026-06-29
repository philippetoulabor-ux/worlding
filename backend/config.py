from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
RAG_INDEX_PATH = FRONTEND_DIR / "rag_index.json"
DATA_JSON_PATH = FRONTEND_DIR / "data.json"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
MAX_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "2000"))
ZETTELKASTEN_NAME = os.environ.get("ZETTELKASTEN_NAME", "Worlding")
RATE_LIMIT = int(os.environ.get("RATE_LIMIT", "10"))
RATE_WINDOW_SECONDS = int(os.environ.get("RATE_WINDOW_SECONDS", "60"))

SYSTEM_PROMPT = f"""Du bist ein Assistent für den Zettelkasten „{ZETTELKASTEN_NAME}".
Antworte NUR auf Basis der bereitgestellten Notiz-Ausschnitte.
Wenn die Notizen die Frage nicht beantworten, sage das ehrlich — erfinde nichts dazu.
Zitiere verwendete Notizen mit ihrem Titel in eckigen Klammern, z.B. [Meine Notiz].
Antworte in der Sprache der Frage."""
