# Obsidian Zettelkasten – 3D-Semantik-Graph

Liest einen Obsidian-Vault ein, erzeugt semantische Embeddings, reduziert sie mit UMAP auf 3D-Koordinaten und visualisiert Notizen plus Wiki-Links als interaktiven 3D-Graphen im Browser — inklusive RAG-Chatbot.

## Projektstruktur

```
.
├── process_notes.py        # Vault scannen, embedden, data.json + rag_index.json
├── dev_server.py           # Lokaler Dev-Server (Frontend + Chat-API)
├── requirements.txt        # Build-Abhängigkeiten
├── requirements-dev.txt    # Dev-Server (FastAPI, uvicorn)
├── .env.example            # OPENAI_API_KEY etc.
├── backend/                # RAG-Logik (shared)
│   ├── config.py
│   ├── embeddings.py
│   └── chat.py
├── api/
│   ├── chat.py             # Vercel Serverless Function
│   └── requirements.txt    # Runtime-Deps für Vercel
├── vercel.json             # Standalone-Deploy
├── vercel.json.example     # Template für Parent-Website (Submodule)
├── sync-and-push.sh
├── vault/
└── frontend/
    ├── index.html          # 3D-Visualisierung + Chat-Widget
    ├── chat.css
    ├── chat.js
    ├── data.json           # generiert: Graph + Notiz-Inhalte
    ├── rag_index.json      # generiert: OpenAI-Vektoren für RAG
    └── assets/
```

## Voraussetzungen

- Python 3.10+
- Ein lokaler Obsidian-Vault mit `.md`-Dateien
- OpenAI API-Key (für RAG-Index und Chat)

## 1. Python-Setup

```bash
cd "/Users/philippe/vb database visualisation"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional, für dev_server.py
cp .env.example .env                 # OPENAI_API_KEY eintragen
```

Beim ersten Lauf lädt `sentence-transformers` das Modell `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB) automatisch herunter.

## 2. Notizen verarbeiten

```bash
python process_notes.py
```

Der Vault liegt standardmäßig in `vault/`. Alternativ ein anderer Pfad:

```bash
python process_notes.py --vault "/pfad/zu/deinem/obsidian-vault"
```

Optionale Parameter:

| Parameter | Standard | Beschreibung |
|-----------|----------|--------------|
| `--output` | `frontend/data.json` | Zielpfad für die JSON-Datei |
| `--rag-output` | `frontend/rag_index.json` | RAG-Vektorindex für Chatbot |
| `--assets` | `frontend/assets` | Zielordner für Anhänge (Bilder, PDFs) |
| `--model` | `paraphrase-multilingual-MiniLM-L12-v2` | Lokales Modell für 3D-UMAP |
| `--rag-model` | `text-embedding-3-small` | OpenAI-Modell für RAG-Index |
| `--skip-rag-index` | — | RAG-Index überspringen (ohne API-Key) |
| `--batch-size` | `32` | Batch-Größe für Embeddings |

Beispiel mit allen Optionen:

```bash
python process_notes.py \
  --vault ~/Documents/Obsidian/Vault \
  --output frontend/data.json \
  --model paraphrase-multilingual-MiniLM-L12-v2 \
  --batch-size 32
```

### Ausgabeformat (`data.json`)

```json
{
  "nodes": [
    {
      "id": "Meine-Notiz",
      "title": "Meine Notiz",
      "body": "# Meine Notiz\n\nMarkdown-Inhalt…",
      "x": 1.23,
      "y": -0.45,
      "z": 0.78
    }
  ],
  "links": [
    { "source": "Notiz-A", "target": "Notiz-B" }
  ],
  "meta": {
    "nodeCount": 42,
    "linkCount": 87,
    "model": "paraphrase-multilingual-MiniLM-L12-v2"
  }
}
```

- **Nodes**: Position aus UMAP (semantische Nähe), plus `body` mit dem Markdown-Inhalt der Notiz
- **Links**: Obsidian-Wiki-Links (`[[Notiz-Titel]]`)

### RAG-Index (`rag_index.json`)

Wird parallel zu `data.json` erzeugt (OpenAI `text-embedding-3-small`):

```json
[
  {
    "id": "Meine-Notiz",
    "title": "Meine Notiz",
    "vector": [0.012, -0.034, ...]
  }
]
```

## 3. Lokal starten (mit Chatbot)

```bash
source .venv/bin/activate
python dev_server.py
```

Browser: [http://localhost:8080](http://localhost:8080)

Der Dev-Server liefert das Frontend und die Chat-API unter `/api/chat`.

### Nur Frontend (ohne Chat)

```bash
cd frontend
python3 -m http.server 8080
```

### Bedienung

- **Maus ziehen**: Kamera drehen
- **Scrollrad**: Zoomen
- **Rechtsklick + ziehen**: Schwenken
- **Hover über Punkt**: Verbundene Links hervorheben
- **Klick auf Punkt**: Notiz im Overlay öffnen (formatiertes Markdown)
- **Wiki-Links im Overlay**: Klick auf `[[verlinkte Notiz]]` öffnet die Zielnotiz
- **Overlay schließen**: ×-Button, Klick auf den Hintergrund oder `Escape`
- **Chat-Button (unten rechts)**: Fragen zu den Notizen stellen
- **Quellen im Chat**: Klick auf Quellen-Zitat öffnet die Notiz und hebt Knoten im Graph hervor

## Deployment als Git-Submodule (Vercel)

Dieses Repo ist als Submodule in deine Website integrierbar:

```bash
# Im Parent-Website-Repo:
git submodule add <repo-url> zettelkasten
git submodule update --init --recursive
```

### Vercel Environment Variables

Im Vercel-Dashboard setzen:

| Variable | Zweck |
|---|---|
| `OPENAI_API_KEY` | Query-Embedding + Chat (gpt-4o-mini) |
| `ZETTELKASTEN_NAME` | Name im System-Prompt (optional) |

### Parent `vercel.json`

Siehe [`vercel.json.example`](vercel.json.example) — Rewrites für `/zettelkasten/*` und Function-Timeout für `api/chat.py`.

Im Submodule-Frontend ggf. API-Pfad setzen:

```html
<meta name="zettelkasten-api" content="/api/chat">
```

### Nach Vault-Update

```bash
source .venv/bin/activate
python process_notes.py
git add frontend/data.json frontend/rag_index.json
git commit -m "Update Zettelkasten index"
git push
# Parent-Repo: Submodule-Pointer updaten → Vercel redeployed
```

## Hinweise

- **Semantische Nähe ≠ Link-Nähe**: UMAP platziert inhaltlich ähnliche Notizen nah beieinander. Obsidian-Links werden separat als Linien gezeichnet.
- **Kleine Vaults** (< 15 Notizen): Es wird automatisch PCA statt UMAP verwendet.
- **Große Vaults** (> 500 Notizen): Embedding und UMAP können einige Minuten dauern.
- **Wiki-Links**: Unterstützt `[[Titel]]`, `[[Titel|Alias]]`, `[[Titel#Abschnitt]]`, `[[Titel^Block]]`. Links zu nicht existierenden Notizen werden ignoriert.

## iCloud-Vault syncen und pushen

Wenn du in Obsidian weiter den **iCloud-Vault** nutzt, bringt ein Skript die Änderungen ins Repo und auf GitHub:

```bash
cd "/Users/philippe/vb database visualisation"
./sync-and-push.sh
```

Optional mit eigener Commit-Nachricht:

```bash
./sync-and-push.sh "Neue Notizen zu Worlding"
```

Anderer iCloud-Pfad:

```bash
ICLOUD_VAULT="/pfad/zum/vault" ./sync-and-push.sh
```

Das Skript kopiert den iCloud-Vault nach `vault/`, baut `data.json` und `frontend/assets/`, committet und pusht.

Ohne iCloud (Vault liegt schon in `vault/`):

```bash
source .venv/bin/activate
python process_notes.py
python dev_server.py
```
