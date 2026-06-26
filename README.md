# Obsidian Zettelkasten – 3D-Semantik-Graph

Liest einen Obsidian-Vault ein, erzeugt semantische Embeddings, reduziert sie mit UMAP auf 3D-Koordinaten und visualisiert Notizen plus Wiki-Links als interaktiven 3D-Graphen im Browser.

## Projektstruktur

```
.
├── process_notes.py      # Vault scannen, embedden, data.json exportieren
├── requirements.txt
├── vault/                # Obsidian-Vault (Markdown + Anhänge)
├── frontend/
│   ├── index.html        # 3D-Visualisierung
│   └── data.json         # generiert (nicht committen)
└── README.md
```

## Voraussetzungen

- Python 3.10+
- Ein lokaler Obsidian-Vault mit `.md`-Dateien

## 1. Python-Setup

```bash
cd "/Users/philippe/vb database visualisation"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
| `--model` | `paraphrase-multilingual-MiniLM-L12-v2` | Embedding-Modell |
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

## 3. Frontend starten

Der Browser blockiert `fetch()` bei `file://`-URLs. Starte daher einen lokalen HTTP-Server:

```bash
cd frontend
python3 -m http.server 8080
```

Dann im Browser öffnen: [http://localhost:8080](http://localhost:8080)

### Bedienung

- **Maus ziehen**: Kamera drehen
- **Scrollrad**: Zoomen
- **Rechtsklick + ziehen**: Schwenken
- **Hover über Punkt**: Verbundene Links hervorheben
- **Klick auf Punkt**: Notiz im Overlay öffnen (formatiertes Markdown)
- **Wiki-Links im Overlay**: Klick auf `[[verlinkte Notiz]]` öffnet die Zielnotiz
- **Overlay schließen**: ×-Button, Klick auf den Hintergrund oder `Escape`

## Hinweise

- **Semantische Nähe ≠ Link-Nähe**: UMAP platziert inhaltlich ähnliche Notizen nah beieinander. Obsidian-Links werden separat als Linien gezeichnet.
- **Kleine Vaults** (< 15 Notizen): Es wird automatisch PCA statt UMAP verwendet.
- **Große Vaults** (> 500 Notizen): Embedding und UMAP können einige Minuten dauern.
- **Wiki-Links**: Unterstützt `[[Titel]]`, `[[Titel|Alias]]`, `[[Titel#Abschnitt]]`, `[[Titel^Block]]`. Links zu nicht existierenden Notizen werden ignoriert.
