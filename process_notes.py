#!/usr/bin/env python3
"""Scan an Obsidian vault, embed notes, reduce to 3D, export graph JSON."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import umap
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_OUTPUT = Path("frontend/data.json")
DEFAULT_VAULT = Path("vault")
IGNORED_DIRS = {".obsidian", ".trash", ".git", ".cursor"}
WIKI_LINK_PATTERN = re.compile(
    r"\[\[([^\]|#^]+)(?:\|[^\]]*)?(?:#[^\]]*)?(?:\^[^\]]*)?\]\]"
)
FRONTMATTER_PATTERN = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
HEADING_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)
MAX_EMBED_CHARS = 8000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process Obsidian notes into a 3D semantic graph JSON."
    )
    parser.add_argument(
        "--vault",
        type=Path,
        default=DEFAULT_VAULT,
        help=f"Path to the Obsidian vault folder (default: {DEFAULT_VAULT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Sentence-transformers model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    return parser.parse_args()


def strip_frontmatter(text: str) -> str:
    return FRONTMATTER_PATTERN.sub("", text, count=1)


def extract_title(note_id: str, text: str) -> str:
    match = HEADING_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return note_id.replace("-", " ").replace("_", " ")


def clean_markdown(text: str) -> str:
    text = WIKI_LINK_PATTERN.sub(r"\1", text)
    text = re.sub(r"!\[\[[^\]]+\]\]", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    text = re.sub(r"`{1,3}[^`]+`{1,3}", " ", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~>|]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_key(value: str) -> str:
    return value.strip().lower()


def scan_vault(vault_path: Path) -> list[dict]:
    notes: list[dict] = []

    for md_path in sorted(vault_path.rglob("*.md")):
        if any(part in IGNORED_DIRS for part in md_path.parts):
            continue

        note_id = md_path.stem
        raw_text = md_path.read_text(encoding="utf-8", errors="replace")
        body = strip_frontmatter(raw_text)
        title = extract_title(note_id, body)
        content = clean_markdown(body)
        links = extract_wiki_links(raw_text)

        notes.append(
            {
                "id": note_id,
                "title": title,
                "body": body,
                "content": content,
                "links": links,
            }
        )

    return notes


def extract_wiki_links(text: str) -> list[str]:
    seen: set[str] = set()
    links: list[str] = []

    for match in WIKI_LINK_PATTERN.finditer(text):
        target = match.group(1).strip()
        key = normalize_key(target)
        if key and key not in seen:
            seen.add(key)
            links.append(target)

    return links


def build_lookup(notes: list[dict]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for note in notes:
        lookup[normalize_key(note["id"])] = note["id"]
        lookup[normalize_key(note["title"])] = note["id"]
    return lookup


def resolve_links(notes: list[dict], lookup: dict[str, str]) -> list[dict]:
    links: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for note in notes:
        source_id = note["id"]
        for target_title in note["links"]:
            target_id = lookup.get(normalize_key(target_title))
            if not target_id or target_id == source_id:
                continue

            edge = (source_id, target_id)
            if edge in seen:
                continue

            seen.add(edge)
            links.append({"source": source_id, "target": target_id})

    return links


def embed_notes(
    notes: list[dict], model_name: str, batch_size: int
) -> np.ndarray:
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [
        f"{note['title']}\n\n{note['content']}"[:MAX_EMBED_CHARS] for note in notes
    ]

    print(f"Generating embeddings for {len(texts)} notes...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings


def reduce_to_3d(embeddings: np.ndarray) -> np.ndarray:
    count = len(embeddings)

    if count == 0:
        return np.empty((0, 3))

    if count == 1:
        return np.array([[0.0, 0.0, 0.0]])

    if count == 2:
        return np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])

    if count == 3:
        return np.array(
            [
                [-0.5, -0.3, 0.0],
                [0.5, -0.3, 0.0],
                [0.0, 0.6, 0.0],
            ]
        )

    if count < 15:
        print(f"Using PCA fallback for {count} notes (UMAP needs more neighbors).")
        reducer = PCA(n_components=3, random_state=42)
        return reducer.fit_transform(embeddings)

    n_neighbors = min(15, count - 1)
    print(f"Running UMAP (3D) with n_neighbors={n_neighbors}...")
    reducer = umap.UMAP(
        n_components=3,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def normalize_coords(coords: np.ndarray, target_span: float = 100.0) -> np.ndarray:
    if len(coords) == 0:
        return coords

    centered = coords - coords.mean(axis=0)
    if len(centered) == 1:
        return centered

    span = float(np.max(centered.max(axis=0) - centered.min(axis=0)))
    if span == 0.0:
        return centered

    return centered * (target_span / span)


def build_output(
    notes: list[dict], coords: np.ndarray, links: list[dict], model_name: str
) -> dict:
    nodes = []
    for note, (x, y, z) in zip(notes, coords):
        nodes.append(
            {
                "id": note["id"],
                "title": note["title"],
                "body": note["body"],
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "z": round(float(z), 4),
            }
        )

    return {
        "nodes": nodes,
        "links": links,
        "meta": {
            "nodeCount": len(nodes),
            "linkCount": len(links),
            "model": model_name,
        },
    }


def main() -> int:
    args = parse_args()
    vault_path = args.vault.expanduser().resolve()

    if not vault_path.is_dir():
        print(f"Error: Vault path does not exist: {vault_path}", file=sys.stderr)
        return 1

    print(f"Scanning vault: {vault_path}")
    notes = scan_vault(vault_path)
    print(f"Found {len(notes)} markdown notes.")

    if not notes:
        print("No notes found. Writing empty graph.")
        output = build_output([], np.empty((0, 3)), [], args.model)
    else:
        lookup = build_lookup(notes)
        links = resolve_links(notes, lookup)
        print(f"Resolved {len(links)} wiki links.")

        embeddings = embed_notes(notes, args.model, args.batch_size)
        coords = normalize_coords(reduce_to_3d(embeddings))
        output = build_output(notes, coords, links, args.model)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Exported {output['meta']['nodeCount']} nodes and "
          f"{output['meta']['linkCount']} links to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
