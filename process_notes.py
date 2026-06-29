#!/usr/bin/env python3
"""Scan an Obsidian vault, embed notes, reduce to 3D, export graph JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import umap
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_RAG_MODEL = "text-embedding-3-small"
DEFAULT_OUTPUT = Path("frontend/data.json")
DEFAULT_RAG_OUTPUT = Path("frontend/rag_index.json")
DEFAULT_ASSETS = Path("frontend/assets")
DEFAULT_VAULT = Path("vault")
IGNORED_DIRS = {".obsidian", ".trash", ".git", ".cursor"}
WIKI_LINK_PATTERN = re.compile(
    r"(?<!!)\[\[([^\]|#^]+)(?:\|([^\]]*))?(?:#[^\]]*)?(?:\^[^\]]*)?\]\]"
)
WIKI_EMBED_PATTERN = re.compile(
    r"!\[\[([^\]|#^]+)(?:\|([^\]]*))?(?:#[^\]]*)?(?:\^[^\]]*)?\]\]"
)
FILE_URL_PATTERN = re.compile(r"file://[^\s)\]\"'<>]+", re.IGNORECASE)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
ASSET_EXTENSIONS = IMAGE_EXTENSIONS | {".pdf", ".mp4", ".mov", ".webm"}
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
        "--assets",
        type=Path,
        default=DEFAULT_ASSETS,
        help=f"Output folder for vault attachments (default: {DEFAULT_ASSETS})",
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
    parser.add_argument(
        "--rag-output",
        type=Path,
        default=DEFAULT_RAG_OUTPUT,
        help=f"RAG index JSON path (default: {DEFAULT_RAG_OUTPUT})",
    )
    parser.add_argument(
        "--rag-model",
        default=DEFAULT_RAG_MODEL,
        help=f"OpenAI embedding model for RAG (default: {DEFAULT_RAG_MODEL})",
    )
    parser.add_argument(
        "--skip-rag-index",
        action="store_true",
        help="Skip OpenAI RAG index generation",
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


def safe_asset_filename(rel: Path, used: dict[str, Path]) -> str:
    suffix = rel.suffix.lower()
    base = re.sub(r"[^a-zA-Z0-9._-]+", "-", rel.stem).strip("-")
    if not base:
        base = "asset"

    name = f"{base}{suffix}"
    counter = 2
    while name in used and used[name] != rel:
        name = f"{base}-{counter}{suffix}"
        counter += 1

    used[name] = rel
    return name


def export_assets(
    vault_path: Path, assets_dir: Path, file_index: dict[str, list[Path]]
) -> dict[Path, str]:
    if assets_dir.exists():
        shutil.rmtree(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    asset_map: dict[Path, str] = {}
    used_names: dict[str, Path] = {}
    for rel_paths in file_index.values():
        for rel in rel_paths:
            if rel in asset_map:
                continue
            filename = safe_asset_filename(rel, used_names)
            shutil.copy2(vault_path / rel, assets_dir / filename)
            asset_map[rel] = f"assets/{filename}"

    return asset_map


def asset_url(rel_path: Path, asset_map: dict[Path, str]) -> str:
    return asset_map.get(rel_path, f"assets/{rel_path.name}")


def build_file_index(vault_path: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}

    for file_path in vault_path.rglob("*"):
        if not file_path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in file_path.parts):
            continue
        if file_path.name.startswith("."):
            continue

        rel = file_path.relative_to(vault_path)
        if rel.suffix.lower() == ".md":
            continue
        if rel.suffix.lower() not in ASSET_EXTENSIONS:
            continue

        key = normalize_key(file_path.name)
        index.setdefault(key, []).append(rel)

    return index


def resolve_vault_file(
    target: str,
    note_rel_dir: Path,
    vault_path: Path,
    file_index: dict[str, list[Path]],
) -> Path | None:
    target = target.strip().replace("\\", "/")
    if not target:
        return None

    direct = vault_path / target
    if direct.is_file():
        return direct.relative_to(vault_path)

    relative_to_note = (vault_path / note_rel_dir / target).resolve()
    if relative_to_note.is_file() and vault_path in relative_to_note.parents:
        return relative_to_note.relative_to(vault_path)

    basename = Path(target).name
    matches = file_index.get(normalize_key(basename), [])
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]

    note_dir = Path(note_rel_dir)
    for rel in matches:
        if rel.parent == note_dir:
            return rel
    for rel in matches:
        if str(note_dir) in str(rel.parent):
            return rel
    return matches[0]


def rewrite_file_urls(
    text: str,
    file_index: dict[str, list[Path]],
    asset_map: dict[Path, str],
) -> str:
    def replacer(match: re.Match[str]) -> str:
        decoded_path = unquote(urlparse(match.group(0)).path)
        basename = Path(decoded_path).name
        matches = file_index.get(normalize_key(basename), [])
        if matches:
            return asset_url(matches[0], asset_map)
        return match.group(0)

    return FILE_URL_PATTERN.sub(replacer, text)


def rewrite_body_for_web(
    body: str,
    note_md_rel: Path,
    vault_path: Path,
    file_index: dict[str, list[Path]],
    note_lookup: dict[str, str],
    asset_map: dict[Path, str],
) -> str:
    note_rel_dir = note_md_rel.parent

    def embed_replacer(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        alias = (match.group(2) or target).strip()
        rel = resolve_vault_file(target, note_rel_dir, vault_path, file_index)
        if rel:
            return f"![{alias}]({asset_url(rel, asset_map)})"
        return f'<span class="note-missing">{alias}</span>'

    def wiki_replacer(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        alias = (match.group(2) or target).strip()
        note_id = note_lookup.get(normalize_key(target))
        if not note_id:
            note_id = note_lookup.get(normalize_key(Path(target).stem))
        if note_id:
            return f"[{alias}](#wiki:{quote(note_id, safe='')})"

        rel = resolve_vault_file(target, note_rel_dir, vault_path, file_index)
        if rel:
            if rel.suffix.lower() in IMAGE_EXTENSIONS:
                return f"![{alias}]({asset_url(rel, asset_map)})"
            return f"[{alias}]({asset_url(rel, asset_map)})"
        return f'<span class="note-missing">{alias}</span>'

    text = WIKI_EMBED_PATTERN.sub(embed_replacer, body)
    text = WIKI_LINK_PATTERN.sub(wiki_replacer, text)
    return rewrite_file_urls(text, file_index, asset_map)


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
                "md_path": md_path.relative_to(vault_path),
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


def embed_texts_for_rag(
    notes: list[dict], model_name: str
) -> list[dict]:
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Warning: OPENAI_API_KEY not set — skipping RAG index. "
            "Set the key or use --skip-rag-index.",
            file=sys.stderr,
        )
        return []

    client = OpenAI(api_key=api_key)
    texts = [
        f"{note['title']}\n\n{note['content']}"[:MAX_EMBED_CHARS] for note in notes
    ]

    print(f"Generating OpenAI RAG embeddings for {len(texts)} notes...")
    response = client.embeddings.create(input=texts, model=model_name)
    ordered = sorted(response.data, key=lambda item: item.index)

    return [
        {
            "id": note["id"],
            "title": note["title"],
            "vector": item.embedding,
        }
        for note, item in zip(notes, ordered)
    ]


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
    file_index = build_file_index(vault_path)
    rag_index: list[dict] = []

    if not notes:
        print("No notes found. Writing empty graph.")
        output = build_output([], np.empty((0, 3)), [], args.model)
    else:
        lookup = build_lookup(notes)
        assets_dir = args.assets.expanduser().resolve()
        asset_map = export_assets(vault_path, assets_dir, file_index)
        print(f"Exported {len(asset_map)} attachments to {assets_dir}.")

        for note in notes:
            note["body"] = rewrite_body_for_web(
                note["body"],
                note["md_path"],
                vault_path,
                file_index,
                lookup,
                asset_map,
            )

        links = resolve_links(notes, lookup)
        print(f"Resolved {len(links)} wiki links.")

        embeddings = embed_notes(notes, args.model, args.batch_size)
        coords = normalize_coords(reduce_to_3d(embeddings))
        output = build_output(notes, coords, links, args.model)

        if not args.skip_rag_index:
            rag_index = embed_texts_for_rag(notes, args.rag_model)

    if not notes:
        assets_dir = args.assets.expanduser().resolve()
        if assets_dir.exists():
            shutil.rmtree(assets_dir)
        assets_dir.mkdir(parents=True, exist_ok=True)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Exported {output['meta']['nodeCount']} nodes and "
          f"{output['meta']['linkCount']} links to {output_path}")

    rag_output_path = args.rag_output.expanduser().resolve()
    rag_output_path.parent.mkdir(parents=True, exist_ok=True)
    rag_output_path.write_text(
        json.dumps(rag_index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Exported {len(rag_index)} RAG vectors to {rag_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
