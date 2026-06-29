#!/usr/bin/env bash
# Sync Obsidian vault from iCloud into vault/, then commit and push to GitHub.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ICLOUD_VAULT="${ICLOUD_VAULT:-$HOME/Library/Mobile Documents/iCloud~md~obsidian/Documents/Worlding}"
TARGET_VAULT="$ROOT/vault"
COMMIT_MSG="${1:-Vault sync}"

if [[ ! -d "$ICLOUD_VAULT" ]]; then
  echo "Error: iCloud vault not found at:" >&2
  echo "  $ICLOUD_VAULT" >&2
  echo "Set ICLOUD_VAULT to override." >&2
  exit 1
fi

echo "Syncing iCloud vault → vault/"
mkdir -p "$TARGET_VAULT"
rsync -a --delete \
  --exclude '.obsidian/' \
  --exclude '.trash/' \
  --exclude '.git/' \
  "$ICLOUD_VAULT/" "$TARGET_VAULT/"

cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  echo "Building graph data and assets..."
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  python "$ROOT/process_notes.py"
else
  echo "Warning: .venv not found — run process_notes.py manually for images/graph." >&2
fi

if git diff --quiet && git diff --cached --quiet && [[ -z "$(git ls-files --others --exclude-standard)" ]]; then
  echo "No changes to commit."
  exit 0
fi

git add .
git commit -m "$COMMIT_MSG"
git push

echo "Done."
