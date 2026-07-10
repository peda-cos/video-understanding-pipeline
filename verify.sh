#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

for tool in uv ffmpeg ffprobe bun; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool not found: $tool" >&2
    exit 1
  fi
done

uv sync --frozen
mkdir -p .local/web
(
  cd web
  bun install --frozen-lockfile
  bun run typecheck
  bun run build
)
uv run python -m unittest discover -s tests -v
