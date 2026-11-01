#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv lock --check --no-config
uv run --frozen quartodoc build
quarto render
uv run --frozen python scripts/check_docs.py
uv run --frozen python scripts/check_examples.py
