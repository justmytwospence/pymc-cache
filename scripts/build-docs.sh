#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
quarto render
uv run --locked python scripts/check_docs.py
