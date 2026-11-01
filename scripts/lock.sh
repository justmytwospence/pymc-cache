#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv lock --check --no-config
uv run --frozen python -c 'import pymc_cache'
