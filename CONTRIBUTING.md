# Contributing

Install [uv](https://docs.astral.sh/uv/) and Quarto. Run `bash scripts/lock.sh`, `uv sync --locked --all-groups`, `uv run pytest -q --cov=pymc_cache --cov-fail-under=85`, `uv run ruff check src tests scripts`, `bash scripts/build-docs.sh` and `uv build --no-sources`. Test both supported PyMC/ArviZ majors before proposing a release. Make one conventional commit per logical change; discuss new cache formats and identity semantics before changing them. Do not check in posterior data.
