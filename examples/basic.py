"""Run a tiny repeatable PyMC sample twice with a private disk cache."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pymc as pm

import pymc_cache as pc


def main() -> None:
    with pm.Model() as model:
        pm.Normal("mu", 0, 1)
    with tempfile.TemporaryDirectory(prefix="pymc-cache-example-") as directory:
        options = dict(model=model, draws=10, tune=10, chains=1, cores=1,
                       random_seed=42, progressbar=False, cache=Path(directory))
        first = pc.sample(**options)
        second = pc.sample(**options)
        assert first is not second
        assert len(pc.DiskCache(directory).entries()) == 1
        print("Reused", len(second.posterior.draw), "draws from disk")


if __name__ == "__main__":
    main()
