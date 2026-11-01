"""Implement a minimal copy-isolated backend without a storage dependency."""
from __future__ import annotations

import copy

import pymc as pm

import pymc_cache as pc


class PrivateBackend:
    def __init__(self):
        self.entries = {}

    def load(self, key, **kwargs):
        value = self.entries.get(key)
        return copy.deepcopy(value) if value is not None else None

    def save(self, key, idata, **kwargs):
        self.entries[key] = copy.deepcopy(idata)


def main():
    with pm.Model() as model:
        pm.Normal("x")
    backend = PrivateBackend()
    options = dict(model=model, cache=backend, draws=2, tune=2, chains=1,
                   cores=1, random_seed=42, progressbar=False)
    first = pc.sample(**options)
    second = pc.sample(**options)
    assert first is not second and len(backend.entries) == 1
    print("Custom backend reused", len(second.posterior.draw), "draws")


if __name__ == "__main__":
    main()
