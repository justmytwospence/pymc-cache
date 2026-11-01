"""Use a preflight lookup to skip sampler provisioning on a hit."""
from __future__ import annotations

import pymc as pm

import pymc_cache as pc


def main() -> None:
    with pm.Model() as model:
        pm.Normal("mu", 0, 1)
    session = pc.CacheSession(model=model, cache=pc.MemoryCache())
    kwargs = {"draws": 10, "tune": 10, "chains": 1, "cores": 1, "random_seed": 42,
              "progressbar": False}
    for _ in range(2):
        lookup = session.lookup(kwargs)
        if lookup.hit:
            trace = lookup.result
        else:
            trace = pm.sample(model=model, **lookup.normalized_kwargs)
            assert session.store(lookup, trace)
        print("Cache hit:", lookup.hit, "draws:", len(trace.posterior.draw))


if __name__ == "__main__":
    main()
