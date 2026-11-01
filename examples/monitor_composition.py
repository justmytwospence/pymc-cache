"""Preflight an optional monitor runner before starting its work.

The callable is injected: pymc-cache does not import or install pymc-monitor.
Only pass a stable sampler_identity/versions for a runner whose semantics you own.
"""
from __future__ import annotations

import pymc as pm

import pymc_cache as pc
from pymc_cache.policy import report_from_idata


def run_with_monitor(model, monitor_run, *, cache, sampler_identity, versions, **options):
    """Reuse a result or call an external monitor; never store unverified adaptive runs."""
    session = pc.CacheSession(cache=cache, model=model)
    lookup = session.lookup(options, sampler_identity=sampler_identity, versions=versions)
    if lookup.hit:
        return lookup.result, True
    result = monitor_run(model=model, **lookup.normalized_kwargs)
    report = report_from_idata(result)
    status = report.get("status", "unverified") if report else (
        "unverified" if options.get("until") else "complete"
    )
    session.store(lookup, result, status=status, effective_sampler=sampler_identity)
    return result, False


def main():
    # Replace pm.sample with your pymc-monitor runner after verifying its call
    # signature, version and convergence-report contract in your installation.
    with pm.Model() as model:
        pm.Normal("x")
    cache = pc.MemoryCache()
    opts = dict(draws=2, tune=2, chains=1, cores=1, random_seed=42, progressbar=False)
    first, hit = run_with_monitor(model, pm.sample, cache=cache,
                                  sampler_identity="native-pymc-example", versions={
                                      "runner": pm.__version__}, **opts)
    assert not hit
    second, hit = run_with_monitor(model, pm.sample, cache=cache,
                                   sampler_identity="native-pymc-example", versions={
                                       "runner": pm.__version__}, **opts)
    assert hit and first is not second
    print("Preflight skipped the runner:", hit)


if __name__ == "__main__":
    main()
