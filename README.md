# pymc-cache

Safe, model-aware caching of native PyMC sampling results. Python 3.11+, PyMC 5.28–6 and ArviZ 0.x–1.x. No cloud account, monitoring service or Modal dependency.

```bash
pip install 'pymc-cache @ git+https://github.com/justmytwospence/pymc-cache.git@<commit-sha>'
```

```python
import pymc as pm
import pymc_cache as pc

with pm.Model() as model:
    mu = pm.Normal("mu", 0, 1)
    options = dict(draws=10, tune=10, chains=1, cores=1, random_seed=42,
                   progressbar=False, model=model, cache="disk")
    trace = pc.sample(**options)
    # The next call loads the result from .pymc-cache/ without sampling:
    again = pc.sample(**options)
```

`pc.sample_smc(model=model, draws=100, cache="disk")` uses the native SMC sampler with an independent key. `cache=True` (default) uses an 8-entry in-memory LRU. `cache=False` runs the native sampler without hashing the model. Pass a directory/Path for a custom disk location; `overwrite=True` skips reads and atomically replaces an existing entry. Both backends copy results on read; memory copies on write too. Disk files are one NetCDF file per full SHA-256 key, with the key embedded in its metadata. Corruption is treated as a miss. `pc.cleanup_cache()` removes only entries owned by this package; `pymc-cache list|prune|clear --dir PATH` inspects/manages the same entries without following symlinks.

## Preflight and integrations

```python
session = pc.CacheSession(cache="disk", model=model)
lookup = session.lookup({"draws": 10, "tune": 10, "chains": 1,
                         "cores": 1, "random_seed": 42, "progressbar": False})
if lookup.hit:
    trace = lookup.result
else:
    trace = pm.sample(model=model, **lookup.normalized_kwargs)
    session.store(lookup, trace, status="complete")
```

A runner used with `pc.sample(..., sample_fn=runner)` must implement `runner.__pymc_cache_normalize__(model, call_kwargs)` and return **exactly** `call_kwargs`, `key_kwargs`, `sampler_identity`, and `versions`. The call arguments passed to the runner and the canonical key arguments must be explicit. The runner returns an inference result; only complete results, or adaptive results with an attested `met` convergence report, are stored. Stopped, futile, capped and incomplete results are returned but never stored. Cache identity includes the complete model graph, constants, mutable shared data, deterministics, coordinates, sampling kwargs, RNG states, operation, sampler and software versions. Unknown/custom Ops fail closed rather than risk a wrong hit. Unsafe steps/callbacks/callables require an explicit `sampler_key` identifying their implementation and configuration; the caller is responsible for changing this key when the callable changes. No code is executed from cache metadata.

Caching may hold sensitive posterior data. Treat `.pymc-cache/` as private, and use `cache=False` for confidential models when persistent data is not allowed. Cache files are not authenticated against a malicious writer; do not use a shared untrusted directory.

See [getting started](getting-started.qmd), [caching walkthrough](guides/walkthrough.qmd), [identity guide](guides/identity.qmd), [storage guide](guides/storage.qmd), [monitor composition](guides/monitor.qmd), [legacy migration](guides/migration.qmd), [troubleshooting](guides/troubleshooting.qmd), [runnable examples](examples/basic.py), and [API reference](reference.qmd). This package does not patch `pm.sample`, intercept model contexts, cache posterior predictive calls or provide cloud execution.
