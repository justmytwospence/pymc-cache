"""Public exports, runnable examples and concurrent storage contracts."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pymc as pm
import pytest
import xarray as xr

import pymc_cache as pc

ROOT = Path(__file__).resolve().parents[1]


def _idata(value):
    import importlib.metadata

    import arviz as az

    posterior = xr.Dataset({"x": (("chain", "draw"), np.full((1, 2), value))})
    if int(importlib.metadata.version("arviz").split(".")[0]) >= 1:
        return xr.DataTree.from_dict({"/posterior": posterior})
    return az.InferenceData(posterior=posterior)


def test_public_exports_and_no_monitor_runtime_dependency():
    assert set(pc.__all__) == {
        "sample", "sample_smc", "CacheBackend", "MemoryCache", "DiskCache",
        "CacheSession", "CachePolicy", "CacheEntry", "CacheLookup", "UncacheableError",
        "CacheWarning", "cleanup_cache", "model_digest", "cache_key", "__version__",
    }
    assert all(hasattr(pc, name) for name in pc.__all__)
    assert (ROOT / "src/pymc_cache/py.typed").is_file()
    from importlib.metadata import distribution

    assert "pymc-monitor" not in " ".join(distribution("pymc-cache").requires or []).lower()
    assert "cloudposterior" not in " ".join(distribution("pymc-cache").requires or [])


@pytest.mark.parametrize("name", ["basic", "preflight", "custom_backend", "monitor_composition"])
def test_runnable_example(name, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "examples" / f"{name}.py")],
                            cwd=tmp_path, text=True, capture_output=True, check=True)
    assert result.stdout.strip()


def test_atomic_concurrent_writers_same_key(tmp_path):
    cache = pc.DiskCache(tmp_path)
    key = "a" * 64
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(lambda n: cache.save(key, _idata(n)), range(4)))
    assert len(cache.entries()) == 1
    assert not list(tmp_path.rglob("*.tmp"))
    assert np.unique(cache.load(key).posterior.x.values).size == 1


def test_unsafe_step_needs_sampler_key_and_changes_identity():
    def step(value):
        return value
    with pytest.raises(pc.UncacheableError):
        pc.cache_key("model", {"step": step})
    first = pc.cache_key("model", {"step": step}, sampler_key="step-v1")
    assert first != pc.cache_key("model", {"step": step}, sampler_key="step-v2")


def test_no_unverified_monitor_result_is_stored():
    spec = importlib.util.spec_from_file_location(
        "monitor_composition_example", ROOT / "examples/monitor_composition.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pm.Model() as model:
        pm.Normal("x")
    calls = []

    def runner(*, model, **options):
        calls.append(1)
        result = _idata(1)
        result.posterior.attrs["pymc_monitor_convergence"] = json.dumps(
            {"status": "stopped", "met": False})
        return result

    cache = pc.MemoryCache()
    for _ in range(2):
        _, hit = module.run_with_monitor(model, runner, cache=cache,
                                         sampler_identity="monitor-v1", versions={"runner": "1"},
                                         until="converged", random_seed=42)
        assert not hit
    assert len(calls) == 2 and len(cache) == 0
