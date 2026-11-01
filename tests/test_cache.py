"""Storage, preflight, and standalone sampling regression tests."""
from __future__ import annotations

import copy
import json
import os
import shutil

import h5py
import numpy as np
import pymc as pm
import pytest
import xarray as xr

from pymc_cache import (
    CacheSession,
    CacheWarning,
    DiskCache,
    MemoryCache,
    cleanup_cache,
    sample,
    sample_smc,
)
from pymc_cache.backends import resolve_cache
from pymc_cache.io import group_names


@pytest.fixture
def model():
    with pm.Model() as m:
        pm.Normal("x", 0, 1)
    return m


@pytest.fixture
def idata():
    import importlib.metadata

    import arviz as az

    posterior = xr.Dataset({"x": (("chain", "draw"), np.ones((1, 2)))})
    if int(importlib.metadata.version("arviz").split(".")[0]) >= 1:
        result = xr.DataTree.from_dict({"/posterior": posterior})
    else:
        result = az.InferenceData(posterior=posterior)
    assert "posterior" in group_names(result)
    return result


def test_memory_copy_and_lru():
    c = MemoryCache(max_entries=2)
    value = {"a": [1]}
    c.save("a", value)
    value["a"].append(2)
    loaded = c.load("a")
    assert loaded == {"a": [1]}
    loaded["a"].append(3)
    assert c.load("a") == {"a": [1]}
    c.save("b", value)
    c.load("a")
    c.save("c", value)
    assert c.load("b") is None
    with pytest.raises(ValueError):
        MemoryCache(0)


def test_disk_roundtrip_metadata_attrs(tmp_path, idata, model):
    key = "a" * 64
    cache = DiskCache(tmp_path, model=model)
    idata.attrs["nested"] = {"test": [1, 2]}
    idata.posterior.attrs["nested"] = {"x": True}
    idata.posterior.x.attrs["nested"] = {"y": 2}
    assert cache.load(key) is None
    cache.save(key, idata)
    assert idata.attrs["nested"] == {"test": [1, 2]}
    path = cache._path(key)
    assert path.name.endswith(key + ".nc")
    assert len(list(tmp_path.rglob("*.nc"))) == 1
    loaded = cache.load(key)
    assert loaded.attrs["nested"] == {"test": [1, 2]}
    assert loaded.posterior.attrs["nested"] == {"x": True}
    assert loaded.posterior.x.attrs["nested"] == {"y": 2}
    assert cache.load("b" * 64) is None
    assert cache.entries()[0].key == key
    # ArviZ 0.x may retain a read-only HDF5 handle after materializing groups.
    # Replace atomically to simulate a corrupt file without opening that handle for writes.

    tampered = path.with_suffix(".tampered")
    shutil.copyfile(path, tampered)
    with h5py.File(tampered, "r+") as h5:
        h5["posterior"].attrs["_pymc_cache_entry_v3"] = '{"schema":3,"key":"wrong"}'
    os.replace(tampered, path)
    with pytest.warns(CacheWarning):
        assert cache.load(key) is None
    with pytest.raises(ValueError):
        cache.load("relative/evil")


def test_corrupt_symlink_and_own_cleanup(tmp_path, idata):
    cache = DiskCache(tmp_path)
    corrupt = cache._path("c" * 64)
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"bad")
    with pytest.warns(CacheWarning):
        assert cache.load("c" * 64) is None
    cache.save("d" * 64, idata)
    other = corrupt.parent / "other.txt"
    other.write_text("keep")
    link = corrupt.parent / ("default-" + "e" * 64 + ".nc")
    link.symlink_to(cache._path("d" * 64))
    assert cache.load("e" * 64) is None
    assert cleanup_cache(tmp_path) == 1
    assert other.exists() and link.is_symlink() and corrupt.exists()
    with pytest.raises(ValueError):
        cache.save("e" * 64, idata)


def test_disk_prune(tmp_path, idata):
    cache = DiskCache(tmp_path)
    for i in range(23):
        cache.save(f"{i:064x}", idata)
    assert len(cache.entries()) == 20
    assert cache.prune(1) == 19
    assert cleanup_cache(tmp_path) == 1
    with pytest.raises(ValueError):
        cache.prune(-1)


def test_resolve_cache(tmp_path):
    assert resolve_cache(False) is None
    assert isinstance(resolve_cache(True), MemoryCache)
    assert isinstance(resolve_cache("disk"), DiskCache)
    assert isinstance(resolve_cache(tmp_path), DiskCache)
    with pytest.raises(TypeError):
        resolve_cache(13)


def test_preflight_and_status(model, idata, tmp_path):
    session = CacheSession(cache=tmp_path, model=model)
    lookup = session.lookup({"draws": 2})
    assert not lookup.hit and lookup.key
    for status in ("stopped", "futile", "capped", "unverified"):
        assert not session.store(lookup, idata, status=status)
    assert session.store(lookup, idata)
    assert session.lookup({"draws": 2}).hit
    assert not session.lookup({"draws": 3}).hit
    assert not session.lookup({"draws": 2}, operation="sample_smc").hit
    assert not session.lookup({"draws": 2}, overwrite=True).hit
    assert CacheSession(cache=False, model=model).lookup({"draws": 2}).key is None


def test_custom_backend_optional_metadata(model, idata):
    class Minimal:
        def __init__(self):
            self.data = {}

        def load(self, key):
            return self.data.get(key)

        def save(self, key, idata):
            self.data[key] = idata

    class Legacy(Minimal):
        def load(self, key, sample_kwargs=None):
            assert sample_kwargs == {"draws": 2}
            return super().load(key)

        def save(self, key, idata, sample_kwargs=None):
            assert sample_kwargs == {"draws": 2}
            super().save(key, idata)

    class Extended(Minimal):
        def load(self, key, *, sample_kwargs=None):
            assert sample_kwargs == {"draws": 2}
            return super().load(key)

        def save(self, key, idata, *, sample_kwargs=None, operation=None, status=None,
                 software_versions=None):
            assert (sample_kwargs, operation, status, software_versions) == (
                {"draws": 2}, "sample", "complete", {"runner": "1"})
            super().save(key, idata)

    class Variadic(Minimal):
        def load(self, key, **kwargs):
            assert kwargs == {"sample_kwargs": {"draws": 2}}
            return super().load(key)

        def save(self, key, idata, **kwargs):
            assert kwargs == {"sample_kwargs": {"draws": 2}, "operation": "sample",
                              "status": "complete", "software_versions": {"runner": "1"}}
            super().save(key, idata)

    class PositionalOnly(Minimal):
        def load(self, key, sample_kwargs=None, /):
            assert sample_kwargs is None
            return super().load(key)

        def save(self, key, idata, sample_kwargs=None, /):
            assert sample_kwargs is None
            super().save(key, idata)

    for backend_type in (Minimal, Legacy, Extended, Variadic, PositionalOnly):
        backend = backend_type()
        session = CacheSession(backend, model=model)
        lookup = session.lookup({"draws": 2}, versions={"runner": "1"})
        assert not lookup.hit
        assert session.store(lookup, idata)
        assert session.lookup({"draws": 2}, versions={"runner": "1"}).hit


def test_backend_uninspectable_signature_uses_minimal_protocol(monkeypatch, model, idata):
    import pymc_cache.session as module

    class Backend:
        def load(self, key):
            return None

        def save(self, key, value):
            self.saved = (key, value)

    backend = Backend()
    signature = module.inspect.signature

    def unavailable(method):
        if getattr(method, "__self__", None) is backend:
            raise ValueError("no signature")
        return signature(method)

    monkeypatch.setattr(module.inspect, "signature", unavailable)
    session = CacheSession(backend, model=model)
    lookup = session.lookup({"draws": 2})
    assert session.store(lookup, idata)
    assert backend.saved == (lookup.key, idata)


def test_backend_internal_typeerror_is_not_retried(model, idata):
    class Backend:
        def __init__(self):
            self.reads = 0
            self.writes = 0

        def load(self, key, sample_kwargs=None):
            self.reads += 1
            raise TypeError("read side effect")

        def save(self, key, idata, sample_kwargs=None):
            self.writes += 1
            raise TypeError("write side effect")

    backend = Backend()
    session = CacheSession(backend, model=model)
    with pytest.warns(CacheWarning, match="cache read failed.*read side effect"):
        lookup = session.lookup({"draws": 2})
    with pytest.warns(CacheWarning, match="cache write failed.*write side effect"):
        assert not session.store(lookup, idata)
    assert (backend.reads, backend.writes) == (1, 1)


@pytest.mark.parametrize(("attribute", "actual"), [
    ("pymc_monitor_convergence", "pymc"),
    ("cloudposterior_convergence", None),
])
def test_reported_sampler_fallback_never_stored_under_nutpie_key(
        monkeypatch, model, idata, attribute, actual):
    cache = MemoryCache()
    calls = []

    def fallback(*, model, **kwargs):
        calls.append(kwargs)
        result = copy.deepcopy(idata)
        result.posterior.attrs[attribute] = json.dumps(
            {"status": "complete", "effective_sampler": actual})
        return result

    monkeypatch.setattr(pm, "sample", fallback)
    kwargs = dict(model=model, cache=cache, draws=2, tune=2, chains=1, cores=1,
                  nuts_sampler="nutpie", random_seed=4)
    for _ in range(2):
        with pytest.warns(CacheWarning, match="effective sampler differs"):
            assert sample(**kwargs) is not None
    assert len(calls) == 2 and len(cache) == 0


def test_matching_reported_sampler_still_cached(model, idata):
    cache = MemoryCache()
    session = CacheSession(cache, model=model)
    lookup = session.lookup({"draws": 2}, sampler_identity="nutpie")
    idata.posterior.attrs["pymc_monitor_convergence"] = json.dumps(
        {"status": "complete", "effective_sampler": "nutpie"})
    assert session.store(lookup, idata, effective_sampler="nutpie")
    assert session.lookup({"draws": 2}, sampler_identity="nutpie").hit


def test_disabled_zero_hash_work(monkeypatch, model):
    import pymc_cache.session as module

    def fail(*args, **kwargs):
        raise AssertionError("hashed on cache=False")

    monkeypatch.setattr(module, "model_digest", fail)
    result = sample(model=model, cache=False, draws=2, tune=2,
                    chains=1, cores=1, progressbar=False)
    assert result is not None


def test_native_repeat_hit_and_overwrite(monkeypatch, model):
    cache = MemoryCache()
    kwargs = dict(model=model, cache=cache, draws=2, tune=2,
                  chains=1, cores=1, progressbar=False, random_seed=4)
    first = sample(**kwargs)
    real = pm.sample
    calls = []

    def spy(*args, **kw):
        calls.append(1)
        return real(*args, **kw)

    monkeypatch.setattr(pm, "sample", spy)
    again = sample(**kwargs)
    np.testing.assert_array_equal(first.posterior.x, again.posterior.x)
    assert calls == []
    sample(**kwargs, overwrite=True)
    assert calls == [1]


def test_runner_contract_and_adaptive_status(model, idata):
    class Runner:
        def __init__(self):
            self.calls = 0
            self.status = "stopped"

        def __pymc_cache_normalize__(self, model, call_kwargs):
            return {"call_kwargs": call_kwargs, "key_kwargs": call_kwargs,
                    "sampler_identity": "runner-1", "versions": {"runner": "1"}}

        def __call__(self, *, model, **kwargs):
            self.calls += 1
            result = copy.deepcopy(idata)
            result.posterior.attrs["pymc_monitor_convergence"] = (
                '{"status":"' + self.status + '","met":' +
                ("true" if self.status == "met" else "false") + "}"
            )
            return result

    runner = Runner()
    cache = MemoryCache()
    for _ in range(2):
        sample(model=model, cache=cache, sample_fn=runner, until="draft")
    assert runner.calls == 2
    runner.status = "met"
    sample(model=model, cache=cache, sample_fn=runner, until="draft")
    sample(model=model, cache=cache, sample_fn=runner, until="draft")
    assert runner.calls == 3


def test_unsafe_callable_and_model_failure(model):
    with pytest.warns(CacheWarning):
        assert CacheSession(model=model).lookup({"callback": lambda: 1}).key is None


def test_smc_operation_isolated(model):
    kwargs = dict(model=model, cache=MemoryCache(), draws=2, chains=1, cores=1,
                  random_seed=1, progressbar=False)
    result = sample_smc(**kwargs)
    assert result is not None
