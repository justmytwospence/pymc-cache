"""Fail-closed regressions and management operations."""
from __future__ import annotations

import copy
import json

import numpy as np
import pymc as pm
import pytest

from pymc_cache import (
    CachePolicy,
    CacheSession,
    CacheWarning,
    DiskCache,
    MemoryCache,
    UncacheableError,
    cache_key,
    model_digest,
    sample,
)
from pymc_cache.cli import main
from pymc_cache.io import get_group, sanitize_inference_data
from pymc_cache.policy import eligible, report_from_idata


def test_identity_types_and_unsafe_objects():
    assert cache_key("m", {"x": (1, 2)}) != cache_key("m", {"x": [1, 2]})
    assert cache_key("m", {"x": {1, 2}}) == cache_key("m", {"x": {2, 1}})
    assert cache_key("m", {"x": float("nan")}) == cache_key("m", {"x": float("nan")})
    assert cache_key("m", {"x": float("inf")}) != cache_key("m", {"x": -float("inf")})
    assert cache_key("m", {"seed": np.random.RandomState(1)}) != cache_key(
        "m", {"seed": np.random.RandomState(2)})
    with pytest.raises(UncacheableError):
        cache_key("m", {"state": object()}, sampler_key="runner-v1")
    with pytest.raises(UncacheableError):
        cache_key("m", {"state": {"unexpected": object()}}, sampler_key="runner-v1")
    with pytest.raises(ValueError):
        cache_key("m", {}, sampler_key="")
    with pytest.raises(ValueError):
        model_digest(object(), model_key="")
    assert cache_key("m", {}, versions={"pymc": "fake"}) != cache_key("m", {})


def test_model_custom_op_requires_explicit_key():
    import pytensor.tensor as pt
    from pytensor.graph.basic import Apply
    from pytensor.graph.op import Op

    class Custom(Op):
        __props__ = ()

        def make_node(self, x):
            x = pt.as_tensor_variable(x)
            return Apply(self, [x], [x.type()])

        def perform(self, node, inputs, outputs):
            outputs[0][0] = inputs[0]

    with pm.Model() as model:
        x = pm.Normal("x")
        pm.Potential("custom", Custom()(x))
    with pytest.raises(UncacheableError, match="custom Op"):
        model_digest(model)
    assert model_digest(model, model_key="implementation-v1") != model_digest(
        model, model_key="implementation-v2")


def test_preflight_policy_mutation_and_errors(monkeypatch):
    with pm.Model() as model:
        data = pm.Data("data", np.array([1., 2.]))
        pm.Normal("x", observed=data)
    backend = MemoryCache()
    session = CacheSession(backend, model=model)
    lookup = session.lookup({"draws": 2, "random_seed": np.random.default_rng(1)})
    assert lookup.key and not lookup.hit
    assert lookup.normalized_kwargs["random_seed"] is not None
    with model:
        pm.set_data({"data": np.array([2., 3.])})
    with pytest.warns(CacheWarning, match="model changed"):
        assert not session.store(lookup, object())
    assert len(backend) == 0
    assert CacheSession(backend, model=model, policy=CachePolicy(False)).lookup(
        {"draws": 2}).disable_reason == "unseeded reuse disabled"
    with pytest.raises(TypeError):
        CacheSession(model=model, policy=object())
    with pytest.raises(TypeError):
        session.lookup([])

    def broken(*args, **kwargs):
        raise OSError("backend failed")

    monkeypatch.setattr(backend, "load", broken)
    with pytest.warns(CacheWarning, match="cache read failed"):
        assert not session.lookup({"random_seed": 1}).hit


def _idata():
    import importlib.metadata

    import arviz as az
    import xarray as xr

    posterior = xr.Dataset({"x": (("chain", "draw"), np.ones((1, 2)))})
    if int(importlib.metadata.version("arviz").split(".")[0]) >= 1:
        return xr.DataTree.from_dict({"/posterior": posterior})
    return az.InferenceData(posterior=posterior)


def test_eligibility_requires_verified_adaptive_report():
    idata = _idata()
    assert eligible(idata, status="complete", kwargs={"draws": 2})
    assert not eligible(idata, status="met", kwargs={"draws": 2})
    assert not eligible(idata, status="complete", kwargs={"draws": 3})
    assert not eligible(idata, status="complete", kwargs={"chains": 2})
    assert not eligible(idata, status="complete", kwargs={"until": "converged"})
    assert not eligible(idata, status="complete", kwargs={}, effective_sampler="a",
                        sampler_identity="b")
    idata.posterior.attrs["pymc_monitor_convergence"] = json.dumps(
        {"status": "met", "met": True})
    assert report_from_idata(idata)["met"]
    assert eligible(idata, status="met", kwargs={"until": "converged"})
    idata.posterior.attrs["pymc_monitor_convergence"] = "invalid"
    assert report_from_idata(idata) is None
    assert not eligible(idata, status="met", kwargs={})


def test_disk_management_cli_and_write_failure(tmp_path):
    idata = _idata()
    cache = DiskCache(tmp_path, max_entries_per_model=3)
    cache.save("a" * 64, idata)
    foreign = tmp_path / "unnamed" / "foreign.nc"
    foreign.write_bytes(b"unowned")
    assert main(["--dir", str(tmp_path), "list", "--json"]) == 0
    assert main(["--dir", str(tmp_path), "prune", "--keep", "0"]) == 0
    assert foreign.exists()
    cache.save("b" * 64, idata)
    assert cache.delete("b" * 64)
    assert not cache.delete("b" * 64)
    cache.save("c" * 64, idata)
    with pytest.raises(SystemExit) as refusal:
        main(["--directory", str(tmp_path), "clear"])
    assert refusal.value.code == 2
    assert [entry.key for entry in cache.entries()] == ["c" * 64]
    assert main(["clear", "--directory", str(tmp_path), "--yes"]) == 0
    assert not cache.entries()
    assert foreign.exists()
    with pytest.raises(ValueError):
        DiskCache(tmp_path, max_entries_per_model=0)
    with pytest.raises(ValueError):
        cache.save("c" * 64, idata, status="stopped")
    with pytest.raises(ValueError):
        cache._path("bad")
    idata.posterior.x.attrs["bad"] = object()
    with pytest.raises(TypeError):
        cache.save("c" * 64, idata)
    assert not list(tmp_path.rglob("*.tmp"))


def test_io_does_not_mutate_source_and_rejects_unsafe_attrs():
    idata = _idata()
    idata.posterior.attrs["complex"] = {"a": (1, None)}
    snapshot = sanitize_inference_data(idata)
    assert snapshot.posterior.attrs["complex"] != idata.posterior.attrs["complex"]
    assert get_group(idata, "absent") is None
    idata.posterior.x.attrs["bad"] = np.array([object()], dtype=object)
    with pytest.raises(TypeError):
        sanitize_inference_data(idata)
    idata.posterior.x.attrs["bad"] = copy.deepcopy(object())
    with pytest.raises(TypeError):
        sanitize_inference_data(idata)


def test_custom_runner_without_identity_never_reuses():
    with pm.Model() as model:
        pm.Normal("x")
    calls = []

    def runner(*, model, **kwargs):
        calls.append(1)
        return _idata()

    with pytest.warns(CacheWarning, match="sample_fn requires"):
        sample(model=model, sample_fn=runner)
    assert len(calls) == 1
