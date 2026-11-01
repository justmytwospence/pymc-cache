"""Lossless divergence-message storage and validation."""
import importlib.metadata
import json

import arviz as az
import numpy as np
import pytest
import xarray as xr

from pymc_cache import CacheWarning, DiskCache
from pymc_cache.io import load_inference_data, sanitize_inference_data, save_inference_data

MARKER = "_pymc_cache_text_codec_v1"


def make_idata(values, group="sample_stats"):
    array = np.asarray(values, dtype=object)
    dims = tuple(f"dim_{i}" for i in range(array.ndim))
    coords = {dim: np.arange(size) + 10 for dim, size in zip(dims, array.shape, strict=True)}
    ds = xr.Dataset({"divergence_message": (dims, array)}, coords=coords)
    ds.divergence_message.attrs["nested"] = {"a": [1, None]}
    ds.divergence_message.encoding.update({"dtype": "S1", "_FillValue": b"x"})
    posterior = xr.Dataset({"x": ("draw", [1.0])})
    if int(importlib.metadata.version("arviz").split(".")[0]) >= 1:
        return xr.DataTree.from_dict({"/posterior": posterior, f"/{group}": ds})
    return az.InferenceData(**{"posterior": posterior, group: ds}, save_warmup=True)


@pytest.mark.parametrize("group", ["sample_stats", "warmup_sample_stats"])
@pytest.mark.parametrize("values", [
    [None, "", "snow ☃\x00end", b"\x00\xff", np.str_("hello"), np.bytes_(b"abc"), "x" * 10000],
    np.array(None, dtype=object),
    np.array([], dtype=object),
])
def test_roundtrip(tmp_path, group, values):
    source = make_idata(values, group)
    original = source[group].divergence_message
    snapshot = sanitize_inference_data(source)
    assert snapshot[group].divergence_message.attrs[MARKER] == "json-tagged"
    assert MARKER not in original.attrs
    assert original.encoding["dtype"] == "S1"
    path = tmp_path / "data.nc"
    save_inference_data(source, path)
    restored = load_inference_data(path)[group].divergence_message
    assert restored.dtype == object
    assert restored.dims == original.dims
    assert restored.shape == original.shape
    for dim in original.dims:
        np.testing.assert_array_equal(restored.coords[dim], original.coords[dim])
    assert restored.attrs == original.attrs
    assert restored.values.tolist() == original.values.tolist()
    assert MARKER not in original.attrs


@pytest.mark.parametrize("record", ['[true]', '[0, 1]', '[1, 2]', '[2, "!!!"]',
                                    '[2, "Zh=="]', '[3]', '{}', 'bad'])
def test_invalid_records(tmp_path, record):
    source = make_idata([None])
    data = source.sample_stats.divergence_message
    data.data = np.array([record])
    data.attrs[MARKER] = "json-tagged"
    data.attrs.pop("nested")
    data.encoding.clear()
    path = tmp_path / "invalid.nc"
    source.to_netcdf(path, engine="h5netcdf")
    with pytest.raises(ValueError, match="malformed"):
        load_inference_data(path)


def test_unknown_and_collision(tmp_path):
    source = make_idata([None])
    source.sample_stats.divergence_message.attrs[MARKER] = "future"
    with pytest.raises(ValueError, match="reserved"):
        sanitize_inference_data(source)
    source.sample_stats.divergence_message.data = np.array(["[0]"])
    source.sample_stats.divergence_message.attrs.pop("nested")
    source.sample_stats.divergence_message.encoding.clear()
    path = tmp_path / "unknown.nc"
    source.to_netcdf(path, engine="h5netcdf")
    with pytest.raises(ValueError, match="unknown"):
        load_inference_data(path)


def test_ordinary_strings_are_unchanged(tmp_path):
    source = make_idata([None])
    data = source.sample_stats.divergence_message
    data.data = np.array(["plain"])
    data.encoding.clear()
    path = tmp_path / "plain.nc"
    save_inference_data(source, path)
    restored = load_inference_data(path).sample_stats.divergence_message
    assert restored.values.tolist() == ["plain"]
    assert MARKER not in restored.attrs


def test_unsafe_object_is_not_stringified():
    class Bad:
        def __str__(self):
            raise AssertionError("str called")

        def __repr__(self):
            raise AssertionError("repr called")

    with pytest.raises(TypeError, match="unsupported divergence_message"):
        sanitize_inference_data(make_idata([Bad()]))


def test_schema_one_preserved(tmp_path):
    import h5py

    cache = DiskCache(tmp_path)
    key = "a" * 64
    cache.save(key, make_idata([None]))
    path = cache._path(key)
    with h5py.File(path, "r+") as h5:
        attrs = h5["posterior"].attrs
        del attrs["_pymc_cache_entry_v2"]
        attrs["_pymc_cache_entry_v1"] = json.dumps({"schema": 1, "key": key})
    before = path.read_bytes()
    with pytest.warns(CacheWarning):
        assert cache.load(key) is None
    assert cache.clear() == 0
    with pytest.raises(ValueError, match="schema-1"):
        cache.save(key, make_idata([None]))
    assert path.read_bytes() == before


def test_real_nutpie_if_installed(tmp_path):
    nutpie = pytest.importorskip("nutpie")
    import pymc as pm

    with pm.Model() as model:
        pm.Normal("x")
    result = nutpie.sample(nutpie.compile_pymc_model(model), draws=2, tune=2, chains=1,
                           cores=1, progress_bar=False, seed=42)
    path = tmp_path / "nutpie.nc"
    save_inference_data(result, path)
    restored = load_inference_data(path)
    np.testing.assert_array_equal(restored.sample_stats.divergence_message,
                                  result.sample_stats.divergence_message)
