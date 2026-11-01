"""Lossless diagnostic text storage and strict validation across ArviZ majors."""
import importlib.metadata
import json
from decimal import Decimal

import arviz as az
import numpy as np
import pytest
import xarray as xr

from pymc_cache import CacheWarning, DiskCache
from pymc_cache.io import load_inference_data, sanitize_inference_data, save_inference_data

MARKER = "_pymc_cache_text_codec_v1"
GROUPS = ("sample_stats", "warmup_sample_stats")
VARS = ("divergence_message", "warning")


def make_idata(values, group="sample_stats", var="divergence_message"):
    array = np.asarray(values, dtype=object)
    dims = tuple(f"dim_{i}" for i in range(array.ndim))
    coords = {dim: np.arange(size) + 10 for dim, size in zip(dims, array.shape, strict=True)}
    ds = xr.Dataset({var: (dims, array)}, coords=coords, attrs={"source": "synthetic"})
    ds[var].attrs["nested"] = {"a": [1, None]}
    ds[var].encoding.update({"dtype": "S1", "_FillValue": b"x"})
    posterior = xr.Dataset({"x": ("draw", [1.0])})
    if int(importlib.metadata.version("arviz").split(".")[0]) >= 1:
        return xr.DataTree.from_dict({"/posterior": posterior, f"/{group}": ds})
    return az.InferenceData(**{"posterior": posterior, group: ds}, save_warmup=True)


def assert_values(actual, expected):
    assert actual.shape == expected.shape
    for index in np.ndindex(actual.shape):
        left, right = actual[index], expected[index]
        if isinstance(right, (float, np.floating)) and np.isnan(right):
            assert isinstance(left, float) and np.isnan(left)
        else:
            assert type(left) is (bytes if type(right) is np.bytes_ else
                                  str if type(right) is np.str_ else type(right))
            assert left == right


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("var", VARS)
@pytest.mark.parametrize("values", [
    [None, "", "nan", "[3]", "snow ☃\x00end", b"\x00\xff", np.str_("hello"),
     np.bytes_(b"abc"), float("nan"), np.float32("nan"), np.longdouble("nan"), "x" * 10000],
    [None, None],
    [float("nan"), np.float16("nan"), np.float32("nan"), np.float64("nan")],
    np.array(None, dtype=object),
    np.array(float("nan"), dtype=object),
    np.array([], dtype=object),
])
def test_roundtrip(tmp_path, group, var, values):
    source = make_idata(values, group, var)
    original = source[group][var]
    snapshot = sanitize_inference_data(source)
    assert snapshot[group][var].attrs[MARKER] == "json-tagged-v2"
    for record in snapshot[group][var].values.flat:
        record.encode("ascii")
        assert json.loads(record)[0] in (0, 1, 2, 3)
    assert MARKER not in original.attrs
    assert original.encoding["dtype"] == "S1"
    path = tmp_path / "data.nc"
    save_inference_data(source, path)
    restored = load_inference_data(path)[group][var]
    assert restored.dtype == object
    assert restored.dims == original.dims
    assert restored.shape == original.shape
    for dim in original.dims:
        np.testing.assert_array_equal(restored.coords[dim], original.coords[dim])
    assert restored.attrs == original.attrs
    assert source[group].attrs["source"] == "synthetic"
    assert MARKER not in original.attrs
    assert_values(restored.values, original.values)


def marked_file(tmp_path, record, marker="json-tagged-v2", group="sample_stats",
                var="divergence_message"):
    source = make_idata([None], group, var)
    data = source[group][var]
    data.data = np.array([record])
    data.attrs[MARKER] = marker
    data.attrs.pop("nested")
    data.encoding.clear()
    path = tmp_path / "marked.nc"
    source.to_netcdf(path, engine="h5netcdf")
    return path


@pytest.mark.parametrize("marker", ["json-tagged", "json-tagged-v2"])
@pytest.mark.parametrize("record", ['[true]', '[0, 1]', '[1, 2]', '[2, "!!!"]',
                                    '[2, "Zh=="]', '[1, NaN]', '[1, Infinity]',
                                    '[4]', '[3, 1]', '[1, "é"]', '{}', 'bad'])
def test_invalid_records(tmp_path, record, marker):
    path = marked_file(tmp_path, record, marker)
    with pytest.raises(ValueError, match="malformed"):
        load_inference_data(path)


@pytest.mark.parametrize("record,expected", [('[0]', None), ('[1, "[3]"]', "[3]"),
                                            ('[2, "AP8="]', b"\x00\xff")])
def test_legacy_divergence_records(tmp_path, record, expected):
    path = marked_file(tmp_path, record, "json-tagged")
    assert load_inference_data(path).sample_stats.divergence_message.values[0] == expected


def test_legacy_rejects_nan_and_warning(tmp_path):
    with pytest.raises(ValueError, match="malformed"):
        load_inference_data(marked_file(tmp_path, '[3]', "json-tagged"))
    with pytest.raises(ValueError, match="unknown"):
        load_inference_data(marked_file(tmp_path, '[0]', "json-tagged", var="warning"))


@pytest.mark.parametrize("group,var", [("sample_stats", "other"), ("posterior", "warning")])
def test_marker_outside_allowlist(tmp_path, group, var):
    with pytest.raises(ValueError, match="unknown"):
        load_inference_data(marked_file(tmp_path, '[0]', group=group, var=var))


def test_unknown_and_collision(tmp_path):
    source = make_idata([None])
    source.sample_stats.divergence_message.attrs[MARKER] = "future"
    with pytest.raises(ValueError, match="reserved"):
        sanitize_inference_data(source)
    with pytest.raises(ValueError, match="unknown"):
        load_inference_data(marked_file(tmp_path, '[0]', "future"))


@pytest.mark.parametrize("var", VARS)
def test_ordinary_strings_are_unchanged(tmp_path, var):
    source = make_idata([None], var=var)
    data = source.sample_stats[var]
    data.data = np.array(["plain"])
    data.encoding.clear()
    path = tmp_path / "plain.nc"
    save_inference_data(source, path)
    restored = load_inference_data(path).sample_stats[var]
    assert restored.values.tolist() == ["plain"]
    assert MARKER not in restored.attrs


class SamplerWarning:
    def __str__(self):
        raise AssertionError("str called")

    def __repr__(self):
        raise AssertionError("repr called")


class Bad:
    def __str__(self):
        raise AssertionError("str called")

    def __repr__(self):
        raise AssertionError("repr called")


@pytest.mark.parametrize("value", [0, 1, True, 1.5, float("inf"), float("-inf"),
                                   np.float32(1), np.float64("inf"), 1j,
                                   Decimal("NaN"), ["x"], {"warning": "x"},
                                   SamplerWarning(), Bad()])
@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("var", VARS)
def test_unsafe_object_is_not_stringified(value, group, var):
    with pytest.raises(TypeError, match=rf"{group}\.{var} at index \(0,\): "
                                       rf"{type(value).__name__}"):
        values = np.empty(1, dtype=object)
        values[0] = value
        sanitize_inference_data(make_idata(values, group, var))


def test_identity_schema_isolation(monkeypatch):
    import pymc_cache.identity as identity

    assert identity.SCHEMA_VERSION == 3
    current = identity.cache_key("model", {"draws": 2})
    for old in (1, 2):
        monkeypatch.setattr(identity, "SCHEMA_VERSION", old)
        assert identity.cache_key("model", {"draws": 2}) != current
    monkeypatch.setattr(identity, "SCHEMA_VERSION", 3)


@pytest.mark.parametrize("schema", [1, 2])
def test_old_schemas_preserved(tmp_path, schema):
    import h5py

    cache = DiskCache(tmp_path)
    key = "a" * 64
    cache.save(key, make_idata([None]))
    path = cache._path(key)
    with h5py.File(path, "r+") as h5:
        attrs = h5["posterior"].attrs
        assert json.loads(attrs["_pymc_cache_entry_v3"])["schema"] == 3
        del attrs["_pymc_cache_entry_v3"]
        attrs[f"_pymc_cache_entry_v{schema}"] = json.dumps({"schema": schema, "key": key})
    before = path.read_bytes()
    with pytest.warns(CacheWarning):
        assert cache.load(key) is None
    assert cache.clear() == 0
    with pytest.raises(ValueError, match="schema-1/2"):
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
    for var in VARS:
        if var in result.sample_stats:
            assert_values(restored.sample_stats[var].values, result.sample_stats[var].values)
