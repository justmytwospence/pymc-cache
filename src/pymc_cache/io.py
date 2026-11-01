"""Lossless-enough, NetCDF-safe storage for PyMC 5 and 6 inference objects."""
from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path

import numpy as np

_CODEC_ATTR = "_pymc_cache_attrs_codec_v1"
_TEXT_CODEC_ATTR = "_pymc_cache_text_codec_v1"
_TEXT_GROUPS = {"sample_stats", "warmup_sample_stats"}
_TEXT_VARS = {"divergence_message", "warning"}
_TEXT_MARKER = "json-tagged-v2"
_REAL_FLOAT_TYPES = frozenset(np.dtype(code).type for code in np.typecodes["Float"])


def group_names(idata) -> list[str]:
    """List group names for both ArviZ InferenceData and xarray DataTree."""
    raw = idata.groups() if callable(idata.groups) else idata.groups
    return [name.strip("/") for name in raw if name.strip("/")]


def get_group(idata, name):
    """Return a group without relying on ArviZ's changed groups interface."""
    try:
        group = getattr(idata, name)
        if group is not None:
            return group
    except (AttributeError, KeyError):
        pass
    try:
        return idata[name]
    except (KeyError, TypeError):
        return None


def group_attrs(idata, name=None):
    """Attributes of the root or a named group."""
    return getattr(idata if name is None else get_group(idata, name), "attrs", None)


def add_group(idata, name: str, group) -> None:
    """Attach an xarray Dataset group across ArviZ's two APIs."""
    import xarray as xr

    ds = group if isinstance(group, xr.Dataset) else group.to_dataset()
    if callable(getattr(idata, "add_groups", None)):
        idata.add_groups({name: ds})
    else:
        idata[name] = ds
    if name not in group_names(idata):
        raise RuntimeError(f"could not attach group {name!r}")


def _json_encode(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if isinstance(value, tuple):
        return {"__tuple__": [_json_encode(v) for v in value]}
    if isinstance(value, list):
        return [_json_encode(v) for v in value]
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("attribute dict keys must be strings")
        return {k: _json_encode(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported NetCDF attribute type {type(value).__name__}")


def _json_decode(value):
    if isinstance(value, dict):
        if set(value) == {"__ndarray__", "dtype"}:
            return np.asarray(value["__ndarray__"], dtype=value["dtype"])
        if set(value) == {"__tuple__"}:
            return tuple(_json_decode(v) for v in value["__tuple__"])
        return {k: _json_decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_decode(v) for v in value]
    return value


def _sanitize_attrs(attrs):
    if attrs is None:
        return
    if _CODEC_ATTR in attrs:
        raise ValueError(f"reserved cache attribute {_CODEC_ATTR} already exists")
    encoded = []
    for key, value in list(attrs.items()):
        if isinstance(value, (dict, tuple)) or value is None:
            attrs[key] = json.dumps(_json_encode(value), ensure_ascii=True)
            encoded.append(key)
        elif isinstance(value, list):
            # NetCDF rejects mixed/empty lists; encode them consistently.
            attrs[key] = json.dumps(_json_encode(value), ensure_ascii=True)
            encoded.append(key)
        elif isinstance(value, np.ndarray) and value.dtype.hasobject:
            raise TypeError(f"object-dtype attribute {key!r} cannot be stored safely")
        elif not isinstance(value, (str, bytes, int, float, bool, np.ndarray, np.number, np.bool_)):
            raise TypeError(f"unsupported NetCDF attribute {key!r}: {type(value).__name__}")
    if encoded:
        attrs[_CODEC_ATTR] = json.dumps(encoded)


def _restore_attrs(attrs):
    if attrs is None or _CODEC_ATTR not in attrs:
        return
    encoded = json.loads(attrs.pop(_CODEC_ATTR))
    for key in encoded:
        attrs[key] = _json_decode(json.loads(attrs[key]))


def _containers(idata):
    yield idata
    for name in group_names(idata):
        group = get_group(idata, name)
        yield group
        for collection in (getattr(group, "data_vars", None), getattr(group, "coords", None)):
            if collection is not None:
                for key in collection:
                    yield group[key]


def _encode_text(data, group, var):
    records = []
    for index in np.ndindex(data.shape):
        value = data.values[index]
        if value is None:
            record = [0]
        elif type(value) in (str, np.str_):
            record = [1, str(value)]
        elif type(value) in (bytes, np.bytes_):
            record = [2, base64.b64encode(bytes(value)).decode("ascii")]
        elif (type(value) is float or type(value) in _REAL_FLOAT_TYPES) and np.isnan(value):
            record = [3]
        else:
            raise TypeError(f"unsupported diagnostic text {group}.{var} at index {index}: "
                            f"{type(value).__name__}")
        records.append(json.dumps(record, ensure_ascii=True, allow_nan=False))
    width = max((len(record) for record in records), default=1)
    encoded = np.asarray(records, dtype=f"U{width}").reshape(data.shape)
    result = data.copy(data=encoded)
    result.attrs[_TEXT_CODEC_ATTR] = _TEXT_MARKER
    for key in ("dtype", "_FillValue", "missing_value"):
        result.encoding.pop(key, None)
    return result


def _reject_json_constant(value):
    raise ValueError("JSON constants are not diagnostic text records")


def _decode_text(data, group, var):
    marker = data.attrs[_TEXT_CODEC_ATTR]
    if (marker != _TEXT_MARKER and
            not (marker == "json-tagged" and var == "divergence_message")):
        raise ValueError(f"unknown diagnostic text codec for {group}.{var}")
    values = np.empty(data.shape, dtype=object)
    for index in np.ndindex(data.shape):
        try:
            raw = data.values[index]
            if not isinstance(raw, (str, np.str_)) or not raw.isascii():
                raise ValueError("text codec record must be an ASCII string")
            record = json.loads(raw, parse_constant=_reject_json_constant)
            if not isinstance(record, list) or not record or type(record[0]) is not int:
                raise ValueError("invalid text codec record")
            if record == [0]:
                value = None
            elif len(record) == 2 and record[0] == 1 and type(record[1]) is str:
                value = record[1]
            elif len(record) == 2 and record[0] == 2 and type(record[1]) is str:
                value = base64.b64decode(record[1], validate=True)
                if base64.b64encode(value).decode("ascii") != record[1]:
                    raise ValueError("non-canonical base64")
            elif record == [3] and marker == _TEXT_MARKER:
                value = float("nan")
            else:
                raise ValueError("invalid text codec record")
            values[index] = value
        except (ValueError, TypeError, binascii.Error) as exc:
            raise ValueError(f"malformed diagnostic text codec record {group}.{var} "
                             f"at index {index}") from exc
    result = data.copy(data=values)
    # xarray normalizes mixed object None/NaN to NaN while constructing a
    # DataArray. Restore the decoded entries after construction, before the
    # caller attaches the array to its group.
    result.values[...] = values
    result.attrs.pop(_TEXT_CODEC_ATTR)
    for key in ("dtype", "_FillValue", "missing_value"):
        result.encoding.pop(key, None)
    return result


def sanitize_inference_data(idata, copy: bool = True):
    """Return an independent copy whose attrs and data can be written to NetCDF.

    Diagnostic text uses a tagged codec in sample_stats and warmup_sample_stats;
    explicitly numeric object arrays in SMC sample_stats are coerced. Other
    object arrays are rejected.
    """
    result = __import__("copy").deepcopy(idata) if copy else idata
    for container in _containers(result):
        _sanitize_attrs(getattr(container, "attrs", None))
    for name in group_names(result):
        group = get_group(result, name)
        for var in list(getattr(group, "data_vars", ())):
            data = group[var]
            if _TEXT_CODEC_ATTR in data.attrs:
                raise ValueError(f"reserved cache attribute {_TEXT_CODEC_ATTR} already exists")
            if name in _TEXT_GROUPS and var in _TEXT_VARS and data.dtype == object:
                group[var] = _encode_text(data, name, var)
                continue
            if data.dtype == object:
                if name == "sample_stats" and var in {
                    "beta", "accept_rate", "log_marginal_likelihood"
                }:
                    group[var] = data.astype("float64")
                else:
                    raise TypeError(f"object-dtype variable {name}.{var} cannot be cached")
    return result


def load_all(idata) -> None:
    """Strictly load every group: never return lazy data backed by a closed file."""
    for name in group_names(idata):
        group = get_group(idata, name)
        loader = getattr(group, "load", None)
        if loader is not None:
            loader()


def load_inference_data(path: str | Path):
    """Load and close a NetCDF cache entry, restoring typed attribute values."""
    import arviz as az

    idata = az.from_netcdf(str(path), engine="h5netcdf")
    try:
        load_all(idata)
    finally:
        close = getattr(idata, "close", None)
        if callable(close):
            close()
    for container in _containers(idata):
        _restore_attrs(getattr(container, "attrs", None))
    for name in group_names(idata):
        group = get_group(idata, name)
        for var in list(group.data_vars):
            data = group[var]
            if _TEXT_CODEC_ATTR in data.attrs:
                if name not in _TEXT_GROUPS or var not in _TEXT_VARS:
                    raise ValueError(f"unknown diagnostic text codec for {name}.{var}")
                group[var] = _decode_text(data, name, var)
    return idata


def save_inference_data(idata, path: str | Path, *, attrs: dict | None = None) -> None:
    """Save a snapshot while leaving every caller-owned attribute unchanged."""
    snapshot = sanitize_inference_data(idata, copy=True)
    if attrs:
        # A root attr is not written by ArviZ 0.x; posterior group is portable.
        group = get_group(snapshot, "posterior")
        if group is None:
            raise ValueError("cannot persist an inference result without a posterior")
        group.attrs.update(attrs)
    snapshot.to_netcdf(str(path), engine="h5netcdf")
