"""Versioned, fail-closed structural identity for PyMC sampling results.

A false miss wastes CPU. A false hit returns a posterior from a different model,
which is much worse; unsupported semantic state therefore disables automatic caching.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import sys
from collections.abc import Mapping

import numpy as np

SCHEMA_VERSION = 3


class UncacheableError(ValueError):
    """Raised when the identity of a model or sampling option cannot be established."""


def _qualified(obj) -> str:
    cls = obj if isinstance(obj, type) else type(obj)
    return cls.__module__ + "." + cls.__qualname__


def _array(value: np.ndarray):
    if value is None:
        return ["none-constant"]
    arr = np.asarray(value)
    if arr.dtype.hasobject:
        raise UncacheableError("object-dtype array has no safe content fingerprint")
    canonical = np.ascontiguousarray(arr)
    return ["array", str(arr.dtype), list(arr.shape), hashlib.sha256(canonical.tobytes()).hexdigest()]


def _token(value, *, sampler_key=None):
    """Typed, JSON-serializable value without repr or memory-address fallbacks."""
    if value is None:
        return ["none"]
    if isinstance(value, np.random.Generator):
        return ["generator", _qualified(value.bit_generator), _token(value.bit_generator.state)]
    if isinstance(value, np.random.RandomState):
        return ["randomstate", _token(value.get_state())]
    if isinstance(value, np.ndarray):
        return _array(value)
    if isinstance(value, np.generic):
        return ["numpy-scalar", str(value.dtype), _token(value.item())]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if math.isnan(value):
            return ["float", "NaN"]
        if math.isinf(value):
            return ["float", "+Inf" if value > 0 else "-Inf"]
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, bytes):
        return ["bytes", value.hex()]
    if isinstance(value, tuple):
        return ["tuple", [_token(v, sampler_key=sampler_key) for v in value]]
    if isinstance(value, list):
        return ["list", [_token(v, sampler_key=sampler_key) for v in value]]
    if isinstance(value, (set, frozenset)):
        items = [_token(v, sampler_key=sampler_key) for v in value]
        return ["set", sorted(items, key=lambda v: _json(v))]
    if isinstance(value, Mapping):
        items = [(_token(k, sampler_key=sampler_key), _token(v, sampler_key=sampler_key))
                 for k, v in value.items()]
        return ["mapping", sorted(items, key=lambda pair: _json(pair[0]))]
    if isinstance(value, type) and value.__module__.startswith(("pymc", "pytensor")):
        return ["builtin-class", _qualified(value)]
    if sampler_key is not None and callable(value):
        return ["explicit-sampler-key", sampler_key, _qualified(value)]
    raise UncacheableError(f"unsupported identity value: {_qualified(value)}")


def _json(value) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=True)


def _variable_type(var):
    t = var.type
    return ["variable", _qualified(t), getattr(t, "dtype", None),
            _token(tuple(getattr(t, "shape", ()) or ())),
            _token(tuple(getattr(t, "broadcastable", ()) or ()))]


def _op_property(value, model_key):
    """Preserve structured builtin Op properties, including nested scalar types."""
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_op_property(v, model_key) for v in value]]
    if type(value).__module__.startswith(("pytensor.", "pymc.")):
        return _op_token(value, model_key)
    return _token(value)


def _op_token(op, model_key):
    name = _qualified(op)
    if not type(op).__module__.startswith(("pytensor.", "pymc.")) and model_key is None:
        raise UncacheableError(f"custom Op {name} requires model_key")
    props = getattr(op, "__props__", ())
    try:
        recorded = [(p, _op_property(getattr(op, p), model_key)) for p in props]
    except (AttributeError, UncacheableError) as exc:
        if model_key is None:
            raise UncacheableError(f"Op {name} has unhashable properties") from exc
        recorded = [["custom-op", model_key]]
    scalar_op = getattr(op, "scalar_op", None)
    if scalar_op is not None:
        recorded.append(["scalar_op", _op_token(scalar_op, model_key)])
    inner = getattr(op, "inner_outputs", None)
    if inner is not None:
        recorded.append(["inner_graph", _graph(list(inner), model_key)])
    return [name, recorded]


def _graph(roots, model_key):
    try:
        from pytensor.compile.sharedvalue import SharedVariable
        from pytensor.graph.basic import Constant
        from pytensor.graph.traversal import graph_inputs, io_toposort
        from pytensor.tensor.random.type import RandomType
    except ImportError:
        from pytensor.compile.sharedvalue import SharedVariable
        from pytensor.graph.basic import Constant, graph_inputs, io_toposort
        from pytensor.tensor.random.type import RandomType

    roots = list(roots)
    inputs = list(graph_inputs(roots))
    nodes = io_toposort(inputs, roots)
    refs = {}
    entries = []

    def ref(var):
        if var in refs:
            return refs[var]
        if getattr(var, "owner", None) is not None:
            raise UncacheableError("non-topological graph node")
        index = len(entries)
        refs[var] = index
        record = ["input", _variable_type(var), getattr(var, "name", None)]
        if isinstance(var, Constant):
            record.append(["constant", _array(var.data)])
        elif isinstance(var, SharedVariable):
            if isinstance(var.type, RandomType):
                record.append(["model-construction-rng"])
            else:
                try:
                    record.append(["shared-data", _array(var.get_value(borrow=True))])
                except Exception as exc:
                    raise UncacheableError("cannot fingerprint shared model data") from exc
        entries.append(record)
        return index

    for node in nodes:
        parents = [ref(v) for v in node.inputs]
        op = _op_token(node.op, model_key)
        for position, var in enumerate(node.outputs):
            refs[var] = len(entries)
            entries.append(["node", op, parents, position, _variable_type(var),
                            getattr(var, "name", None)])
    return [entries, [ref(var) for var in roots]]


def data_digest(model) -> str:
    """Hash all reachable shared inputs and observed values by full contents."""
    try:
        roots = list(model.observed_RVs) + list(model.deterministics) + list(model.free_RVs)
        token = _graph(roots, model_key="data-only")
        # Graph still participates: a changed graph also changes this conservative digest.
        return hashlib.sha256(_json(token).encode()).hexdigest()
    except UncacheableError:
        raise
    except Exception as exc:
        raise UncacheableError("cannot fingerprint observed or shared model data") from exc


def model_digest(model, *, model_key: str | None = None) -> str:
    """Content hash of model graph, data, outputs and coordinates.

    Custom Python operations require a user-maintained ``model_key``. Their
    implementation cannot be inferred reliably from an Op's class name.
    """
    if model_key is not None and (not isinstance(model_key, str) or not model_key):
        raise ValueError("model_key must be a nonempty string")
    try:
        roots = [model.logp()]
        roots.extend(model.free_RVs)
        roots.extend(model.observed_RVs)
        roots.extend(model.deterministics)
        roots.extend(model.potentials)
        dims = getattr(model, "named_vars_to_dims", {}) or {}
        metadata = [
            [(name, _variable_type(var), _token(tuple(dims.get(name, ()))))
             for name, var in sorted(model.named_vars.items())],
            _token(model.coords or {}),
            [rv.name for rv in model.free_RVs],
            [rv.name for rv in model.deterministics],
            model_key,
        ]
        content = ["pymc-cache-model-v1", metadata, _graph(roots, model_key)]
        return hashlib.sha256(_json(content).encode()).hexdigest()
    except UncacheableError:
        raise
    except Exception as exc:
        raise UncacheableError(f"cannot safely fingerprint model: {exc}") from exc


def software_versions() -> dict[str, str]:
    """Exact library versions that affect posterior generation or file format."""
    names = ("pymc", "pytensor", "numpy", "scipy", "arviz", "arviz-base", "arviz-stats",
             "arviz-plots", "nutpie", "numpyro", "blackjax")
    found = {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    for name in names:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return found


def cache_key(model_identity: str | bytes, sample_kwargs: Mapping, *, operation="sample",
              versions: Mapping | None = None, namespace="", sampler_key: str | None = None) -> str:
    """Full SHA-256 identity of a model, sampler settings and environment."""
    if not isinstance(model_identity, (str, bytes)):
        raise TypeError("model_identity must be str or bytes")
    if not isinstance(operation, str) or not operation:
        raise ValueError("operation must be a nonempty string")
    if not isinstance(namespace, str):
        raise TypeError("namespace must be a string")
    if sampler_key is not None and (not isinstance(sampler_key, str) or not sampler_key):
        raise ValueError("sampler_key must be a nonempty string")
    if not isinstance(sample_kwargs, Mapping):
        raise TypeError("sample_kwargs must be a mapping")
    # Integration versions are additional identity, never a way to mask installed versions.
    effective_versions = [software_versions(), dict(versions or {})]
    content = ["pymc-cache-key", SCHEMA_VERSION, _token(model_identity), operation,
               namespace, _token(effective_versions),
               _token(sample_kwargs, sampler_key=sampler_key), sampler_key]
    return hashlib.sha256(_json(content).encode()).hexdigest()
