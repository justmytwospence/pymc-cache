"""Explicit (non-monkeypatching) PyMC sampling with reusable results."""
from __future__ import annotations

import copy
import importlib.util
import os
import warnings

from pymc_cache.policy import CachePolicy, CacheWarning, report_from_idata
from pymc_cache.session import CacheSession


def _model(model):
    import pymc as pm

    return model if model is not None else pm.modelcontext(None)


def _native_kwargs(model, kwargs: dict, *, operation: str) -> tuple[dict, dict, str]:
    import pymc as pm

    call = dict(kwargs)
    key = dict(kwargs)
    key.pop("progressbar", None)
    if operation == "smc":
        call.setdefault("draws", 2000)
        call.setdefault("chains", max(2, min(4, os.cpu_count() or 1)))
        key.update({"draws": call["draws"], "chains": call["chains"]})
        kernel = call.get("kernel")
        if kernel is not None and getattr(kernel, "__module__", "").startswith("pymc.smc"):
            key["kernel"] = f"{kernel.__module__}.{kernel.__qualname__}"
        return call, key, "smc"
    call.setdefault("draws", 1000)
    cores = call.get("cores") or min(4, os.cpu_count() or 1)
    call.setdefault("cores", cores)
    call.setdefault("chains", max(2, min(4, cores)))
    sampler = call.get("nuts_sampler")
    if sampler is None:
        if int(pm.__version__.split(".")[0]) >= 6 and importlib.util.find_spec("nutpie"):
            discrete = any(str(rv.dtype).startswith(("int", "uint", "bool"))
                           for rv in model.free_RVs)
            sampler = "pymc" if discrete else "nutpie"
        else:
            sampler = "pymc"
        call["nuts_sampler"] = sampler
    call["tune"] = call.get("tune") if call.get("tune") is not None else (
        400 if int(pm.__version__.split(".")[0]) >= 6 and sampler == "nutpie" else 1000
    )
    key.update({k: call[k] for k in ("draws", "tune", "chains", "cores", "nuts_sampler")})
    return call, key, sampler


def _runner(model, call_kwargs, sample_fn, *, operation):
    import pymc as pm

    func = sample_fn if sample_fn is not None else (
        pm.sample_smc if operation == "smc" else pm.sample
    )
    with model:
        return func(model=model, **call_kwargs)


def _call(*, model=None, cache=True, overwrite=False, policy=None, namespace="",
          model_key=None, sampler_key=None, sample_fn=None, operation="sample", **sample_kwargs):
    model = _model(model)
    if policy is None:
        policy = CachePolicy()
    if not isinstance(policy, CachePolicy):
        raise TypeError("policy must be CachePolicy")
    call = dict(sample_kwargs)
    key_kwargs = dict(call)
    sampler_identity = operation
    versions = None
    hook = getattr(sample_fn, "__pymc_cache_normalize__", None)
    if sample_fn is not None and hook is not None:
        normalized = hook(model, copy.deepcopy(call))
        if not isinstance(normalized, dict) or set(normalized) != {
            "call_kwargs", "key_kwargs", "sampler_identity", "versions"
        }:
            raise TypeError("sample_fn normalization must return call_kwargs, key_kwargs, "
                            "sampler_identity, versions")
        call = dict(normalized["call_kwargs"])
        key_kwargs = dict(normalized["key_kwargs"])
        sampler_identity = normalized["sampler_identity"]
        versions = dict(normalized["versions"])
    elif sample_fn is None:
        call, key_kwargs, sampler_identity = _native_kwargs(model, call, operation=operation)
    elif cache is not False and sampler_key is None:
        warnings.warn("sampling result is not cacheable: sample_fn requires sampler_key or "
                      "__pymc_cache_normalize__", CacheWarning, stacklevel=2)
        cache = False
    if cache is not False and call.get("return_inferencedata") is False:
        warnings.warn("sampling result is not cacheable: return_inferencedata=False",
                      CacheWarning, stacklevel=2)
        cache = False
    session = CacheSession(cache, model=model, policy=policy, namespace=namespace,
                           model_key=model_key, sampler_key=sampler_key)
    lookup = session.lookup(key_kwargs, operation=operation, overwrite=overwrite,
                            versions=versions, sampler_identity=sampler_identity)
    if lookup.hit:
        return lookup.result
    result = _runner(model, call, sample_fn, operation=operation)
    report = report_from_idata(result)
    status = report.get("status", "complete") if report else "complete"
    session.store(lookup, result, status=status, effective_sampler=sampler_identity)
    return result


def sample(*, model=None, cache=True, overwrite=False, policy=None, namespace="",
           model_key=None, sampler_key=None, sample_fn=None, **sample_kwargs):
    """Run ``pm.sample`` (or a trusted runner), reusing a matching result.

    No global PyMC function is replaced. ``cache='disk'`` persists under
    ``.pymc-cache``; ``cache=False`` always samples anew. A user-supplied
    sample_fn must opt into safe identity or supply a trusted sampler_key.
    """
    return _call(model=model, cache=cache, overwrite=overwrite, policy=policy,
                 namespace=namespace, model_key=model_key, sampler_key=sampler_key,
                 sample_fn=sample_fn, operation="sample", **sample_kwargs)


def sample_smc(*, model=None, cache=True, overwrite=False, policy=None, namespace="",
               model_key=None, sampler_key=None, **sample_kwargs):
    """Run and cache ``pm.sample_smc`` without conflating it with NUTS."""
    return _call(model=model, cache=cache, overwrite=overwrite, policy=policy,
                 namespace=namespace, model_key=model_key, sampler_key=sampler_key,
                 operation="smc", **sample_kwargs)
