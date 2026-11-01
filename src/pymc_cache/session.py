"""Preflight lookup and delayed storage for local and remote sampling."""
from __future__ import annotations

import copy
import inspect
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

from pymc_cache.backends import CacheBackend, resolve_cache
from pymc_cache.identity import UncacheableError, cache_key, model_digest
from pymc_cache.policy import CachePolicy, CacheWarning, eligible, report_from_idata


def _backend_kwargs(method, metadata: dict) -> dict:
    """Pass optional metadata only when a backend accepts it as keywords.

    If inspection is unavailable, use the minimal positional load/save protocol.
    Do not probe by calling the method: a failed write may already have effects.
    """
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return {}
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return metadata
    return {name: value for name, value in metadata.items()
            if name in parameters and parameters[name].kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}


@dataclass(frozen=True)
class CacheLookup:
    """Immutable record of a pre-sampling cache lookup.

    ``result`` is non-None only for a hit; ``key`` is usable by a later store.
    ``normalized_kwargs`` is a snapshot, including pre-sampling RNG state.
    """

    backend: CacheBackend | None
    key: str | None
    hit: bool
    result: object | None
    normalized_kwargs: dict
    disable_reason: str | None
    operation: str
    versions: dict
    sampler_identity: str | None
    pre_run_model_digest: str | None


class CacheSession:
    """Compute identity and check a cache before beginning costly sampling.

    Reuse the returned CacheLookup with :meth:`store` after sampling completes.
    This API lets a remote caller avoid provisioning a container on a hit.
    """

    def __init__(self, cache=True, *, model=None, policy=None, namespace="",
                 model_key=None, sampler_key=None):
        self.model = model
        self.backend = resolve_cache(cache, model=model)
        self.policy = policy if policy is not None else CachePolicy()
        if not isinstance(self.policy, CachePolicy):
            raise TypeError("policy must be CachePolicy")
        self.namespace = namespace
        self.model_key = model_key
        self.sampler_key = sampler_key

    def lookup(self, sample_kwargs: Mapping, *, operation="sample", overwrite=False,
               versions: Mapping | None = None, sampler_identity: str | None = None) -> CacheLookup:
        """Return a hit, a storable miss, or a safe uncacheable miss."""
        if not isinstance(sample_kwargs, Mapping):
            raise TypeError("sample_kwargs must be a mapping")
        kwargs = copy.deepcopy(dict(sample_kwargs))
        base = (self.backend, None, False, None, kwargs, None, operation,
                dict(versions or {}), sampler_identity, None)
        if self.backend is None:
            return CacheLookup(*base)
        if not self.policy.reuse_unseeded and kwargs.get("random_seed") is None:
            return CacheLookup(self.backend, None, False, None, kwargs,
                               "unseeded reuse disabled", operation, dict(versions or {}),
                               sampler_identity, None)
        if self.model is None:
            raise ValueError("model is required for cache lookups")
        try:
            ident = model_digest(self.model, model_key=self.model_key)
            key = cache_key(ident, {"kwargs": kwargs, "sampler_identity": sampler_identity},
                            operation=operation, versions=versions, namespace=self.namespace,
                            sampler_key=self.sampler_key)
        except UncacheableError as exc:
            warnings.warn(f"sampling result is not cacheable: {exc}", CacheWarning, stacklevel=2)
            return CacheLookup(self.backend, None, False, None, kwargs, str(exc), operation,
                               dict(versions or {}), sampler_identity, None)
        try:
            cached = None if overwrite else self.backend.load(
                key, **_backend_kwargs(self.backend.load, {"sample_kwargs": kwargs}))
        except Exception as exc:
            warnings.warn(f"cache read failed; sampling anew: {exc}", CacheWarning, stacklevel=2)
            cached = None
        return CacheLookup(self.backend, key, cached is not None, cached, kwargs, None,
                           operation, dict(versions or {}), sampler_identity, ident)

    def store(self, lookup: CacheLookup, idata, *, status="complete",
              effective_sampler=None) -> bool:
        """Publish only if graph and result still match the preflight identity.

        A reported sampler fallback is not stored under the preflight key; callers
        must preflight a new identity if they want to cache that result.
        """
        if lookup.backend is None or lookup.key is None or lookup.hit:
            return False
        if self.model is None:
            return False
        try:
            if model_digest(self.model, model_key=self.model_key) != lookup.pre_run_model_digest:
                warnings.warn("model changed while sampling; result was not cached", CacheWarning,
                              stacklevel=2)
                return False
        except UncacheableError as exc:
            warnings.warn(f"sampling result is not cacheable: {exc}", CacheWarning, stacklevel=2)
            return False
        report = report_from_idata(idata)
        if (report is not None and "effective_sampler" in report and
                report["effective_sampler"] != lookup.sampler_identity):
            warnings.warn("reported effective sampler differs from preflight identity; "
                          "result was not cached", CacheWarning, stacklevel=2)
            return False
        if not eligible(idata, status=status, kwargs=lookup.normalized_kwargs,
                        operation=lookup.operation,
                        effective_sampler=(report["effective_sampler"] if report is not None
                                           and "effective_sampler" in report
                                           else effective_sampler),
                        sampler_identity=lookup.sampler_identity):
            return False
        try:
            metadata = {"sample_kwargs": lookup.normalized_kwargs,
                        "operation": lookup.operation, "status": status,
                        "software_versions": lookup.versions}
            lookup.backend.save(lookup.key, idata,
                                **_backend_kwargs(lookup.backend.save, metadata))
        except Exception as exc:
            warnings.warn(f"sampling succeeded but cache write failed: {exc}", CacheWarning,
                          stacklevel=2)
            return False
        return True
