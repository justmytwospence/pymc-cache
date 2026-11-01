"""Result eligibility: never turn an interrupted or diagnostically failed run into a hit."""
from __future__ import annotations

import json
from dataclasses import dataclass

from pymc_cache.io import get_group


class CacheWarning(UserWarning):
    """A cache was bypassed or an entry could not be persisted."""


@dataclass(frozen=True)
class CachePolicy:
    """Select whether unseeded sampling may reuse a previous run."""

    reuse_unseeded: bool = True


def report_from_idata(idata) -> dict | None:
    post = get_group(idata, "posterior")
    attrs = getattr(post, "attrs", {}) if post is not None else {}
    for name in ("pymc_monitor_convergence", "cloudposterior_convergence"):
        payload = attrs.get(name)
        if not payload:
            continue
        try:
            result = json.loads(payload) if isinstance(payload, str) else payload
            if isinstance(result, dict):
                return result
        except (TypeError, ValueError):
            return None
    return None


def eligible(idata, *, status: str, kwargs: dict, operation="sample",
             effective_sampler=None, sampler_identity=None) -> bool:
    """Check posterior completeness and report before publishing a result."""
    if status not in ("complete", "met"):
        return False
    post = get_group(idata, "posterior")
    if post is None or not all(dim in post.sizes for dim in ("chain", "draw")):
        return False
    requested_chains = kwargs.get("chains")
    if requested_chains is not None and post.sizes["chain"] != requested_chains:
        return False
    if effective_sampler is not None and sampler_identity is not None:
        if effective_sampler != sampler_identity:
            return False
    report = report_from_idata(idata)
    adaptive = kwargs.get("until") not in (None, False)
    if report:
        actual = report.get("status")
        if actual not in ("complete", "met"):
            return False
        if adaptive and (actual != "met" or report.get("met") is not True):
            return False
    elif adaptive or status == "met":
        return False
    expected_draws = kwargs.get("draws")
    if status == "complete" and expected_draws is not None:
        if post.sizes["draw"] != expected_draws:
            return False
    return True
