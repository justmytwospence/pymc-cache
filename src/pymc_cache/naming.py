"""Human-readable, path-safe names; correctness is provided by full cache keys."""
from __future__ import annotations

import re


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")[:80] or "unnamed"


def model_slug(model) -> str:
    """A descriptive directory name, never part of the machine identity."""
    name = getattr(model, "name", None)
    if not name:
        variables = getattr(model, "free_RVs", ())
        name = "_".join(str(rv.name).split("::")[-1] for rv in variables[:4])
        if len(variables) > 4:
            name += f"_plus{len(variables) - 4}"
    return slugify(name or "unnamed")


def params_label(kwargs: dict | None) -> str:
    """Short, sanitized parameter hint for browsing disk entries."""
    kwargs = kwargs or {}
    parts = [f"{name}{kwargs[name]}" for name in (
        "draws", "tune", "chains", "cores", "nuts_sampler", "target_accept"
    ) if kwargs.get(name) is not None and not (name == "nuts_sampler" and kwargs[name] == "pymc")]
    return slugify("_".join(parts) if parts else "default")
