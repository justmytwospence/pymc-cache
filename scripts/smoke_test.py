"""Exercise the installed distribution, not the source tree (run from any directory)."""
from __future__ import annotations

import sys
import tempfile
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np
import pymc as pm

import pymc_cache as pc


def main() -> None:
    dist = distribution("pymc-cache")
    installed = Path(pc.__file__).resolve()
    assert installed.is_relative_to(Path(sys.prefix).resolve()), installed
    assert "src/pymc_cache" not in str(installed)
    assert (installed.parent / "py.typed").is_file()
    assert version("pymc-cache") == pc.__version__
    assert all(getattr(pc, name, None) is not None for name in pc.__all__)
    assert len(pc.__all__) == len(set(pc.__all__))
    assert not any("pymc-monitor" in req.lower() or "cloudposterior" in req.lower()
                   for req in dist.requires or [])
    with pm.Model() as model:
        pm.Normal("x", 0, 1)
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(model=model, draws=2, tune=2, chains=1, cores=1,
                  random_seed=19, progressbar=False, cache=Path(tmp))
        first = pc.sample(**kw)
        second = pc.sample(**kw)
        np.testing.assert_array_equal(first.posterior["x"], second.posterior["x"])
        assert first is not second
        assert len(pc.DiskCache(tmp).entries()) == 1
        assert pc.cleanup_cache(tmp) == 1
    print("Installed pymc-cache smoke passed")


if __name__ == "__main__":
    main()
