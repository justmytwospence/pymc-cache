import subprocess
import sys

import numpy as np
import pymc as pm
import pytest

from pymc_cache import UncacheableError, cache_key, model_digest


def model(prior=1, observed=(1., 2.), deterministic=1, coords=("a", "b")):
    with pm.Model(coords={"row": coords}) as m:
        x = pm.Normal("x", 0, prior)
        pm.Deterministic("pred", x * deterministic)
        pm.Normal("obs", x, 1, observed=np.asarray(observed))
    return m


def test_graph_sensitive():
    original = model_digest(model())
    assert original == model_digest(model())
    for other in [model(prior=2), model(observed=(2., 1.)), model(deterministic=2),
                  model(coords=("b", "a"))]:
        assert model_digest(other) != original


def test_cross_process_stability():
    script = "import pymc as pm; from pymc_cache import model_digest;\nwith pm.Model() as m:\n x=pm.Normal('x',0,1)\nprint(model_digest(m))"
    def digest():
        return subprocess.check_output([sys.executable, "-c", script], text=True).strip().splitlines()[-1]
    assert digest() == digest()


def test_data_mutation():
    with pm.Model() as m:
        x = pm.Data("x", np.array([1., 2.]))
        mu = pm.Normal("mu", 0, 1)
        pm.Normal("y", mu * x, 1, observed=np.array([1., 2.]))
    first = model_digest(m)
    with m:
        pm.set_data({"x": np.array([2., 1.])})
    assert model_digest(m) != first


def test_key_rng_versions_and_callables():
    assert cache_key("m", {"seed": np.random.default_rng(1)}) != cache_key("m", {"seed": np.random.default_rng(2)})
    assert cache_key("m", {"seed": np.random.default_rng(1)}) == cache_key("m", {"seed": np.random.default_rng(1)})
    assert cache_key("m", {"k": np.array([1., 2.])}) != cache_key("m", {"k": np.array([2., 1.])})
    assert cache_key("m", {"ab": "c"}) != cache_key("m", {"a": "bc"})
    with pytest.raises(UncacheableError):
        cache_key("m", {"callback": lambda: None})
    assert cache_key("m", {"callback": lambda: None}, sampler_key="callback-v1")
    with pytest.raises(UncacheableError):
        cache_key("m", {"x": np.array([object()], dtype=object)})


def test_no_fallback_for_invalid_model():
    with pytest.raises(UncacheableError):
        model_digest(object())
