"""Verify archive scope and run installed wheel/sdist smokes in disposable venvs."""
from __future__ import annotations

import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
REQUIRED = {"src/pymc_cache/identity.py", "src/pymc_cache/backends.py",
            "tests/test_cache.py", "scripts/smoke_test.py", "README.md", "LICENSE",
            "_quarto.yml", "guides/identity.qmd", "api/index.qmd",
            "src/pymc_cache/py.typed"}
FORBIDDEN = (".extraction-source", ".git/", ".venv/", "_site/", ".nc", ".coverage")


def check(names: list[str], *, wheel: bool) -> None:
    if not names or any(any(part in name for part in FORBIDDEN) for name in names):
        raise ValueError("empty archive or forbidden private/build file in artifact")
    normalized = [name if wheel else "/".join(name.split("/")[1:]) for name in names]
    if wheel:
        if not {"pymc_cache/identity.py", "pymc_cache/py.typed"}.issubset(normalized) or not any(
            n.endswith(".dist-info/METADATA") for n in normalized
        ):
            raise ValueError("wheel is missing code or package metadata")
    elif not REQUIRED.issubset(normalized):
        raise ValueError(f"sdist missing {sorted(REQUIRED - set(normalized))}")
    if any("cloudposterior" in name or "pymc_monitor" in name for name in normalized):
        raise ValueError("artifact contains code from another project")


def main() -> None:
    wheels = sorted(DIST.glob("pymc_cache-0.1.0-*.whl"))
    sdists = sorted(DIST.glob("pymc_cache-0.1.0.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("expected exactly one 0.1.0 wheel and sdist in dist/")
    wheel, sdist = wheels[0], sdists[0]
    with zipfile.ZipFile(wheel) as archive:
        check(archive.namelist(), wheel=True)
        metadata_name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        metadata = archive.read(metadata_name).decode()
        if "Name: pymc-cache" not in metadata or "Version: 0.1.0" not in metadata:
            raise ValueError("incorrect wheel distribution metadata")
        if any(name in metadata.lower() for name in ("requires-dist: pymc-monitor",
                                                   "requires-dist: cloudposterior")):
            raise ValueError("foreign runtime dependency in wheel metadata")
    with tarfile.open(sdist, "r:gz") as archive:
        check(archive.getnames(), wheel=False)
    scratch = ROOT / "build"
    scratch.mkdir(exist_ok=True)
    for artifact in (wheel, sdist):
        with tempfile.TemporaryDirectory(prefix="pymc-cache-smoke-", dir=scratch) as temp:
            venv = Path(temp) / "venv"
            subprocess.run(["uv", "venv", str(venv), "--python", sys.executable], check=True)
            python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
            subprocess.run(["uv", "pip", "install", "--python", str(python), str(artifact)],
                           check=True)
            subprocess.run([str(python), str(ROOT / "scripts/smoke_test.py")], cwd=temp,
                           check=True)
            print(f"Verified isolated installation: {artifact.name}")


if __name__ == "__main__":
    main()
