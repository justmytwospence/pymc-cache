"""Cache backends: bounded in-memory LRU and atomic, inspectable NetCDF disk storage."""
from __future__ import annotations

import copy
import json
import os
import re
import threading
import uuid
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pymc_cache.io import get_group, load_inference_data, save_inference_data
from pymc_cache.naming import model_slug, params_label
from pymc_cache.policy import CacheWarning

_ENTRY_SCHEMA = 1
_ENTRY_META = "_pymc_cache_entry_v1"


class CacheBackend(Protocol):
    """Minimal backend interface; extra management operations are optional."""

    def load(self, key: str, **kwargs): ...

    def save(self, key: str, idata, **kwargs) -> None: ...


class MemoryCache:
    """Thread-safe, copy-isolated LRU of posterior results."""

    def __init__(self, max_entries: int = 8):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._store: OrderedDict[str, object] = OrderedDict()
        self._lock = threading.RLock()

    def load(self, key: str, **kwargs):
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return copy.deepcopy(self._store[key])

    def save(self, key: str, idata, **kwargs) -> None:
        snapshot = copy.deepcopy(idata)
        with self._lock:
            self._store[key] = snapshot
            self._store.move_to_end(key)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def clear(self) -> int:
        """Discard stored entries and return the number removed."""
        with self._lock:
            removed = len(self._store)
            self._store.clear()
            return removed

    def __len__(self):
        with self._lock:
            return len(self._store)


@dataclass(frozen=True)
class CacheEntry:
    """Metadata for an owned, valid disk entry (without loading posterior arrays)."""

    key: str
    model_slug: str
    operation: str
    created_at: float
    size_bytes: int
    status: str
    path: Path
    software_versions: dict[str, str]


class DiskCache:
    """One-file, full-key-verified atomic NetCDF result cache.

    Entries live at ``<root>/<model>/<parameters>-<full-sha256>.nc``. Contents
    have an independently checked full key and schema; legacy files are never
    reused. The directory is a trusted local cache, not an untrusted file reader.
    """

    def __init__(self, base_dir: str | Path | None = None, model=None, *,
                 max_entries_per_model: int = 20):
        if (isinstance(max_entries_per_model, bool) or
                not isinstance(max_entries_per_model, int) or max_entries_per_model < 1):
            raise ValueError("max_entries_per_model must be a positive integer")
        self._base = Path(base_dir) if base_dir is not None else Path(".pymc-cache")
        self._model_slug = model_slug(model)
        self._keep = max_entries_per_model

    def _path(self, key: str, sample_kwargs: dict | None = None) -> Path:
        if not isinstance(key, str) or not re.fullmatch(r"[a-f0-9]{64}", key):
            raise ValueError("cache key must be a full 64-character SHA-256 hexadecimal digest")
        path = self._base / self._model_slug / f"{params_label(sample_kwargs)}-{key}.nc"
        if not path.resolve().is_relative_to(self._base.resolve()):
            raise ValueError("cache path escapes the cache root")
        return path

    @staticmethod
    def _read_metadata(path: Path) -> dict:
        import h5py

        with h5py.File(path, "r") as f:
            # Both ArviZ 0 and ArviZ 1 write group attrs to the HDF5 group.
            raw = f["posterior"].attrs[_ENTRY_META]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        if data.get("schema") != _ENTRY_SCHEMA:
            raise ValueError("unknown cache schema")
        return data

    def load(self, key: str, sample_kwargs: dict | None = None):
        path = self._path(key, sample_kwargs)
        if self._base.is_symlink() or path.parent.is_symlink() or path.is_symlink() or not path.exists():
            return None
        try:
            metadata = self._read_metadata(path)
            if metadata.get("key") != key or metadata.get("status") not in {"complete", "met"}:
                raise ValueError("cache entry identity or status does not match")
            idata = load_inference_data(path)
            post = get_group(idata, "posterior")
            post.attrs.pop(_ENTRY_META, None)
            return idata
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
            warnings.warn(f"pymc-cache: ignoring unreadable cache file {path} ({exc})",
                          CacheWarning, stacklevel=2)
            return None

    def save(self, key: str, idata, sample_kwargs: dict | None = None, *,
             operation="sample", status="complete", software_versions=None) -> None:
        path = self._path(key, sample_kwargs)
        if status not in ("complete", "met"):
            raise ValueError("only complete or met results may be stored")
        if self._base.is_symlink() or path.parent.is_symlink() or path.is_symlink():
            raise ValueError("refusing to write through a cache symlink")
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._base.is_symlink() or path.parent.is_symlink():
            raise ValueError("refusing to write through a cache symlink")
        import time

        metadata = {"key": key, "schema": _ENTRY_SCHEMA, "status": status,
                    "operation": operation, "created_at": time.time(),
                    "software_versions": software_versions or {}}
        tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            save_inference_data(idata, tmp, attrs={_ENTRY_META: json.dumps(metadata)})
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        self.prune(self._keep)

    def entries(self) -> list[CacheEntry]:
        """List this format's entries without materializing posterior data."""
        results = []
        if not self._base.exists() or self._base.is_symlink():
            return results
        for directory in self._base.iterdir():
            if not directory.is_dir() or directory.is_symlink():
                continue
            for path in directory.glob("*.nc"):
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    metadata = self._read_metadata(path)
                    key = metadata["key"]
                    if not re.fullmatch(r"[0-9a-f]{64}", key) or not path.name.endswith(
                        f"-{key}.nc"
                    ):
                        continue
                    results.append(CacheEntry(key, directory.name, metadata["operation"],
                                              float(metadata["created_at"]), path.stat().st_size,
                                              metadata["status"], path,
                                              metadata.get("software_versions", {})))
                except (OSError, ValueError, KeyError, TypeError):
                    continue
        return sorted(results, key=lambda entry: entry.created_at, reverse=True)

    def delete(self, key: str) -> bool:
        """Remove matching owned entries, never arbitrary files or symlinks."""
        matches = [entry for entry in self.entries() if entry.key == key]
        for entry in matches:
            entry.path.unlink()
        return bool(matches)

    def prune(self, keep: int | None = None) -> int:
        """Keep the N newest entries per model; return count removed."""
        if keep is None:
            keep = self._keep
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 0:
            raise ValueError("keep must be a nonnegative integer")
        counts: dict[str, int] = {}
        removed = 0
        for entry in self.entries():
            current = counts.get(entry.model_slug, 0)
            counts[entry.model_slug] = current + 1
            if current >= keep:
                entry.path.unlink()
                removed += 1
        return removed

    def clear(self) -> int:
        """Delete only format-owned entries, preserving other files."""
        entries = self.entries()
        for entry in entries:
            entry.path.unlink()
        return len(entries)


_default_memory_cache: MemoryCache | None = None
_default_lock = threading.Lock()


def get_default_cache() -> MemoryCache:
    """Lazily construct the process-wide memory cache."""
    global _default_memory_cache
    with _default_lock:
        if _default_memory_cache is None:
            _default_memory_cache = MemoryCache()
        return _default_memory_cache


def resolve_cache(cache_arg, model=None, *, default_directory=".pymc-cache") -> CacheBackend | None:
    """Resolve bool, memory/disk alias, directory, or custom backend."""
    if cache_arg is False:
        return None
    if cache_arg is True or cache_arg == "memory":
        return get_default_cache()
    if cache_arg == "disk":
        return DiskCache(default_directory, model=model)
    if isinstance(cache_arg, (str, Path)):
        return DiskCache(cache_arg, model=model)
    if callable(getattr(cache_arg, "load", None)) and callable(getattr(cache_arg, "save", None)):
        return cache_arg
    raise TypeError(f"cache must be bool | 'disk' | str | Path | CacheBackend; got {cache_arg!r}")


def cleanup_cache(base_dir: str | Path | None = None) -> int:
    """Remove only new-format entries in a directory; do not delete the root."""
    return DiskCache(base_dir).clear()
