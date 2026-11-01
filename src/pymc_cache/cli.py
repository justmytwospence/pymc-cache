"""Inspect and manage only entries owned by pymc-cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pymc_cache.backends import DiskCache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pymc-cache", description=__doc__)
    parser.add_argument("--directory", "--dir", dest="directory", default=".pymc-cache",
                        help="cache directory (default: .pymc-cache)")
    sub = parser.add_subparsers(dest="command", required=True)
    listed = sub.add_parser("list", help="inspect owned cache entries")
    listed.add_argument("--json", action="store_true", help="machine-readable metadata")
    prune = sub.add_parser("prune", help="keep the newest entries per model")
    prune.add_argument("--keep", type=int, default=20)
    clear = sub.add_parser("clear", help="delete only entries owned by pymc-cache")
    clear.add_argument("--yes", action="store_true", required=True,
                       help="confirm permanent removal of owned entries")
    for command in (listed, prune, clear):
        command.add_argument("--directory", "--dir", dest="directory",
                             default=argparse.SUPPRESS, help="cache directory")
    args = parser.parse_args(argv)
    cache = DiskCache(Path(args.directory))
    if args.command == "list":
        entries = cache.entries()
        if args.json:
            print(json.dumps([{"key": e.key, "model": e.model_slug,
                               "operation": e.operation, "status": e.status,
                               "path": str(e.path), "size_bytes": e.size_bytes,
                               "created_at": e.created_at,
                               "software_versions": e.software_versions}
                              for e in entries], indent=2))
        else:
            for entry in entries:
                print(f"{entry.key}  {entry.model_slug}  {entry.status}  "
                      f"{entry.size_bytes} B  {entry.path}")
            print(f"{len(entries)} entries")
    elif args.command == "prune":
        print(f"Removed {cache.prune(args.keep)} entries")
    else:
        print(f"Removed {cache.clear()} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
