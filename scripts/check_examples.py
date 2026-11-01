"""Execute documented Python snippets in isolation; don't execute shell install examples."""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = ("README.md", "getting-started.qmd", "guides/identity.qmd")


def main():
    for page in PAGES:
        blocks = re.findall(r"^```python\n(.*?)^```", (ROOT / page).read_text(), re.M | re.S)
        if not blocks:
            raise RuntimeError(f"No Python examples in {page}")
        with tempfile.TemporaryDirectory(prefix="pymc-cache-docs-") as directory:
            subprocess.run([sys.executable, "-c", "\n\n".join(blocks)], cwd=directory,
                           check=True)
        print(f"Executed {len(blocks)} Python snippets in {page}")


if __name__ == "__main__":
    main()
