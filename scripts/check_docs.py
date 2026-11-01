"""Check rendered pages, public API coverage and internal links."""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pymc_cache

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "_site"
PAGES = [ROOT / name for name in (
    "index.qmd", "getting-started.qmd", "reference.qmd", "contributing.qmd",
    "releasing.qmd",
)] + sorted((ROOT / "guides").glob("*.qmd")) + sorted((ROOT / "api").glob("*.qmd"))


def main() -> None:
    generated = {path.stem for path in (ROOT / "api").glob("*.qmd")} - {"index"}
    exported = set(pymc_cache.__all__) - {"__version__"}
    if generated != exported:
        raise SystemExit(f"Generated API differs from public exports: "
                         f"missing={sorted(exported - generated)}, "
                         f"unexpected={sorted(generated - exported)}")
    if not PAGES or not (SITE / "index.html").is_file():
        raise SystemExit("Quarto site is missing; run bash scripts/build-docs.sh")
    for source in PAGES:
        rendered = SITE / source.relative_to(ROOT).with_suffix(".html")
        if not rendered.is_file():
            raise SystemExit(f"Missing rendered page: {rendered}")
        html = rendered.read_text()
        if "<main" not in html:
            raise SystemExit(f"Missing page content: {rendered}")
        for href in re.findall(r'href="([^"]+)"', html):
            parsed = urlsplit(href)
            if parsed.scheme or parsed.netloc or href.startswith(("#", "mailto:", "javascript:")):
                continue
            path = unquote(parsed.path)
            if not path:
                continue
            target = (SITE / path.lstrip("/") if path.startswith("/") else
                      rendered.parent / path)
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                raise SystemExit(f"Broken link in {rendered}: {href}")
    print(f"Checked {len(PAGES)} rendered Quarto pages and local links")


if __name__ == "__main__":
    main()
