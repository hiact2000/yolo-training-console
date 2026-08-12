"""Render cam8_holdout_experiment_report.md into a self-contained HTML report.

Figures are inlined as base64 data URIs so the HTML can be mailed or opened from
anywhere, which matters because experiments/**/*.png is gitignored -- the HTML is
the portable copy of the charts.
"""

from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

import markdown

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments" / "cam8_holdout_yolov8s"

STYLE = """
:root {
  --surface: #fcfcfb; --plane: #f9f9f7; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --rule: #c3c2b7; --accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface: #1a1a19; --plane: #0d0d0d; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --rule: #383835; --accent: #3987e5;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 3rem 1.25rem 6rem; background: var(--plane); color: var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height: 1.65;
  font-size: 16px;
}
main { max-width: 62rem; margin: 0 auto; background: var(--surface); padding: 2.5rem 3rem 4rem;
  border: 1px solid var(--grid); border-radius: 10px; }
h1 { font-size: 1.9rem; line-height: 1.25; margin: 0 0 1.5rem; letter-spacing: -0.01em; }
h2 { font-size: 1.35rem; margin: 2.75rem 0 0.85rem; padding-top: 1.25rem;
  border-top: 1px solid var(--grid); letter-spacing: -0.005em; }
h3 { font-size: 1.08rem; margin: 1.9rem 0 0.6rem; color: var(--ink); }
h4 { font-size: 0.98rem; margin: 1.4rem 0 0.4rem; color: var(--ink-2); }
p, li { color: var(--ink-2); }
strong { color: var(--ink); }
a { color: var(--accent); }
code { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; font-size: 0.86em;
  background: var(--plane); padding: 0.12em 0.38em; border-radius: 4px; color: var(--ink); }
pre { background: var(--plane); padding: 1rem 1.15rem; border-radius: 8px;
  border: 1px solid var(--grid); overflow-x: auto; }
pre code { background: none; padding: 0; }
.table-wrap { overflow-x: auto; margin: 1.1rem 0; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem;
  font-variant-numeric: tabular-nums; }
th, td { padding: 0.5rem 0.7rem; text-align: left; border-bottom: 1px solid var(--grid); }
th { color: var(--ink); font-weight: 600; border-bottom: 1.5px solid var(--rule);
  white-space: nowrap; }
td { color: var(--ink-2); }
td:not(:first-child), th:not(:first-child) { text-align: right; }
tbody tr:last-child td { border-bottom: none; }
figure { margin: 1.6rem 0; }
img { max-width: 100%; height: auto; display: block; border-radius: 8px;
  border: 1px solid var(--grid); background: #fcfcfb; }
figcaption { font-size: 0.84rem; color: var(--muted); margin-top: 0.5rem; }
blockquote { margin: 1.2rem 0; padding: 0.65rem 1.1rem; border-left: 3px solid var(--accent);
  background: var(--plane); border-radius: 0 6px 6px 0; }
blockquote p { margin: 0.3rem 0; }
hr { border: none; border-top: 1px solid var(--grid); margin: 2.5rem 0; }
ul, ol { padding-left: 1.35rem; }
li { margin: 0.28rem 0; }
"""


def inline_images(html: str, base_dir: Path) -> tuple[str, int, int]:
    """Replace <img src="relative.png"> with a base64 data URI."""
    inlined = missing = 0

    def repl(match: re.Match) -> str:
        nonlocal inlined, missing
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = (base_dir / src).resolve()
        if not path.exists():
            missing += 1
            return match.group(0)
        suffix = path.suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        inlined += 1
        return f'src="data:image/{mime};base64,{encoded}"'

    return re.sub(r'src="([^"]+)"', repl, html), inlined, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--md", type=Path, default=EXP_ROOT / "cam8_holdout_experiment_report.md"
    )
    parser.add_argument(
        "--out", type=Path, default=EXP_ROOT / "cam8_holdout_experiment_report.html"
    )
    parser.add_argument("--title", default="Cam8 Hold-out Experiment")
    args = parser.parse_args()

    if not args.md.exists():
        raise SystemExit(f"missing markdown source: {args.md}")

    body = markdown.markdown(
        args.md.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"],
    )
    body = re.sub(r"(<table>)", r'<div class="table-wrap">\1', body)
    body = re.sub(r"(</table>)", r"\1</div>", body)
    body, inlined, missing = inline_images(body, args.md.parent)

    args.out.write_text(
        "\n".join(
            [
                "<!doctype html>",
                '<html lang="zh-Hant">',
                "<head>",
                '<meta charset="utf-8">',
                '<meta name="viewport" content="width=device-width, initial-scale=1">',
                f"<title>{args.title}</title>",
                f"<style>{STYLE}</style>",
                "</head>",
                "<body>",
                "<main>",
                body,
                "</main>",
                "</body>",
                "</html>",
            ]
        ),
        encoding="utf-8",
    )
    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"{args.out}  ({size_mb:.2f} MB, {inlined} images inlined, {missing} missing)")


if __name__ == "__main__":
    main()
