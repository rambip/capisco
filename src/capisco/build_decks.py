"""Build CEFR decks and publish them as GitHub release assets.

For each requested level it runs `capisco-pipeline --config configs/<lvl>.yml`, then
uploads the resulting `decks/*.json` to a GitHub release via the `gh` CLI (the viewer
reads decks from the latest release's .json assets). Run from the repo root so the
`configs/` and `decks/` paths resolve.

Each level costs OpenRouter LLM calls, so build a level at a time when iterating.

Usage:
    uv run capisco-build b1                 # build just B1
    uv run capisco-build a1 a2 b1           # build several
    uv run capisco-build all                # build every level
    uv run capisco-build b1 --no-upload     # build locally, skip the release
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

LEVELS = ["a1", "a2", "b1", "b2", "c1", "c2"]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_release(tag: str) -> None:
    """Create the release if it doesn't already exist (idempotent)."""
    exists = subprocess.run(
        ["gh", "release", "view", tag],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not exists:
        run([
            "gh", "release", "create", tag,
            "--title", "Latest decks",
            "--notes", "CEFR decks built by build_decks.py. Consumed by the viewer.",
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CEFR decks and upload to a GitHub release")
    parser.add_argument("levels", nargs="+", metavar="LEVEL",
                        help=f"one or more of {LEVELS}, or 'all'")
    parser.add_argument("--input", type=Path, default=Path("it-en-pairs.tsv"),
                        help="Tatoeba TSV (run fetch_data.py first)")
    parser.add_argument("--tag", default="decks-latest", help="release tag to upload assets to")
    parser.add_argument("--no-upload", action="store_true", help="build only; don't touch the release")
    args = parser.parse_args()

    levels = LEVELS if "all" in args.levels else args.levels
    bad = [lv for lv in levels if lv not in LEVELS]
    if bad:
        sys.exit(f"error: unknown level(s) {bad}; choose from {LEVELS} or 'all'")
    if not args.input.exists():
        sys.exit(f"error: {args.input} not found — run `uv run fetch_data.py` first")

    built: list[Path] = []
    for lv in levels:
        cfg = Path("configs") / f"{lv}.yml"
        if not cfg.exists():
            sys.exit(f"error: {cfg} not found")
        print(f"\n=== building {lv} ===", flush=True)
        run(["uv", "run", "capisco-pipeline", "--config", str(cfg), "--input", str(args.input)])
        out = Path("decks") / f"{lv}.json"
        if not out.exists():
            sys.exit(f"error: expected {out} after building {lv}")
        built.append(out)

    if args.no_upload:
        print(f"\nbuilt {len(built)} deck(s); skipping upload: " + ", ".join(str(p) for p in built))
        return

    print(f"\n=== uploading to release '{args.tag}' ===", flush=True)
    ensure_release(args.tag)
    run(["gh", "release", "upload", args.tag, *map(str, built), "--clobber"])
    print(f"uploaded {len(built)} deck(s) to '{args.tag}'")


if __name__ == "__main__":
    main()
