"""Fetch a Tatoeba Italian<->English sentence-pair TSV (pipeline input).

Downloads Tatoeba's weekly per-language exports and the global links table, joins
them with polars, and writes the 4-column TSV that `pipeline.py --input` expects:

    id_ita <TAB> italian <TAB> id_eng <TAB> english

There is no per-pair export from the bulk endpoint, so the links table (all language
pairs, ~430 MB decompressed) is the one heavy piece; everything else is small. The
whole join takes a few seconds. Intermediates go to a temp dir and are deleted unless
--keep is given, so the only file left behind is the output TSV.

Usage:
    uv run capisco-fetch --output it-en-pairs.tsv
"""

from __future__ import annotations

import argparse
import bz2
import io
import sys
import tarfile
import tempfile
from pathlib import Path

import polars as pl
import requests

BASE = "https://downloads.tatoeba.org/exports"
SRC_URL = f"{BASE}/per_language/ita/ita_sentences.tsv.bz2"
TGT_URL = f"{BASE}/per_language/eng/eng_sentences.tsv.bz2"
LINKS_URL = f"{BASE}/links.tar.bz2"

# per_language sentence file: id <TAB> lang <TAB> text
SENT_COLS = ["id", "lang", "text"]
# links.csv: sentence_id <TAB> translation_id
LINK_COLS = ["a", "b"]


def download(url: str, dest: Path) -> Path:
    """Stream `url` to `dest`, printing a one-line progress indicator."""
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                pct = f"{done / total:.0%}" if total else f"{done >> 20} MB"
                print(f"  {dest.name}: {pct}", end="\r", flush=True)
    print(f"  {dest.name}: {dest.stat().st_size >> 20} MB         ")
    return dest


def read_sentences(bz2_path: Path) -> pl.DataFrame:
    """Decompress a per-language .bz2 export in memory and return (id, text)."""
    data = bz2.decompress(bz2_path.read_bytes())
    df = pl.read_csv(
        io.BytesIO(data), separator="\t", has_header=False, quote_char=None,
        new_columns=SENT_COLS, schema_overrides={"id": pl.Utf8})
    return df.select("id", "text")


def read_links(tar_path: Path) -> pl.DataFrame:
    """Extract links.csv from the tar.bz2 (in memory) and return (a, b) id pairs."""
    with tarfile.open(tar_path, "r:bz2") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("links.csv"))
        data = tar.extractfile(member).read()
    return pl.read_csv(
        io.BytesIO(data), separator="\t", has_header=False, quote_char=None,
        new_columns=LINK_COLS, schema_overrides={"a": pl.Utf8, "b": pl.Utf8})


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Tatoeba IT-EN sentence pairs")
    parser.add_argument("--output", type=Path, default=Path("it-en-pairs.tsv"),
                        help="output 4-column TSV path")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="dir for downloaded archives (default: a temp dir, auto-deleted)")
    parser.add_argument("--keep", action="store_true",
                        help="keep downloaded archives instead of deleting them")
    args = parser.parse_args()

    tmp = tempfile.mkdtemp(prefix="tatoeba-") if args.workdir is None else None
    workdir = args.workdir or Path(tmp)
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        print("downloading Tatoeba exports ...", flush=True)
        src_bz2 = download(SRC_URL, workdir / "ita_sentences.tsv.bz2")
        tgt_bz2 = download(TGT_URL, workdir / "eng_sentences.tsv.bz2")
        links_tar = download(LINKS_URL, workdir / "links.tar.bz2")

        print("joining ...", flush=True)
        ita = read_sentences(src_bz2).rename({"id": "a", "text": "ita"})
        eng = read_sentences(tgt_bz2).rename({"id": "b", "text": "eng"})
        links = read_links(links_tar)
        pairs = (links
                 .join(ita, on="a", how="inner")
                 .join(eng, on="b", how="inner")
                 .select("a", "ita", "b", "eng"))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        pairs.write_csv(args.output, separator="\t", include_header=False)
        print(f"wrote {pairs.height:,} IT-EN pairs -> {args.output}")
    finally:
        if tmp is not None and not args.keep:
            for f in workdir.iterdir():
                f.unlink()
            workdir.rmdir()
        elif args.keep:
            print(f"kept archives in {workdir}")


if __name__ == "__main__":
    main()
