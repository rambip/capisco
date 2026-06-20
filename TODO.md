# Distribution plan

How this project becomes shareable. The guiding decision: **we don't promise bit-for-bit
reproducibility** — Tatoeba changes weekly and the LLM alignment is non-deterministic — so
we don't ship checksums or pin a corpus snapshot. We ship the *recipe* plus a small demo
artifact that makes the viewer work with zero setup.

## What goes where

Four buckets for every piece of data:

| Piece | Decision | Why |
|---|---|---|
| `pipeline.py`, `fetch_data.py`, `viewer.html` | **commit** | the actual project |
| `freq/*.csv` (itWaC lists, ~44 MB) | **commit** | MIT licensed; small enough; the vocab filter is useless without them and they never change |
| `sample_sentences.json` (100 sentences) | **commit** | makes the viewer work out-of-the-box as a demo; treated as a *sample artifact*, not a reproducible build |
| `it-en-pairs.tsv` (Tatoeba IT-EN, ~54 MB) | **generate** with `fetch_data.py` | regenerable in ~2 min; gitignored; would go stale anyway |
| OpenRouter API key | **delegate** to user (`.env`, see `.env.example`) | secret; per-user cost |
| CEFR decks (`decks/*.json`) | **build locally with `build_decks.py` → publish as release assets** | big-ish, non-reproducible, change when re-run; the viewer fetches them from the latest release at runtime (not committed) |

Nothing needs hosting on HuggingFace: the freq lists are committed (MIT), and built
decks live on a GitHub release. If we later want a big prepared deck, HF is an option —
but the release-asset path already covers it.

## Decks (CEFR levels)

Six difficulty bands, one YAML per level in `configs/` (`a1.yml`..`c2.yml`): each sets
`n`, `min_rank`/`max_rank` (frequency-rank band of the *rarest* content word), token
length bounds, and `output: decks/<lvl>.json`. `pipeline.py --config configs/b1.yml`
fills its args from the YAML (CLI flags still override). `level`/`label` keys are read
by the chooser-side label map, not by the pipeline.

> **TODO: find a better ranking.** Bands are currently cut purely on itWaC
> lemma-frequency rank of the rarest content word. This is crude — it ignores
> multiword expressions, syntax/morphology difficulty, and named entities, and the
> rank cutoffs (1k/2.5k/5k/9k/15k) are guesses, not calibrated to real CEFR lists.
> The band edges are disjoint by construction (`min_rank` floor), so re-tuning is just
> editing the YAMLs and rebuilding.

The viewer's deck chooser lists every `.json` asset on the **latest GitHub release**
(`rambip/capisco`); clicking one sets `?deck=<asset-name>` (resolved against the latest
release on load, so URLs stay short, e.g. `?deck=b1.json#392`). A 🏠 button returns to
the chooser; "load a file…" remains for `file://`/offline use.

## Getting the corpus — there *is* effectively an API

Tatoeba has no per-pair bulk export, but it publishes weekly per-language exports plus a
global links table. `fetch_data.py` downloads the three files and joins them with polars:

- `per_language/ita/ita_sentences.tsv.bz2` (~9 MB)
- `per_language/eng/eng_sentences.tsv.bz2` (~24 MB)
- `links.tar.bz2` (~142 MB — all language pairs; the one heavy piece)

→ **709k IT-EN pairs** in the 4-column TSV `pipeline.py` expects, ~2 min end-to-end,
intermediates auto-deleted. (`tatoebatools` on PyPI wraps these same files; we don't need
the extra dependency.) The unstable official REST API is not worth depending on.

README documents both the script and the manual export-tool fallback.

## Repo structure

Keep it flat — it's a two-file tool (`pipeline.py` + `viewer.html`) plus a fetch helper.

```
capisco/
├── README.md              # setup + data tutorial + usage
├── TODO.md                # this file
├── pyproject.toml / uv.lock
├── .env.example           # OPENROUTER_API_KEY=...
├── fetch_data.py          # Tatoeba -> it-en-pairs.tsv
├── pipeline.py            # TSV (+ configs/*.yml) -> decks/<lvl>.json
├── build_decks.py         # build level(s) -> upload to "decks-latest" release (gh)
├── viewer.html            # offline viewer (open in browser; -> index.html on Pages)
├── sample_sentences.json  # committed demo deck (file-picker fallback)
├── configs/               # a1.yml .. c2.yml — CEFR band definitions
├── freq/                  # itWaC frequency lists (MIT)
├── .github/workflows/
│   └── pages.yml          # deploy viewer.html as the Pages site
└── dev/
    └── disagreements.py   # sm-vs-md POS diagnostic (kept as a dev tool)
```

(`decks/` is a gitignored build dir; built decks are published as release assets.)

## Cleanup checklist

- [x] `fetch_data.py` written + verified (709k pairs, ~2 min, self-cleaning)
- [ ] delete uv stub `main.py`
- [ ] delete throwaways: `temp_align_test.py`, `temp_explore_real.py`,
      `temp_freq_explore.py`, `temp_metric_test.py`, `temp_pipeline_proto.py`,
      `temp_backfill_dep.py`, `explore.py`
- [ ] move `temp_disagreements.py` → `dev/disagreements.py`
- [ ] add `.env.example`
- [ ] confirm `.gitignore` keeps `.env`, `*.tsv`, `sentences*.json` out but **not**
      `sample_sentences.json` or `freq/*.csv` (currently correct)
- [ ] write `README.md` (see below)
- [ ] set up GitHub Pages to serve `viewer.html` + a deck
- [ ] add build-corpus Action (scheduled + `workflow_dispatch`, key from secrets)

## Building & publishing decks

- **`pages.yml` — deploy viewer** (push to `main` touching `viewer.html`, or manual):
  copies `viewer.html` → `_site/index.html` and deploys via `upload-pages-artifact` +
  `deploy-pages` (Actions deploy, so no `/docs` folder constraint). The viewer pulls
  decks from the latest release at runtime, so the artifact is just the one file.
- **`build_decks.py` — build decks locally** (run by hand, not in CI): `uv run
  build_decks.py b1` (or `all`, or several levels) runs `pipeline.py --config
  configs/<lvl>.yml` per level, then `gh release upload decks-latest decks/*.json
  --clobber` (creates the `decks-latest` release if missing). `--no-upload` builds only.
  - Local, not CI: each level costs LLM calls and uses the local `.env` key; build a
    level at a time to control spend. Run `uv run fetch_data.py` first for the TSV.

## Remaining setup (one-time, on GitHub)

- [ ] enable Pages → "GitHub Actions" source
- [ ] `uv run fetch_data.py` then `uv run build_decks.py a1` (start with one level) to
      populate the `decks-latest` release
- [ ] confirm the hosted viewer lists the new deck(s)

## README must cover

1. Install: `uv sync` (pulls spaCy + both IT models + polars).
2. Try the demo now: open `viewer.html`, load `sample_sentences.json`.
3. Build your own deck:
   - `uv run fetch_data.py` → `it-en-pairs.tsv`
   - copy `.env.example` → `.env`, add OpenRouter key
   - `uv run pipeline.py --input it-en-pairs.tsv --n 200 --output sentences.json`
   - load `sentences.json` in the viewer
4. Manual corpus fallback (Tatoeba export tool) if the download endpoint moves.
5. Phase 0 rationale (link/short note): itWaC freq filter, `md` model, dual-model gerund
   override — already summarized in `pipeline.py`'s docstring.
