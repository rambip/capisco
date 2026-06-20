# Italian POS viewer

Offline Italian grammar trainer: Tatoeba sentences → spaCy POS/lemma + LLM word
alignment → a single self-contained HTML viewer (hover for POS colors, lemma, alignment).

## Install
- `uv sync` — pulls spaCy, both IT models (`sm` + `md`), polars

## Try the demo
- open `viewer.html` in a browser, load `sample_sentences.json`

## Build your own deck
- `uv run fetch_data.py` → `it-en-pairs.tsv` (downloads Tatoeba, ~2 min)
- `cp .env.example .env`, add your OpenRouter key
- `uv run pipeline.py --input it-en-pairs.tsv --n 200 --output sentences.json`
- load `sentences.json` in the viewer

## Data sources
- Tatoeba IT-EN pairs — auto-fetched by `fetch_data.py`; manual fallback: Tatoeba export tool
- `freq/` — itWaC frequency lists (MIT), drive the vocabulary filter

## Notes
- Phase 0 rationale (freq filter, `md` model, dual-model gerund override): see `pipeline.py` docstring
- `dev/disagreements.py` — sm-vs-md POS diagnostic
