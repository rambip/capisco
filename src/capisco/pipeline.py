"""Italian POS viewer — data pipeline (Part 1).

Reads a Tatoeba Italian<->English TSV, keeps sentences that are short and use only
common vocabulary, tags them with spaCy (POS + lemma), aligns each Italian token to
the English translation via an LLM, and writes a single JSON file consumed by the
offline viewer.

Phase 0 findings baked in (see project notes):
  * spaCy has no usable token.rank/token.prob, so the vocabulary filter uses itWaC
    lemma-frequency lists instead (freq/*.csv, Latin-1 encoded).
  * Uses the `md` model: `sm` misreads feminine adjectives as past participles
    (e.g. "perfetta" -> VERB), which `md` gets right (ADJ) with a better parse.
  * Alignment uses a strict [index, word] contract validated against en_tokens.

Usage:
    uv run capisco-pipeline --input it-en-pairs.tsv --config configs/b1.yml
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import requests
import spacy
import yaml
from pydantic import BaseModel, ValidationError


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# itWaC frequency lists (franfranz/Word_Frequency_Lists_ITA). Latin-1 encoded.
FREQ_LISTS = (
    "itwac_nouns_lemmas_notail_2_0_0.csv",
    "itwac_verbs_lemmas_notail_2_1_0.csv",
    "itwac_adj_lemmas_notail_2_1_0.csv",
)
# POS tags whose lemmas are subject to the frequency filter. Function words, ADV,
# PROPN, etc. are exempt (they are common or not covered by the lists).
CONTENT_POS = {"NOUN", "VERB", "ADJ", "AUX"}
RANK_MISS = 10**9  # rank for a content lemma absent from the (de-tailed) lists

# Tatoeba "pair of languages" TSV: id_ita, italian, id_eng, english.
ITA_COL, ENG_COL = 1, 3

ALIGN_PROMPT = """You are given the tokens of an Italian sentence and an indexed list of
English translation tokens. Return a JSON array with one object per Italian token, in order.
Each object has exactly three keys:
  "it":    the Italian token, exactly as given.
  "lemma": its dictionary base form (infinitive for verbs, masculine singular for adjectives,
           singular for nouns). Lowercase, unless it is a proper noun.
  "en":    the English tokens it corresponds to, each as an [index, word] pair where index is
           the position in the English list and word is the exact word at that index.

Rules for "en":
  - Use [] when the Italian token has no real counterpart in the English sentence. This covers
    punctuation and particles, but ALSO content words that the English rephrases or drops: do
    NOT force a match onto an unrelated word. Prefer [] over a wrong link.
  - When several Italian tokens together express one English phrase (e.g. "è alzata" -> "got
    up"), map each of those Italian tokens to the full English phrase.

Example -- "E scomparso senza lasciare traccia." / "It disappeared without a trace.":
"lasciare" has no English counterpart here (the English drops it), so its "en" is [].

Italian tokens: {it}
English tokens (index:word): {en}

Respond ONLY with a JSON array, e.g.:
[{{"it": "Si", "lemma": "si", "en": [[0, "She"]]}},
 {{"it": "e", "lemma": "essere", "en": [[1, "got"], [2, "up"]]}},
 {{"it": "alzata", "lemma": "alzare", "en": [[1, "got"], [2, "up"]]}},
 {{"it": "presto", "lemma": "presto", "en": [[3, "early"]]}},
 {{"it": ".", "lemma": ".", "en": []}}]"""

# Reused across concurrent alignment calls for HTTP connection pooling.
_SESSION = requests.Session()


# --------------------------------------------------------------------------- #
# Data contracts
# --------------------------------------------------------------------------- #

class Token(BaseModel):
    text: str
    pos: str
    lemma: str
    space: bool                 # does a space follow this token (stored on the preceding token)
    align: list[int] = []       # indices into en_tokens
    dep: str = ""               # raw spaCy dependency relation (kept for transparency)
    mark: str = ""              # derived role hint: "" | "passive" | "reflexive"


class Sentence(BaseModel):
    en: str
    en_tokens: list[str]
    en_space: list[bool]        # does a space follow each en_token (for clean rendering)
    tokens: list[Token]


class AlignmentEntry(BaseModel):
    """One element of the LLM response array, e.g.
    {"it": "alzata", "lemma": "alzare", "en": [[1, "got"], [2, "up"]]}.

    `en` is the list of [index, word] English matches ([] = no correspondence); its
    index<->word consistency against en_tokens is checked in `align_sentence`. `lemma`
    is the LLM's base form, used only to override spaCy on VERB tokens (see build_sentence).
    """
    it: str
    lemma: str = ""
    en: list[tuple[int, str]] = []


# --------------------------------------------------------------------------- #
# Frequency filter
# --------------------------------------------------------------------------- #

def build_rank_map(freq_dir: Path) -> dict[str, int]:
    """Build a {lemma: rank} map from the itWaC lists (rank 1 = most frequent).

    Merges nouns/verbs/adjectives, keeping each lemma's highest raw frequency, then
    ranks all distinct lemmas by descending frequency.
    """
    freq: dict[str, int] = {}
    for name in FREQ_LISTS:
        path = freq_dir / name
        with path.open(encoding="latin-1", newline="") as fh:
            for row in csv.DictReader(fh):
                lemma = row["lemma"].strip().lower()
                f = int(row["Freq"])
                if f > freq.get(lemma, 0):
                    freq[lemma] = f
    ordered = sorted(freq, key=lambda lemma: freq[lemma], reverse=True)
    return {lemma: i + 1 for i, lemma in enumerate(ordered)}


# --------------------------------------------------------------------------- #
# Pipeline steps (spec function contracts)
# --------------------------------------------------------------------------- #

def load_tatoeba(
    path: Path,
    rank_map: dict[str, int],
    max_rank: int,
    min_tokens: int,
    max_tokens: int,
    nlp,
    min_rank: int = 0,
) -> Iterator[tuple[str, str]]:
    """Yield filtered (italian, english) pairs lazily.

    Drops a sentence if its non-space token count is outside [min_tokens, max_tokens].
    A sentence's difficulty is the frequency rank of its *rarest* content lemma
    (NOUN/VERB/ADJ/AUX); a sentence is kept only when that rarest rank falls in
    `(min_rank, max_rank]`. The `min_rank` floor lets CEFR-style bands be disjoint
    (e.g. B1 takes 2500-5000, A2 takes 1000-2500). Lemmas absent from the lists count
    as rank RANK_MISS, so any unknown content word pushes the sentence above max_rank
    and it is dropped. De-duplicates by Italian surface text (Tatoeba stores many
    English translations per Italian sentence).

    NOTE: deviates from the original spec signature by taking `rank_map` — spaCy's
    token.rank turned out to be unusable, so frequency comes from the itWaC lists.
    """
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) <= ENG_COL:
                continue
            it_text, en_text = row[ITA_COL].strip(), row[ENG_COL].strip()
            if not it_text or not en_text or it_text in seen:
                continue
            seen.add(it_text)
            doc = nlp(it_text)
            n_tok = sum(1 for t in doc if not t.is_space)
            if n_tok < min_tokens or n_tok > max_tokens:
                continue
            ranks = [
                rank_map.get(t.lemma_.lower(), RANK_MISS)
                for t in doc
                if t.pos_ in CONTENT_POS and not t.is_punct
            ]
            hardest = max(ranks) if ranks else 0
            if not (min_rank < hardest <= max_rank):
                continue
            yield it_text, en_text


def token_mark(t) -> str:
    """Derive a role hint that the bare POS tag can't express.

    - "reflexive": clitic pronouns (mi/ti/si ...) attached to the verb (expl*),
      which look like ordinary object pronouns under POS=PRON.
    - "gerund": the -ando/-endo form (VerbForm=Ger), shown to learners as "-ing".
    - "passive": a lexical verb whose auxiliary is `aux:pass` (e.g. "sposata" in
      "Sono sposata"). The pass marker sits on the AUX child, not the verb, so this
      needs the parse tree — it can't be read off the token alone.
    """
    if t.dep_ in ("expl", "expl:impers", "expl:pass"):
        return "reflexive"
    if t.morph.get("VerbForm") == ["Ger"]:
        return "gerund"
    if t.pos_ == "VERB" and any(c.dep_ == "aux:pass" for c in t.children):
        return "passive"
    return ""


def tag_sentence(nlp_sm, nlp_md, text: str) -> list[Token]:
    """Tag with both models and reconcile POS.

    Default to `md` (the better parser), but let `sm` win where it is reliably right:
    sm correctly tags -ando/-endo gerunds (VerbForm=Ger, with the right infinitive
    lemma) that md often flattens to ADV — e.g. "sorridendo" -> sorridere. The chosen
    model also supplies lemma/dep/mark. The two models share a tokenizer, so the docs
    align 1:1.
    """
    doc_sm, doc_md = nlp_sm(text), nlp_md(text)
    assert len(doc_sm) == len(doc_md), f"tokenization mismatch on: {text!r}"
    tokens = []
    for a, b in zip(doc_sm, doc_md):
        if a.pos_ != b.pos_ and a.morph.get("VerbForm") == ["Ger"]:
            src, pos = a, "VERB"        # trust sm's gerund over md's ADV
        else:
            src, pos = b, b.pos_        # md wins (on agreement or any other disagreement)
        tokens.append(Token(
            text=b.text, pos=pos, lemma=src.lemma_, space=bool(b.whitespace_),
            dep=src.dep_, mark=token_mark(src)))
    return tokens


EN_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def en_tokenize(text: str) -> list[str]:
    """Split an English sentence into word and punctuation tokens."""
    return [m.group() for m in EN_TOKEN_RE.finditer(text)]


def en_spacing(text: str) -> list[bool]:
    """Per en_token: is it followed by whitespace in the original text?

    Lets the viewer reconstruct the English line tightly (no "it ." or "don ' t"),
    since the regex tokenizer drops the original whitespace.
    """
    matches = list(EN_TOKEN_RE.finditer(text))
    return [
        (i + 1 < len(matches) and text[m.end():matches[i + 1].start()] != "")
        for i, m in enumerate(matches)
    ]


def align_sentence(
    it_tokens: list[str],
    en_tokens: list[str],
    model: str,
) -> list[AlignmentEntry] | None:
    """Call the LLM and validate the alignment. Return None on any failure.

    Strict validation: the array has exactly len(it_tokens) entries; each entry has a
    single key equal to the matching Italian token; every English match is an
    [index, word] pair with en_tokens[index] == word. No retry, no partial result.
    """
    en_indexed = " ".join(f"{i}:{w}" for i, w in enumerate(en_tokens))
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": ALIGN_PROMPT.format(
                it=json.dumps(it_tokens, ensure_ascii=False), en=en_indexed),
        }],
    }
    try:
        resp = _SESSION.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, KeyError, ValueError):
        return None

    if raw.startswith("```"):                       # strip markdown fences if present
        raw = raw.split("```")[1]
        raw = raw[4:] if raw.startswith("json") else raw
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list) or len(parsed) != len(it_tokens):
        return None

    entries: list[AlignmentEntry] = []
    for i, obj in enumerate(parsed):
        try:
            entry = AlignmentEntry(**obj)
        except (ValidationError, TypeError):
            return None
        if entry.it != it_tokens[i]:
            return None
        for idx, word in entry.en:
            if not (0 <= idx < len(en_tokens)) or en_tokens[idx] != word:
                return None
        entries.append(entry)
    return entries


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build_sentence(
    pair: tuple[list[Token], str, list[str], list[bool]],
    model: str,
) -> Sentence | None:
    """Align one pre-tagged candidate. Returns None if alignment fails (dropped)."""
    tokens, en_text, en_tokens, en_space = pair
    alignment = align_sentence([t.text for t in tokens], en_tokens, model)
    if alignment is None:
        return None
    for tok, entry in zip(tokens, alignment):
        tok.align = [idx for idx, _ in entry.en]
        # spaCy mangles irregular verb lemmas (vado->vado, leggo->"leggare"); trust the
        # LLM's lemma for verbs only (it over-lemmatizes nouns, e.g. spesa->"spesare").
        if tok.pos == "VERB" and entry.lemma:
            tok.lemma = entry.lemma
    return Sentence(en=en_text, en_tokens=en_tokens, en_space=en_space, tokens=tokens)


def load_dotenv(path: Path = Path(".env")) -> None:
    """Populate os.environ from a .env file if present (does not overwrite existing)."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Italian POS viewer data pipeline")
    parser.add_argument("--config", type=Path, help="YAML file supplying defaults for the options below (CLI args override it)")
    parser.add_argument("--input", type=Path, help="Tatoeba TSV path")
    parser.add_argument("--n", type=int, default=200, help="exact number of sentences to produce")
    parser.add_argument("--max-rank", type=int, default=5000, help="keep sentences whose rarest content lemma rank is <= this")
    parser.add_argument("--min-rank", type=int, default=0, help="band floor: keep sentences whose rarest content lemma rank is > this")
    parser.add_argument("--min-tokens", type=int, default=4, help="min sentence length (tokens, incl. punctuation)")
    parser.add_argument("--max-tokens", type=int, default=12, help="max sentence length (tokens)")
    parser.add_argument("--model", default="google/gemini-3.1-flash-lite", help="OpenRouter model")
    parser.add_argument("--output", type=Path, default=Path("sentences.json"), help="output JSON path")
    parser.add_argument("--freq-dir", type=Path, default=Path("freq"), help="dir with itWaC CSV lists")
    parser.add_argument("--concurrency", type=int, default=10, help="parallel alignment requests")

    # Two-pass: a --config YAML fills defaults, then a normal parse lets CLI flags win.
    PATH_KEYS = {"input", "output", "freq_dir"}
    pre, _ = parser.parse_known_args()
    if pre.config:
        cfg = yaml.safe_load(pre.config.read_text()) or {}
        known = {a.dest for a in parser._actions}
        parser.set_defaults(**{
            k: (Path(v) if k in PATH_KEYS else v)
            for k, v in cfg.items() if k in known
        })
    args = parser.parse_args()
    if args.input is None:
        parser.error("--input is required (pass it on the CLI or in --config)")

    load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("error: OPENROUTER_API_KEY is not set (env var or .env)")

    print("building frequency rank map ...", flush=True)
    rank_map = build_rank_map(args.freq_dir)
    print(f"  {len(rank_map):,} distinct content lemmas", flush=True)

    nlp_sm = spacy.load("it_core_news_sm")
    nlp_md = spacy.load("it_core_news_md")     # md drives filtering + final tags

    candidates = load_tatoeba(
        args.input, rank_map, args.max_rank, args.min_tokens, args.max_tokens, nlp_md,
        min_rank=args.min_rank)

    results: list[Sentence] = []
    dropped = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        batch: list[tuple[str, str]] = []

        def flush() -> None:
            nonlocal dropped
            prepared = [
                (tag_sentence(nlp_sm, nlp_md, it), en, en_tokenize(en), en_spacing(en))
                for it, en in batch
            ]
            for sent in ex.map(lambda p: build_sentence(p, args.model), prepared):
                if sent is None:
                    dropped += 1
                elif len(results) < args.n:
                    results.append(sent)
            batch.clear()
            print(f"  {len(results)}/{args.n} kept ({dropped} dropped)", end="\r", flush=True)

        for pair in candidates:
            batch.append(pair)
            if len(batch) >= args.concurrency:
                flush()
                if len(results) >= args.n:
                    break
        if len(results) < args.n and batch:
            flush()

    print()
    if len(results) < args.n:
        print(f"warning: corpus exhausted with only {len(results)}/{args.n} sentences", file=sys.stderr)

    out = [s.model_dump() for s in results[:args.n]]
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(out)} sentences -> {args.output}")


if __name__ == "__main__":
    main()
