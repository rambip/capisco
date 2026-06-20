"""THROWAWAY — study where it_core_news_sm and it_core_news_md disagree on POS.

For each sentence prints both models' POS/lemma/morph per token, flags the
disagreements, and shows what the resolver (md-default + sm-overrides) would pick.
Used to design the override rules (e.g. gerunds, which sm catches and md flattens to ADV).
"""
import spacy

CASES = [
    "Per essere perfetta le mancava solo un difetto.",   # perfetta: VERB vs ADJ
    "Affronta la vita sorridendo!",                       # sorridendo: VERB(ger) vs ADV
    "Da te proprio non me l'aspettavo.",                  # the new one
    "Sei una brava persona.",                             # Sei: NUM vs AUX
    "Sono sposata e ho due bambini",                      # sono: AUX vs VERB
    "La donna che ho visto era alta",                     # che: PRON vs SCONJ
    "Sto mangiando una mela",                             # gerund both agree?
    "Correndo, è caduto",                                 # sentence-initial gerund
]

sm = spacy.load("it_core_news_sm")
md = spacy.load("it_core_news_md")


def resolve(a, b):
    """a = sm token, b = md token. Returns (pos, mark, who)."""
    if a.pos_ == b.pos_:
        return b.pos_, _mark(b), "agree"
    # sm reliably tags gerunds that md flattens to ADV — trust sm there.
    if a.morph.get("VerbForm") == ["Ger"]:
        return "VERB", "gerund", "sm"
    return b.pos_, _mark(b), "md"


def _mark(t):
    if t.dep_ in ("expl", "expl:impers", "expl:pass"):
        return "reflexive"
    if t.morph.get("VerbForm") == ["Ger"]:
        return "gerund"
    if t.pos_ == "VERB" and any(c.dep_ == "aux:pass" for c in t.children):
        return "passive"
    return ""


for s in CASES:
    print("•", s)
    da, db = sm(s), md(s)
    for a, b in zip(da, db):
        dis = a.pos_ != b.pos_
        pos, mark, who = resolve(a, b)
        flag = f"  DISAGREE -> pick {who}: {pos}{' (' + mark + ')' if mark else ''}" if dis else ""
        sm_morph = f"sm[{a.lemma_}|{a.morph.get('VerbForm') or ''}]"
        md_morph = f"md[{b.lemma_}|{b.morph.get('VerbForm') or ''}]"
        print(f"    {a.text:<12} sm={a.pos_:<6} md={b.pos_:<6} {sm_morph:<24}{md_morph:<24}{flag}")
    print()
