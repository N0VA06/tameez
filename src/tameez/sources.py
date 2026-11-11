"""Load every raw dataset into tidy tables with stable ids, and assign each row a role.

A row's role (Set A, a Set B bucket, a test split, a Hinglish-rewrite source, ...) comes
from a hash of its id, so `make_hinglish.py` and `build_data.py` agree on the split
without sharing state, and a test row can never drift into training between runs.
"""

import hashlib
import json
import re

import pandas as pd
from tqdm import tqdm

from .common import RAW
from .text import is_hindi_deva, script_of


def slot(source, row_id, n=10):
    """Stable bucket 0..n-1 for a row."""
    return int(hashlib.md5(f"{source}:{row_id}".encode()).hexdigest(), 16) % n


def text_key(text):
    """Id for a piece of text (used to cache Guard labels and Hinglish rewrites)."""
    return hashlib.sha1(re.sub(r"\s+", " ", text.strip().lower()).encode()).hexdigest()[:16]


def _norm(s):
    return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else ""


def _turns(conv):
    """IndicAlign conversations: [[user, assistant], ...] -> list of (user, assistant)."""
    out = []
    for turn in conv if conv is not None else []:
        t = list(turn)
        if len(t) >= 2 and t[0] and t[1]:
            out.append((t[0].strip(), t[1].strip()))
    return out


# ---------------------------------------------------------------- Set A sources

def aya_hindi():
    df = pd.read_parquet(RAW / "aya_dataset/data/train-00000-of-00001.parquet")
    df = df[df.language == "Hindi"].reset_index(drop=True)
    df = df.assign(id="aya-" + df.index.astype(str), prompt=df.inputs.str.strip(), answer=df.targets.str.strip())
    df = df[df.prompt.map(script_of) == "deva"]               # 120 Hindi rows have English prompts
    return df.drop_duplicates(subset="prompt")[["id", "prompt", "answer"]]


def leaked_dolly_instructions():
    """English Dolly-15k instructions behind the Aya Eval `dolly_human_edited` Hindi test set."""
    ev = pd.read_parquet(RAW / "aya_evaluation_suite/dolly_human_edited/test-00000-of-00001.parquet")
    ev = ev[ev.language == "hin"]
    dolly = pd.read_json(RAW / "dolly_15k/databricks-dolly-15k.jsonl", lines=True)
    return set(dolly.iloc[ev.source_id.tolist()].instruction.map(_norm))


def dolly_t():
    """IndicAlign Dolly-T Hindi pairs, minus the 200 rows that leak into Aya Eval."""
    df = pd.read_parquet(RAW / "indic_align_instruct/indicalign-instruct/dolly/Dolly.parquet",
                         columns=["doc_id", "eng_Latn", "hin_Deva"])
    leaked = leaked_dolly_instructions()
    rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc="load dolly-t", mininterval=5):
        eng, hin = _turns(r.eng_Latn), _turns(r.hin_Deva)
        if not eng or not hin:
            continue
        eng_p = _norm(eng[0][0])
        if eng_p in leaked or any(eng_p.startswith(x) for x in leaked if len(x) > 30):
            continue
        rows.append({"id": f"dolly-{r.doc_id}", "prompt": hin[0][0], "answer": hin[0][1]})
    return pd.DataFrame(rows)


def oasst_t():
    """IndicAlign OASST-T Hindi conversations: one per opening prompt (tree branches share it)."""
    df = pd.read_parquet(RAW / "indic_align_instruct/indicalign-instruct/oasst/oasst.parquet",
                         columns=["doc_id", "hin_Deva"])
    rows, seen = [], set()
    for r in df.itertuples():
        turns = _turns(r.hin_Deva)
        if not turns:
            continue
        key = _norm(turns[0][0])
        if key in seen:
            continue
        seen.add(key)
        rows.append({"id": f"oasst-{r.doc_id}", "turns": turns})
    return pd.DataFrame(rows)


def anudesh():
    """Human-written Anudesh prompts (first turn), tagged by script; other Indic scripts dropped."""
    df = pd.read_parquet(RAW / "indic_align_instruct/indicalign-instruct/anudesh/anudesh1.parquet")
    rows = []
    for r in tqdm(df.itertuples(), total=len(df), desc="load anudesh", mininterval=5):
        turns = _turns(r.interactions)
        if not turns:
            continue
        p, a = turns[0]
        s = script_of(p)
        if s == "deva" and not is_hindi_deva(p):       # Anudesh also has Marathi/Nepali in Devanagari
            continue
        if s in ("deva", "roman_hinglish", "english") and 10 <= len(p) <= 600:
            rows.append({"id": f"anudesh-{r.id}", "prompt": p, "answer": a, "script": s})
    return pd.DataFrame(rows).drop_duplicates(subset="prompt")


# ---------------------------------------------------------------- Set B sources

def harmful_prompts():
    """IndicAlign-Toxic prompts (first user turn): HH-RLHF-T + Toxic-Matrix, deduplicated."""
    out = []
    for name, path, max_len in [("hhrlhf", "hhrlhf/hh-rlhf.parquet", 400),
                                ("toxicmatrix", "toxicmatrix/toxic_prompts_sarvam.parquet", 400)]:
        df = pd.read_parquet(RAW / "indic_align_toxic/indicalign-toxic" / path,
                             columns=["doc_id", "eng_Latn", "hin_Deva"])
        for r in tqdm(df.itertuples(), total=len(df), desc=f"load {name}", mininterval=5):
            hin, eng = _turns(r.hin_Deva) or [(None, None)], _turns(r.eng_Latn) or [(None, None)]
            out.append({"id": f"{name}-{r.doc_id}", "source": name,
                        "prompt_hi": hin[0][0], "prompt_en": eng[0][0]})
    df = pd.DataFrame(out)
    df = df[df.prompt_hi.notna() & (df.prompt_hi.str.len().between(10, 400))]
    df = df[df.prompt_hi.map(script_of) == "deva"]
    return df.drop_duplicates(subset="prompt_hi").reset_index(drop=True)


def rants(lang):
    """Toxic textdetox texts (lang = hi | hin | en) used to build angry-user prompts."""
    df = pd.read_parquet(RAW / f"textdetox_toxicity/data/{lang}-00000-of-00001.parquet")
    df = df[df.toxic == 1].reset_index(drop=True)
    df = df.assign(id=f"textdetox-{lang}-" + df.index.astype(str), text=df.text.str.strip())
    return df[df.text.str.len().between(15, 300)].drop_duplicates(subset="text")[["id", "text"]]


def tricky_seeds():
    data = json.loads((RAW.parent / "tricky_seeds.json").read_text(encoding="utf8"))
    return data["seeds"]


# ---------------------------------------------------------------- Set C / D sources

def aya_redteaming_hindi():
    df = pd.read_json(RAW / "aya_redteaming/aya_hin.jsonl", lines=True).reset_index(drop=True)
    df = df.assign(id="redteam-" + df.index.astype(str))
    df["harm_category"] = df.harm_category.map(json.loads)
    return df.drop_duplicates(subset="prompt")


def indicsafe_hindi():
    df = pd.read_csv(RAW / "indicsafe/data/Full_translated_prompts.csv")
    df = df[df.Language == "Hindi"].reset_index(drop=True)
    df["category"] = df.Category.str.replace(r"\s*Prompts?$", "", regex=True)
    df["expected"] = df.category.map(lambda c: "help" if c in ("Harmless Control", "Tricky Ambiguous") else "refuse")
    return df.assign(id="indicsafe-" + df.index.astype(str), prompt=df.Translated_Prompt.str.strip())


def suraksha_hindi():
    path = RAW / "suraksha_eval/data/hindi/test.jsonl"
    return pd.read_json(path, lines=True) if path.exists() else pd.DataFrame()


def aya_eval_hindi():
    df = pd.read_parquet(RAW / "aya_evaluation_suite/dolly_human_edited/test-00000-of-00001.parquet")
    df = df[df.language == "hin"].reset_index(drop=True)
    return df.assign(id="ayaeval-" + df.id.astype(str))


def milu_hindi():
    path = RAW / "milu/Hindi/test-00000-of-00001.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


# ---------------------------------------------------------------- roles
# Every row gets exactly one role from a stable hash of its id. Test roles (C_*) are
# never used for training or as generation seeds. None = not used.

def role(source, row_id, script=None):
    if source == "aya":            # 869 rows
        s = slot("aya", row_id)
        return "A" if s < 7 else "B_question" if s < 9 else "C_angry_question"
    if source == "dolly":          # 14.8K pairs
        s = slot("dolly", row_id, 20)
        return "A" if s < 13 else "A_hin_rewrite" if s < 15 else None
    if source == "anudesh":
        s = slot("anudesh", row_id)
        if script == "deva":
            return ("B_benign" if s < 4 else "B_question" if s < 6 else "B_hin_rewrite" if s < 8
                    else "C_angry_question" if s < 9 else None)
        if script == "roman_hinglish":
            return "B_benign" if s < 4 else "B_question" if s < 8 else "C_angry_question"
        if script == "english":
            return "B_benign" if s < 1 else "B_question" if s < 2 else "C_angry_question" if s < 3 else None
    if source in ("rant_hi", "rant_hin", "rant_en"):
        return "B_rant" if slot(source, row_id) < 8 else "C_rant"
    if source == "harmful":
        s = slot("harmful", row_id)
        return "B_harmful_hi" if s < 7 else "B_harmful_hin_rewrite" if s < 9 else "B_harmful_en"
    if source == "redteam":
        return "C_hin_rewrite" if slot("redteam", row_id) < 2 else "C"
    if source == "seed":
        return "C_tricky" if int(row_id.lstrip("t")) % 4 == 0 else "B_tricky"   # t04, t08, ... t40
    raise ValueError(source)
