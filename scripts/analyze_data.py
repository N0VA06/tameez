"""Profile every raw dataset in data/raw/ and write docs/data_stats.json.

Prints counts, script mix, lengths and overlaps only. It never prints toxic
text or lexicon entries, so the output is safe to share.

Usage:
    python -I scripts/analyze_data.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "docs" / "data_stats.json"

DEVA = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")
# Very common Roman-Hindi function words; enough to tell Hinglish from English
HINGLISH_WORDS = set(
    "hai hain kya nahi nahin mein main aap tum kaise ka ki ke ko se bhi tha thi "
    "kar karo karna raha rahe hoon ho yeh ye woh wo kuch accha acha bahut mujhe "
    "tera teri mera meri apna kyun kyon".split()
)


def script_of(text):
    """deva | roman_hinglish | english | mixed | other (other Indic scripts / empty)."""
    if not isinstance(text, str):
        return "other"
    d, l = len(DEVA.findall(text)), len(LATIN.findall(text))
    if d + l == 0:
        return "other"
    if d / (d + l) > 0.7:
        return "deva"
    if l / (d + l) > 0.7:
        words = re.findall(r"[a-z]+", text.lower())
        hits = sum(w in HINGLISH_WORDS for w in words)
        return "roman_hinglish" if words and hits / len(words) > 0.08 else "english"
    return "mixed"


def first_turn(conv):
    """IndicAlign stores [[user, assistant], ...]; return (user, assistant) of turn 1."""
    if conv is None or len(conv) == 0:
        return None, None
    t = list(conv[0])
    return (t[0] if t else None), (t[1] if len(t) > 1 else None)


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower()) if isinstance(s, str) else ""


def length_stats(series):
    s = series.dropna().str.len()
    return {"median": int(s.median()), "p90": int(s.quantile(0.9)), "max": int(s.max())} if len(s) else {}


def profile_text(series):
    return {
        "n": int(series.notna().sum()),
        "script": dict(Counter(series.map(script_of))),
        "chars": length_stats(series),
        "exact_duplicates": int(series.dropna().map(norm).duplicated().sum()),
    }


stats = {}

# ---------------- Set A: Hindi SFT ----------------
aya = pd.read_parquet(RAW / "aya_dataset/data/train-00000-of-00001.parquet")
aya_hi = aya[aya.language == "Hindi"]
aya_test = pd.read_parquet(RAW / "aya_dataset/data/test-00000-of-00001.parquet")
stats["aya_dataset_hindi"] = {
    "train_rows": len(aya_hi),
    "test_rows": int((aya_test.language == "Hindi").sum()),
    "prompt": profile_text(aya_hi.inputs),
    "answer": profile_text(aya_hi.targets),
    "annotation_type": dict(Counter(aya_hi.annotation_type)),
    "unique_annotators": int(aya_hi.user_id.nunique()),
    "top_annotator_share": round(aya_hi.user_id.value_counts().iloc[0] / len(aya_hi), 3),
}

ia_dir = RAW / "indic_align_instruct/indicalign-instruct"
for name, path in [("dolly", "dolly/Dolly.parquet"), ("oasst", "oasst/oasst.parquet")]:
    df = pd.read_parquet(ia_dir / path, columns=["num_turns", "eng_Latn", "hin_Deva", "hin_Latn"])
    entry = {"rows": len(df), "num_turns": dict(Counter(df.num_turns.astype(int)))}
    for col in ["hin_Deva", "hin_Latn"]:
        p, r = zip(*df[col].map(first_turn))
        entry[col] = {"prompt": profile_text(pd.Series(p)), "answer": profile_text(pd.Series(r))}
    stats[f"indicalign_instruct_{name}"] = entry

anu = pd.read_parquet(ia_dir / "anudesh/anudesh1.parquet")
p, r = zip(*anu.interactions.map(first_turn))
anu_p, anu_r = pd.Series(p), pd.Series(r)
stats["indicalign_instruct_anudesh"] = {
    "rows": len(anu),
    "num_turns": dict(Counter(anu.num_turns)),
    "prompt": profile_text(anu_p),
    "answer": profile_text(anu_r),
    "prompt_vs_answer_script": {
        f"{a}->{b}": int(n) for (a, b), n in Counter(zip(anu_p.map(script_of), anu_r.map(script_of))).most_common(8)
    },
}

# ---------------- Set B: alignment pool ----------------
tox_dir = RAW / "indic_align_toxic/indicalign-toxic"
tox_prompts = {}
for name, path in [("hhrlhf", "hhrlhf/hh-rlhf.parquet"), ("toxicmatrix", "toxicmatrix/toxic_prompts_sarvam.parquet")]:
    df = pd.read_parquet(tox_dir / path, columns=["num_turns", "eng_Latn", "hin_Deva", "hin_Latn"])
    entry = {"rows": len(df), "num_turns": dict(Counter(df.num_turns.astype(int)))}
    for col in ["eng_Latn", "hin_Deva", "hin_Latn"]:
        p, r = zip(*df[col].map(first_turn))
        entry[col] = {"prompt": profile_text(pd.Series(p)), "answer_present": int(pd.Series(r).notna().sum())}
        tox_prompts[(name, col)] = pd.Series(p)
    stats[f"indicalign_toxic_{name}"] = entry

for lang in ["hi", "hin", "en"]:
    df = pd.read_parquet(RAW / f"textdetox_toxicity/data/{lang}-00000-of-00001.parquet")
    stats[f"textdetox_toxicity_{lang}"] = {
        "rows": len(df),
        "toxic": int(df.toxic.sum()),
        "non_toxic": int((df.toxic == 0).sum()),
        "text": profile_text(df.text),
        "toxic_text_script": dict(Counter(df[df.toxic == 1].text.map(script_of))),
    }

# ---------------- Lexicons ----------------
lex = {}
for lang in ["hi", "hin", "en"]:
    lex[f"textdetox_{lang}"] = pd.read_parquet(RAW / f"textdetox_lexicon/data/{lang}-00000-of-00001.parquet").text
setu_dir = RAW / "setu_nsfw/setu/data/filter_data/nsfw"
for lang in ["hindi", "english"]:
    lines = [x.strip() for x in (setu_dir / f"{lang}.txt").read_text(encoding="utf8").splitlines() if x.strip()]
    lex[f"setu_{lang}"] = pd.Series(lines)

# Benign Hindi/English corpus for false-positive estimation (word-level)
benign_words = Counter()
for text in pd.concat([aya_hi.inputs, aya_hi.targets, anu_p.dropna(), anu_r.dropna()]):
    benign_words.update(re.findall(r"[\wऀ-ॿ]+", text.lower()))

lex_sets = {k: set(v.map(norm)) for k, v in lex.items()}
stats["lexicons"] = {}
for k, v in lex.items():
    entries = lex_sets[k]
    single = [e for e in entries if " " not in e]
    stats["lexicons"][k] = {
        "entries": len(v),
        "unique_normalised": len(entries),
        "multi_word": sum(" " in e for e in entries),
        "short_le3_chars": sum(len(e) <= 3 for e in entries),
        "script": dict(Counter(v.map(script_of))),
        # entries that occur as a word in benign Aya/Anudesh text -> false-positive risk
        "found_in_benign_text": sum(benign_words[e] > 0 for e in single),
        "found_in_benign_text_10plus": sum(benign_words[e] >= 10 for e in single),
    }
all_lex = set().union(*lex_sets.values())
stats["lexicons"]["union_unique"] = len(all_lex)
stats["lexicons"]["overlap_setu_hindi_vs_textdetox_hi"] = len(lex_sets["setu_hindi"] & lex_sets["textdetox_hi"])
stats["lexicons"]["overlap_setu_english_vs_textdetox_en"] = len(lex_sets["setu_english"] & lex_sets["textdetox_en"])

# ---------------- Set C: safety test ----------------
rt = pd.read_json(RAW / "aya_redteaming/aya_hin.jsonl", lines=True)
stats["aya_redteaming_hindi"] = {
    "rows": len(rt),
    "prompt": profile_text(rt.prompt),
    "global_or_local": dict(Counter(rt.global_or_local)),
    "harm_category": dict(Counter(c for cats in rt.harm_category.map(json.loads) for c in cats)),
}

isafe = pd.read_csv(RAW / "indicsafe/data/Full_translated_prompts.csv")
isafe["cat"] = isafe.Category.str.replace(r"\s*Prompts?$", "", regex=True)
isafe_hi = isafe[isafe.Language == "Hindi"]
stats["indicsafe"] = {
    "rows": len(isafe),
    "languages": int(isafe.Language.nunique()),
    "hindi_rows": len(isafe_hi),
    "raw_category_labels": int(isafe.Category.nunique()),
    "normalised_category_labels": int(isafe.cat.nunique()),
    "hindi_by_category": dict(Counter(isafe_hi.cat)),
    "hindi_translated_prompt": profile_text(isafe_hi.Translated_Prompt),
    "english_prompts_shared_across_languages": int(isafe.Prompt.map(norm).nunique()),
}

# ---------------- Set D: skill test ----------------
dhe = pd.read_parquet(RAW / "aya_evaluation_suite/dolly_human_edited/test-00000-of-00001.parquet")
dhe_hi = dhe[dhe.language == "hin"]
stats["aya_eval_dolly_human_edited_hindi"] = {
    "rows": len(dhe_hi),
    "prompt": profile_text(dhe_hi.inputs),
    "script": dict(Counter(dhe_hi.script)),
}

# ---------------- Leakage checks (exact match on normalised English) ----------------
leak = {}
dolly15k = pd.read_json(RAW / "dolly_15k/databricks-dolly-15k.jsonl", lines=True)
eval_src_instr = set(dolly15k.iloc[dhe_hi.source_id.tolist()].instruction.map(norm))
ia_dolly = pd.read_parquet(ia_dir / "dolly/Dolly.parquet", columns=["eng_Latn"])
ia_dolly_eng = ia_dolly.eng_Latn.map(lambda c: norm(first_turn(c)[0]))
# IndicAlign may join instruction + context, so also match on the instruction prefix
ia_dolly_set = set(ia_dolly_eng)
leak["aya_eval_dolly_hindi_prompts_also_in_indicalign_dolly"] = sum(
    1 for s in eval_src_instr if s in ia_dolly_set or any(e.startswith(s) for e in ia_dolly_eng)
)
isafe_en = set(isafe.Prompt.map(norm))
for (name, col), ser in tox_prompts.items():
    if col == "eng_Latn":
        leak[f"indicsafe_english_prompts_in_{name}"] = len(isafe_en & set(ser.map(norm)))
rt_hi = set(rt.prompt.map(norm))
for (name, col), ser in tox_prompts.items():
    if col == "hin_Deva":
        leak[f"aya_redteaming_hindi_prompts_in_{name}"] = len(rt_hi & set(ser.map(norm)))
leak["aya_dataset_hindi_test_in_train"] = len(
    set(aya_test[aya_test.language == "Hindi"].inputs.map(norm)) & set(aya_hi.inputs.map(norm))
)
stats["leakage_exact_match"] = leak

OUT.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
print(json.dumps(stats, indent=2, ensure_ascii=False))
