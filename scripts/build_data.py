"""Build sets A (SFT), B (GRPO prompt pool), C (safety test), D (skill test) + manifest.

Inputs : data/raw/, data/tricky_seeds.json, data/processed/cache/synthetic.json (optional)
Outputs: data/processed/
    A_sft.json            {"id","messages","source","lang"}                         Stage 1 SFT
    B_pool.json           {"prompt","reference","bucket","lang","allowed_terms","source","seed_id"}  GRPO
    C_safety.json         {"id","prompt","set","category","expected","lang","allowed_terms"}  safety test
    C_mini.json           200-prompt stratified subset of C for scoring checkpoints
    D_ayaeval.json        Aya Eval dolly-human-edited Hindi (open-ended helpfulness)
    D_milu.json           MILU Hindi test (MCQ)
    manifest.json         counts per set / bucket / lang / source, filters applied
    cache/guard_prompt_labels.json   Llama-Guard-3-1B prompt labels (tameez.labels; reused)
    cache/dedup_report.json          rows removed as near-duplicates of C/D

Usage:
    python -I scripts/build_data.py            # CPU; uses cached Guard labels if present
    python scripts/build_data.py --guard       # GPU: label uncached prompts with Llama Guard 3 1B first
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tameez import labels as guard  # noqa: E402
from tameez import sources as S  # noqa: E402
from tameez.common import (CACHE, PROCESSED, SEED, SYSTEM_PROMPT, get_logger, read_jsonl,  # noqa: E402
                           timed, user_messages, write_json, write_jsonl)

log = get_logger("build_data")
from tameez.text import lang_tag, script_of  # noqa: E402

QUOTA = {  # bucket -> lang -> rows   (~5K; ~56% hi / 32% hin / 12% en, see docs/DATA_ANALYSIS.md)
    "harmful": {"hi": 1150, "hin": 450, "en": 200},
    "abusive_legit": {"hi": 650, "hin": 450, "en": 150},
    "benign": {"hi": 650, "hin": 400, "en": 200},
}
ANGRY_TEST = {"hi": 100, "hin": 60, "en": 40}
MAX_SFT_CHARS = 3000

WRAP = {  # angry-user templates: real rant + real question
    "hi": ["{rant}\n\nखैर, ये बताओ: {q}", "{q}\n\n{rant}", "{rant}... अब काम की बात: {q}", "जल्दी जवाब दो। {rant}\n{q}"],
    "hin": ["{rant}\n\nkhair, ye batao: {q}", "{q}\n\n{rant}", "{rant}... ab kaam ki baat: {q}", "jaldi bata. {rant}\n{q}"],
    "en": ["{rant}\n\nAnyway, answer this: {q}", "{q}\n\n{rant}", "{rant}... now the actual question: {q}"],
}

rng = random.Random(SEED)


def take(rows, n):
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:n]


def short_rant(text, limit=200):
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    return cut[:max(cut.rfind(" "), limit // 2)].rstrip(",;: ")


def synthetic():
    path = CACHE / "synthetic.json"
    rows = [r for r in read_jsonl(path) if r["valid"]] if path.exists() else []
    log.info(f"synthetic: {len(rows)} valid generations" + ("" if rows else " (none yet: run make_synthetic.py)"))
    return rows


# ---------------------------------------------------------------- Set A

def build_a(syn):
    rows = []
    for r in S.dolly_t().itertuples():
        if S.role("dolly", r.id) == "A":
            rows.append({"id": r.id, "turns": [(r.prompt, r.answer)], "source": "indicalign_dolly", "lang": "hi"})
    for r in S.aya_hindi().itertuples():
        if S.role("aya", r.id) == "A":
            rows.append({"id": r.id, "turns": [(r.prompt, r.answer)], "source": "aya_dataset", "lang": "hi"})
    for r in S.oasst_t().itertuples():
        rows.append({"id": r.id, "turns": r.turns, "source": "indicalign_oasst", "lang": "hi"})
    pairs = {}
    for s in syn:
        if s["task"] == "sft_hin":
            pairs.setdefault(s["id"], {})[s["field"]] = s["accepted"][0]
    for rid, p in pairs.items():
        if "prompt" in p and "answer" in p:
            rows.append({"id": rid + "-hin", "turns": [(p["prompt"], p["answer"])], "source": "aya_hinglish_rewrite", "lang": "hin"})

    out = []
    for r in rows:
        if sum(len(a) + len(b) for a, b in r["turns"]) > MAX_SFT_CHARS:
            continue
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for u, a in r["turns"]:
            msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
        out.append({"id": r["id"], "messages": msgs, "source": r["source"], "lang": r["lang"]})
    return out


# ---------------------------------------------------------------- candidates for B and C

def b_candidates(syn):
    """All prompts that may enter Set B, before Guard filtering and quotas."""
    cand = {"harmful": [], "benign": [], "question": [], "tricky": []}
    harm = S.harmful_prompts()
    for r in harm.itertuples():
        role = S.role("harmful", r.id)
        if role == "B_harmful_hi":
            cand["harmful"].append({"text": r.prompt_hi, "lang": "hi", "source": r.source, "seed_id": r.id})
        elif role == "B_harmful_en" and isinstance(r.prompt_en, str) and 10 <= len(r.prompt_en) <= 400:
            cand["harmful"].append({"text": r.prompt_en, "lang": "en", "source": r.source, "seed_id": r.id})
    # thin out the huge Toxic-Matrix pool so HH-RLHF's short, varied prompts are not drowned out
    tm = [c for c in cand["harmful"] if c["source"] == "toxicmatrix"]
    cand["harmful"] = [c for c in cand["harmful"] if c["source"] != "toxicmatrix"] + take(tm, 3000)

    for r in S.anudesh().itertuples():
        role = S.role("anudesh", r.id, r.script)
        lang = {"deva": "hi", "roman_hinglish": "hin", "english": "en"}[r.script]
        item = {"text": r.prompt, "lang": lang, "source": "anudesh", "seed_id": r.id}
        if role == "B_benign":
            cand["benign"].append(item)
        elif role == "B_question":
            cand["question"].append(item)
    for r in S.aya_hindi().itertuples():
        if S.role("aya", r.id) == "B_question":
            cand["question"].append({"text": r.prompt, "lang": "hi", "source": "aya_dataset", "seed_id": r.id})

    for s in syn:
        text = s["accepted"][0] if s["accepted"] else None
        if s["task"] == "harmful_hin":   # Guard judges the Hindi original: it misses harm in Roman Hinglish
            cand["harmful"].append({"text": text, "lang": "hin", "source": "aya_hinglish_rewrite", "seed_id": s["id"],
                                    "guard_text": s["src"]})
        elif s["task"] == "benign_hin":   # split rewrites between benign prompts and angry-user questions
            key = "benign" if S.slot("benign_hin", s["id"], 2) == 0 else "question"
            cand[key].append({"text": text, "lang": "hin", "source": "aya_hinglish_rewrite", "seed_id": s["id"]})

    paras = {(s["id"], s["field"]): s["accepted"] for s in syn if s["task"] == "seed_para"}
    for seed in S.tricky_seeds():
        if S.role("seed", seed["id"]) != "B_tricky":
            continue
        for lang in ("hi", "hin", "en"):
            for i, text in enumerate([seed[lang]] + paras.get((seed["id"], lang), [])):
                cand["tricky"].append({"text": text, "lang": lang, "source": "tricky_seed" if i == 0 else "tricky_paraphrase",
                                       "seed_id": seed["id"], "allowed_terms": seed["allowed_terms"]})
    return cand


def rant_pool(prefix):
    """Rants by reply language, from textdetox (prefix 'B' = train roles, 'C' = test roles)."""
    pool = {"hi": [], "hin": [], "en": []}
    for src in ("hi", "hin", "en"):
        for r in S.rants(src).itertuples():
            if not S.role(f"rant_{src}", r.id).startswith(prefix):
                continue
            s = script_of(r.text)
            lang = {"deva": "hi", "roman_hinglish": "hin", "english": "en"}.get(s)
            if lang and (src == lang or (src == "hin" and lang == "hi")):
                pool[lang].append({"text": short_rant(r.text), "id": r.id})
    return pool


def wrap(question, rant, lang, i):
    templates = WRAP[lang]
    return templates[i % len(templates)].format(rant=rant["text"], q=question["text"])


# ---------------------------------------------------------------- Guard

# ---------------------------------------------------------------- near-duplicate check

def shingles(text, k=5):
    t = re.sub(r"\s+", " ", text.lower()).strip()
    return {t[i:i + k] for i in range(max(1, len(t) - k + 1))}


def dedup_against(rows, get_text, test_texts, threshold=0.7):
    """Drop rows whose MinHash-estimated Jaccard with any test text is >= threshold."""
    from datasketch import MinHash, MinHashLSH

    def mh(text):
        m = MinHash(num_perm=64, seed=SEED)
        for s in shingles(text):
            m.update(s.encode("utf8"))
        return m

    lsh = MinHashLSH(threshold=threshold, num_perm=64)
    for i, t in enumerate(tqdm(test_texts, desc="index C/D for dedup", mininterval=5)):
        lsh.insert(f"t{i}", mh(t))
    keep, dropped = [], 0
    for r in tqdm(rows, desc="dedup vs C/D", mininterval=5):
        if lsh.query(mh(get_text(r))):
            dropped += 1
        else:
            keep.append(r)
    return keep, dropped


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--guard", action="store_true", help="label uncached prompts with Llama Guard 3 1B (GPU)")
    args = ap.parse_args()
    log.info(f"args: {vars(args)}")
    syn = synthetic()
    manifest = {"filters": {}}
    t_c = timed(log, "build C (safety test) and D (skill test)").__enter__()

    # ---- C and D first: everything else is deduplicated against them
    c_rows = []
    for r in S.aya_redteaming_hindi().itertuples():
        c_rows.append({"id": r.id, "text": r.prompt, "set": "aya_redteaming", "category": "|".join(r.harm_category),
                       "expected": "refuse", "lang": "hi"})
    for s in syn:
        if s["task"] == "redteam_hin":
            c_rows.append({"id": s["id"] + "-hin", "text": s["accepted"][0], "set": "aya_redteaming_hinglish",
                           "category": "", "expected": "refuse", "lang": "hin"})
    for r in S.indicsafe_hindi().itertuples():
        c_rows.append({"id": r.id, "text": r.prompt, "set": "indicsafe", "category": r.category,
                       "expected": r.expected, "lang": "hi"})
    for r in S.suraksha_hindi().itertuples():
        c_rows.append({"id": f"suraksha-{r.id}", "text": r.question_indic, "set": "suraksha", "category": r.type_of_harm,
                       "expected": "none", "lang": "hi"})
    paras = {(s["id"], s["field"]): s["accepted"] for s in syn if s["task"] == "seed_para"}
    for seed in S.tricky_seeds():
        if S.role("seed", seed["id"]) == "C_tricky":
            for lang in ("hi", "hin", "en"):
                for i, text in enumerate([seed[lang]] + paras.get((seed["id"], lang), [])):
                    c_rows.append({"id": f"{seed['id']}-{lang}-{i}", "text": text, "set": "tricky_test", "category": seed["id"],
                                   "expected": "help", "lang": lang, "allowed_terms": seed["allowed_terms"]})
    questions = {"hi": [], "hin": [], "en": []}
    for r in S.anudesh().itertuples():
        if S.role("anudesh", r.id, r.script) == "C_angry_question":
            questions[{"deva": "hi", "roman_hinglish": "hin", "english": "en"}[r.script]].append({"text": r.prompt, "id": r.id})
    for r in S.aya_hindi().itertuples():
        if S.role("aya", r.id) == "C_angry_question":
            questions["hi"].append({"text": r.prompt, "id": r.id})
    c_rants = rant_pool("C")
    for lang, n in ANGRY_TEST.items():
        for i, (q, rt) in enumerate(zip(take(questions[lang], n), take(c_rants[lang], n))):
            c_rows.append({"id": f"angry-{lang}-{i}", "text": wrap(q, rt, lang, i), "set": "angry_test",
                           "category": f"q:{q['id']}|r:{rt['id']}", "expected": "help", "lang": lang})
    for r in c_rows:
        r.setdefault("allowed_terms", [])

    d_aya = [{"id": r.id, "prompt": user_messages(r.inputs), "reference_answer": r.targets, "lang": "hi"}
             for r in S.aya_eval_hindi().itertuples()]
    milu = S.milu_hindi()
    d_milu = milu[["question", "option1", "option2", "option3", "option4", "target", "domain", "subject"]].to_dict("records")
    test_texts = [r["text"] for r in c_rows] + [r["prompt"][0]["content"] for r in d_aya] + milu.question.tolist()

    t_c.__exit__(None, None, None)
    log.info(f"C: {len(c_rows)} rows, D_ayaeval: {len(d_aya)}, D_milu: {len(d_milu)}")

    # ---- A
    with timed(log, "build A (SFT)"):
        a_rows = build_a(syn)
        a_rows, a_dropped = dedup_against(a_rows, lambda r: r["messages"][1]["content"], test_texts)
    log.info(f"A: {len(a_rows)} rows ({a_dropped} near-duplicates of C/D removed)")
    manifest["filters"]["A_near_duplicates_of_C_D_removed"] = a_dropped

    # ---- B
    with timed(log, "collect B candidates"):
        cand = b_candidates(syn)
    log.info(f"B candidates: { {k: len(v) for k, v in cand.items()} }")
    gtext = lambda c: c.get("guard_text", c["text"])  # noqa: E731
    labels = guard.label([gtext(c) for k in ("harmful", "benign", "question") for c in cand[k]], log, args.guard)
    lab = lambda t: guard.is_unsafe(labels, t)  # noqa: E731
    before = {k: len(v) for k, v in cand.items()}
    cand["harmful"] = [c for c in cand["harmful"] if lab(gtext(c)) is not False]     # keep unsafe (or unlabelled)
    cand["benign"] = [c for c in cand["benign"] if lab(c["text"]) is not True]        # drop unsafe "benign"
    cand["question"] = [c for c in cand["question"] if lab(c["text"]) is not True]
    manifest["filters"]["guard_1b"] = {k: {"before": before[k], "after": len(cand[k])} for k in before}
    log.info(f"after Guard filter: {manifest['filters']['guard_1b']}")
    manifest["filters"]["guard_labels_available"] = sum(lab(gtext(c)) is not None for k in cand for c in cand[k])

    b_rows = []

    def add(item, bucket, text=None, allowed=None, seed_id=None, source=None):
        b_rows.append({"prompt": user_messages(text or item["text"]),
                       "reference": "refuse" if bucket == "harmful" else "help",
                       "bucket": bucket, "lang": item["lang"], "allowed_terms": allowed or item.get("allowed_terms", []),
                       "source": source or item["source"], "seed_id": seed_id or item["seed_id"]})

    for bucket in ("harmful", "benign"):
        for lang, n in QUOTA[bucket].items():
            for item in take([c for c in cand[bucket] if c["lang"] == lang], n):
                add(item, bucket)
    b_rants = rant_pool("B")
    for lang, n in QUOTA["abusive_legit"].items():
        qs = take([c for c in cand["question"] if c["lang"] == lang], n)
        rs = take(b_rants[lang], len(qs))
        for i, (q, rt) in enumerate(zip(qs, rs * (len(qs) // max(1, len(rs)) + 1))):
            add(q, "abusive_legit", text=wrap(q, rt, lang, i), seed_id=f"q:{q['seed_id']}|r:{rt['id']}",
                source=f"{q['source']}+textdetox")
    for item in cand["tricky"]:
        add(item, "tricky_benign")
    b_rows, b_dropped = dedup_against(b_rows, lambda r: r["prompt"][0]["content"], test_texts)
    log.info(f"B: {len(b_rows)} rows ({b_dropped} near-duplicates of C/D removed)")
    manifest["filters"]["B_near_duplicates_of_C_D_removed"] = b_dropped
    rng.shuffle(b_rows)

    # ---- C mini-set (scores every GRPO checkpoint)
    mini = []
    mini_rng = random.Random(SEED + 1)    # own RNG: rebuilding the RL pool never changes the mini-set
    for (s, e), n in {("angry_test", "help"): 50, ("tricky_test", "help"): 30, ("indicsafe", "help"): 40,
                      ("indicsafe", "refuse"): 30, ("aya_redteaming", "refuse"): 30, ("aya_redteaming_hinglish", "refuse"): 20}.items():
        pool = [r for r in c_rows if r["set"] == s and r["expected"] == e]
        mini_rng.shuffle(pool)
        mini += pool[:n]

    to_c = lambda r: {**{k: v for k, v in r.items() if k != "text"}, "prompt": user_messages(r["text"])}  # noqa: E731
    write_jsonl(a_rows, PROCESSED / "A_sft.json")
    write_jsonl(b_rows, PROCESSED / "B_pool.json")
    write_jsonl([to_c(r) for r in c_rows], PROCESSED / "C_safety.json")
    write_jsonl([to_c(r) for r in mini], PROCESSED / "C_mini.json")
    write_jsonl(d_aya, PROCESSED / "D_ayaeval.json")
    write_jsonl(d_milu, PROCESSED / "D_milu.json")

    count = lambda rows, *keys: dict(Counter("/".join(str(r[k]) for k in keys) for r in rows))  # noqa: E731
    manifest.update({
        "A_sft": {"rows": len(a_rows), "by_source": count(a_rows, "source"), "by_lang": count(a_rows, "lang")},
        "B_pool": {"rows": len(b_rows), "by_bucket": count(b_rows, "bucket"), "by_lang": count(b_rows, "lang"),
                   "by_bucket_lang": count(b_rows, "bucket", "lang"), "by_source": count(b_rows, "source"),
                   "prompt_lang_check": dict(Counter(lang_tag(r["prompt"][0]["content"]) == r["lang"] for r in b_rows))},
        "C_safety": {"rows": len(c_rows), "by_set": count(c_rows, "set"), "by_set_expected": count(c_rows, "set", "expected"),
                     "by_lang": count(c_rows, "lang")},
        "C_mini": {"rows": len(mini), "by_set": count(mini, "set")},
        "D_ayaeval": {"rows": len(d_aya)}, "D_milu": {"rows": len(d_milu)},
        "synthetic_used": dict(Counter(s["task"] for s in syn)),
    })
    write_json(manifest, PROCESSED / "manifest.json")
    log.info("manifest:\n" + json.dumps(manifest, indent=1, ensure_ascii=False))
    log.info(f"wrote A/B/C/D + manifest to {PROCESSED}")


if __name__ == "__main__":
    main()
