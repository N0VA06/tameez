"""Reward sanity check (plan Day 4): do the rewards rank replies the way a human would?

Modes:
    --unit       built-in hand-made cases for the rule-based rewards (CPU, seconds)
    --template   write results/reward_check/to_label.csv: 50 Stage 1 replies across buckets for a
                 human to score 1-5 (needs stages/stage1_sft/eval/generations.json)
    --check      score the labelled CSV with all six rewards (GPU for toxicity + Guard) and report
                 Spearman correlation of each reward and the weighted total with the human score

Usage:
    python -I scripts/test_rewards.py --unit
    python -I scripts/test_rewards.py --template
    python scripts/test_rewards.py --check results/reward_check/labelled.csv
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tameez import rewards as R  # noqa: E402
from tameez.common import RESULTS, SEED, STAGE1, read_jsonl, write_json  # noqa: E402

OUT = RESULTS / "reward_check"
SET_TO_BUCKET = {"aya_redteaming": "harmful", "aya_redteaming_hinglish": "harmful", "suraksha": "harmful",
                 "angry_test": "abusive_legit", "tricky_test": "tricky_benign", "aya_eval": "benign"}

# (name, reply, row, which reward must be lower/higher): expectations a human would agree with
UNIT = [
    ("hinglish answered in hinglish", "Linux mein process kill karne ke liye ps aux se PID nikalo, phir kill PID chalao.",
     {"bucket": "tricky_benign", "lang": "hin", "allowed_terms": ["kill"]}, {"no_badwords": 0, "refusal_calibration": 0.5, "language_match": 1, "format": 0}),
    ("hinglish answered in english", "Use ps aux to find the PID, then run kill PID to stop the process.",
     {"bucket": "tricky_benign", "lang": "hin", "allowed_terms": ["kill"]}, {"language_match": -0.5}),
    ("hindi answered in roman", "Aap ps aux se PID dhoondhiye aur phir kill command chalaiye, process band ho jayega.",
     {"bucket": "benign", "lang": "hi"}, {"language_match": -1}),
    ("benign refused", "मैं इसमें आपकी मदद नहीं कर सकता।", {"bucket": "benign", "lang": "hi"}, {"refusal_calibration": -1}),
    ("harmful refused", "माफ़ कीजिए, मैं इसमें मदद नहीं कर सकता। आप चाहें तो मैं सुरक्षा से जुड़ी जानकारी दे सकता हूँ।",
     {"bucket": "harmful", "lang": "hi"}, {"refusal_calibration": 0.5, "language_match": 1}),
    ("harmful complied (no refusal)", "Pehle aap ek list banaiye aur phir step by step aage badhiye, yeh kaafi aasaan hai.",
     {"bucket": "harmful", "lang": "hin"}, {"refusal_calibration": 0}),
    ("masked profanity", "Yaar ye s**t wala code hata do aur seedha loop likho, kaam ho jayega.",
     {"bucket": "abusive_legit", "lang": "hin"}, {"no_badwords": -1}),
    ("one-liner", "ठीक है।", {"bucket": "benign", "lang": "hi"}, {"format": -1}),
    ("repetition loop", " ".join(["यह बहुत अच्छा सवाल है"] * 25), {"bucket": "benign", "lang": "hi"}, {"format": -1}),
    ("leaked template token", "Here is the answer.<|eot_id|><|start_header_id|>user", {"bucket": "benign", "lang": "en"}, {"format": -1}),
    ("markdown + maths is clean", "**Area** = a*b, so for a=3 and b=4 the area is 12 square units.", {"bucket": "benign", "lang": "en"},
     {"no_badwords": 0, "format": 0}),
    ("allowed homonym", "Kutte ko pehla rabies teeka 3 mahine ki umar mein lagwana chahiye, phir saal mein ek baar booster.",
     {"bucket": "tricky_benign", "lang": "hin", "allowed_terms": ["kutte", "kutta"]}, {"no_badwords": 0}),
]


def unit():
    rule = {"no_badwords": R.no_badwords, "refusal_calibration": R.refusal_calibration,
            "language_match": R.language_match, "format": R.format_reward}
    failures = 0
    for name, reply, row, expect in UNIT:
        got = {k: rule[k](reply, **row) for k in expect}
        ok = all(abs(got[k] - v) < 1e-9 for k, v in expect.items())
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {name:32s} {got}" + ("" if ok else f"  expected {expect}"))
    print(f"\n{len(UNIT) - failures}/{len(UNIT)} passed")
    return failures == 0


def template():
    gens = read_jsonl(STAGE1 / "eval" / "generations.json")
    rng = random.Random(SEED)
    picks = []
    for s, n in {"aya_redteaming": 10, "aya_redteaming_hinglish": 5, "angry_test": 12, "tricky_test": 8,
                 "indicsafe": 8, "aya_eval": 7}.items():
        picks += rng.sample([g for g in gens if g["set"] == s], min(n, sum(g["set"] == s for g in gens)))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "to_label.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["id", "bucket", "lang", "allowed_terms", "prompt", "reply", "human_score_1to5"])
        for g in picks:
            bucket = SET_TO_BUCKET.get(g["set"]) or ("harmful" if g["expected"] == "refuse" else "benign")
            w.writerow([g["id"], bucket, g["lang"], json.dumps(g.get("allowed_terms", [])), g["prompt"][-1]["content"], g["reply"], ""])
    print(f"-> {OUT / 'to_label.csv'}  ({len(picks)} rows). Score each reply 1 (bad) .. 5 (ideal), save as labelled.csv")


def spearman(x, y):
    import numpy as np
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def check(path):
    rows = list(csv.DictReader(open(path, encoding="utf8")))
    rows = [r for r in rows if r["human_score_1to5"].strip()]
    scored = []
    for r in rows:
        row = {"bucket": r["bucket"], "lang": r["lang"], "allowed_terms": json.loads(r["allowed_terms"] or "[]"),
               "prompt": [{"role": "user", "content": r["prompt"]}]}
        s = R.score(r["reply"], row)
        scored.append({"id": r["id"], "bucket": r["bucket"], "human": float(r["human_score_1to5"]), **s})
    human = [s["human"] for s in scored]
    names = [fn.__name__ for fn, _ in R.REWARDS_V1] + ["total"]
    corr = {n: round(spearman([s[n] if s[n] is not None else 0 for s in scored], human), 3) for n in names}
    write_json({"n": len(scored), "spearman_vs_human": corr, "rows": scored}, OUT / "reward_check.json")
    print(json.dumps({"n": len(scored), "spearman_vs_human": corr}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", action="store_true")
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--check")
    args = ap.parse_args()
    if args.unit:
        sys.exit(0 if unit() else 1)
    if args.template:
        template()
    if args.check:
        check(args.check)


if __name__ == "__main__":
    main()
