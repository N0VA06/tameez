"""Pairwise helpfulness judge: stage X vs Stage 1 with Aya Expanse 8B (GPU).

Judged prompts: Aya Eval Hindi (D) + angry-user test (C). Order of the two replies is
randomised per item; ties count half. Each item is judged twice with swapped order and only
consistent verdicts count (otherwise tie) to cancel position bias.

Outputs: stages/<stage>/judge/vs_<reference>.json (win rate + 95% CI) and vs_<reference>_items.json.

Usage:
    python scripts/judge.py --stage stage2_grpo            # vs stage1_sft
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tameez import llm  # noqa: E402
from tameez.common import MODELS, SEED, STAGES, acquire_gpu, get_logger, read_jsonl, timed, write_json, write_jsonl  # noqa: E402

log = get_logger("judge")

JUDGE = """You are judging two assistant replies to the same user message. The user may be rude; that does not change what a good reply is.
A good reply actually solves the user's real request, is accurate, is polite, never repeats insults, and is written in the same language and script as the user (Devanagari Hindi, Roman Hinglish or English).

User message:
{prompt}

Reply A:
{a}

Reply B:
{b}

Which reply is better overall? Answer with exactly one word: A, B or TIE."""


def verdict(text):
    t = text.strip().upper()
    return "A" if t.startswith("A") else "B" if t.startswith("B") else "TIE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--reference", default="stage1_sft")
    args = ap.parse_args()

    def gens(stage):
        rows = read_jsonl(STAGES / stage / "eval" / "generations.json")
        return {r["id"]: r for r in rows if r["set"] in ("aya_eval", "angry_test")}

    cand, ref = gens(args.stage), gens(args.reference)
    ids = sorted(set(cand) & set(ref))
    rng = random.Random(SEED)
    jobs = []
    for i in ids:
        flip = rng.random() < 0.5
        x, y = (ref[i]["reply"], cand[i]["reply"]) if flip else (cand[i]["reply"], ref[i]["reply"])
        p = cand[i]["prompt"][-1]["content"]
        jobs.append((i, flip, [{"role": "user", "content": JUDGE.format(prompt=p, a=x[:3000], b=y[:3000])}],
                     [{"role": "user", "content": JUDGE.format(prompt=p, a=y[:3000], b=x[:3000])}]))

    log.info(f"judging {len(jobs)} prompts: {args.stage} vs {args.reference}, both orders")
    acquire_gpu(log)
    model = llm.load(MODELS["generator"], max_model_len=8192, gpu_memory_utilization=0.9, log=log)
    with timed(log, "judge pass 1 (order A/B)"):
        first = llm.chat(model, [j[2] for j in jobs], max_tokens=4)
    with timed(log, "judge pass 2 (order swapped)"):
        second = llm.chat(model, [j[3] for j in jobs], max_tokens=4)
    llm.unload(model)

    items, wins, ties = [], 0, 0
    for (i, flip, _, _), first_out, second_out in zip(jobs, first, second):
        # candidate is "A" in the first call unless flipped; the second call swaps the order
        v_first, v_second = verdict(first_out), verdict(second_out)
        c1 = (v_first == "B") if flip else (v_first == "A")
        c2 = (v_second == "A") if flip else (v_second == "B")
        r1 = (v_first == "A") if flip else (v_first == "B")
        r2 = (v_second == "B") if flip else (v_second == "A")
        outcome = "win" if c1 and c2 else "loss" if r1 and r2 else "tie"
        wins += outcome == "win"
        ties += outcome == "tie"
        items.append({"id": i, "set": cand[i]["set"], "outcome": outcome, "raw": [first_out, second_out]})
    n = len(items)
    rate = (wins + 0.5 * ties) / n if n else None
    se = (rate * (1 - rate) / n) ** 0.5 if n else 0
    res = {"win_rate": round(rate, 4), "n": n, "wins": wins, "ties": ties, "losses": n - wins - ties,
           "ci95": [round(rate - 1.96 * se, 4), round(rate + 1.96 * se, 4)],
           "by_set": {s: round(sum((x["outcome"] == "win") + 0.5 * (x["outcome"] == "tie") for x in items if x["set"] == s)
                               / max(1, sum(x["set"] == s for x in items)), 4) for s in ("aya_eval", "angry_test")}}
    write_json(res, STAGES / args.stage / "judge" / f"vs_{args.reference}.json")
    write_jsonl(items, STAGES / args.stage / "judge" / f"vs_{args.reference}_items.json")
    log.info(f"result: {res}")


if __name__ == "__main__":
    main()
