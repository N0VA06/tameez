"""Compare Tameez runs (e.g. Llama-3.2-3B, Tiny Aya Global, Tiny Aya Fire) stage by stage. CPU only.

    python -I scripts/compare_runs.py --run . "Llama-3.2-3B" --run tiny_aya_run "Tiny Aya Global" \
                                      --run tiny_aya_fire_run "Tiny Aya Fire" --out COMPARISON.md

A run root is a folder holding stages/stage{0_base,1_sft,2_grpo}/... as written by the pipeline. Any number of runs.
Rates are recomputed exactly as count / n from metrics.json (which stores them to 4 decimals) before rounding.
"""

import argparse
import json
from pathlib import Path

STAGES = [("stage0_base", "Stage 0 · base"), ("stage1_sft", "Stage 1 · Hindi SFT"), ("stage2_grpo", "Stage 2 · GRPO")]

# (label, key path in metrics.json, better: "down" | "up" | "")
METRICS = [
    ("Unsafe replies, all harmful prompts", "unsafe_rate.all_harmful", "down"),
    ("• Hindi red-teaming", "unsafe_rate.aya_redteaming", "down"),
    ("• Hinglish red-teaming", "unsafe_rate.aya_redteaming_hinglish", "down"),
    ("• IndicSafe harmful", "unsafe_rate.indicsafe", "down"),
    ("• SurakshaEval", "unsafe_rate.suraksha", "down"),
    ("Bad-word replies", "badword_rate", "down"),
    ("Abusive replies (MuRIL)", "toxicity.abusive_rate", "down"),
    ("Explicit refusal of harmful prompts", "refusal_on_harmful", "up"),
    ("Over-refusal, all benign", "over_refusal.all_benign", "down"),
    ("• IndicSafe harmless + tricky", "over_refusal.indicsafe", "down"),
    ("• own tricky test", "over_refusal.tricky_test", "down"),
    ("• angry users", "over_refusal.angry_test", "down"),
    ("Reply in user's script", "script_fidelity.all", "up"),
    ("• Hinglish prompts", "script_fidelity.hin", "up"),
    ("MILU Hindi accuracy", "milu_hindi_accuracy", "up"),
    ("Mean reply length (chars)", "mean_reply_chars", ""),
]


# MILU Hindi questions per domain (14,831 in all): the overall accuracy is stored to 4 decimals, which cannot tell
# e.g. 6,458 from 6,459 correct, but every per-domain accuracy pins its count exactly
MILU_DOMAIN_N = {"Arts & Humanities": 2862, "Engineering & Tech": 2248, "Science": 2624, "Social Sciences": 1777,
                 "Environmental Sciences": 1993, "Health & Medicine": 598, "Business Studies": 1422, "Law & Governance": 1307}


def get(d, path):
    """The value at path; for a {rate, n} entry the exact fraction count / n (the rates are stored to 4 decimals)."""
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    if isinstance(d, dict) and "rate" in d:
        n = d.get("n")
        return round(d["rate"] * n) / n if n else d["rate"]
    if isinstance(d, dict) and "accuracy" in d:
        dom = d.get("by_domain") or {}
        if set(dom) == set(MILU_DOMAIN_N):
            return sum(round(a * MILU_DOMAIN_N[k]) for k, a in dom.items()) / sum(MILU_DOMAIN_N.values())
        return d["accuracy"]
    return d


def load(root, stage, sub="eval/metrics.json"):
    p = Path(root) / "stages" / stage / sub
    return json.loads(p.read_text()) if p.exists() else None


def pc(v):
    return f"{100 * v:.1f}"


def fmt(v, key):
    if v is None:
        return "–"
    if key == "mean_reply_chars":
        return f"{v:.0f}"
    return f"{pc(v)}%"


def best_of(vals, names, better):
    ok = [(v, n) for v, n in zip(vals, names) if v is not None]
    if not better or len(ok) < 2:
        return ""
    top = min(v for v, _ in ok) if better == "down" else max(v for v, _ in ok)
    winners = [n for v, n in ok if v == top]
    return winners[0] if len(winners) == 1 else "tie"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, action="append", metavar=("ROOT", "NAME"), required=True)
    ap.add_argument("--out", default="COMPARISON.md")
    args = ap.parse_args()
    runs = [(root, name) for root, name in args.run]
    names = [n for _, n in runs]
    S = {name: [load(root, s) for s in ("stage0_base", "stage1_sft", "stage2_grpo")] for root, name in runs}
    L = [f"# {', '.join(names[:-1])} and {names[-1]}: Tameez pipeline, same data, rewards, settings and scorecard", ""]

    # 1. every stage of every model
    head = " | ".join(f"{n} S{i}" for n in names for i in range(3))
    L += ["## Every stage, every model", "", f"| Metric | Better | {head} | Best final |",
          "|" + "---|" * (3 + 3 * len(names))]
    for label, key, better in METRICS:
        cells, finals = [], []
        for n in names:
            vals = [get(m, key) if m else None for m in S[n]]
            cells += [fmt(v, key) for v in vals]
            finals.append(vals[2])
        arrow = {"down": "↓", "up": "↑"}.get(better, "")
        L.append(f"| {label} | {arrow} | " + " | ".join(cells) + f" | {best_of(finals, names, better)} |")

    # 2. what GRPO changed (Stage 1 -> Stage 2) per model
    L += ["", "## What GRPO changed (Stage 1 → Stage 2), per model", "",
          "| Metric | " + " | ".join(f"{n}: S1 → S2" for n in names) + " |", "|" + "---|" * (1 + len(names))]
    for label, key, _better in METRICS:
        cells = []
        for n in names:
            s1, s2 = S[n][1], S[n][2]
            v1, v2 = (get(s1, key) if s1 else None), (get(s2, key) if s2 else None)
            if v1 is None or v2 is None:
                cells.append("–")
            elif key == "mean_reply_chars":
                cells.append(f"{v1:.0f} → {v2:.0f}")
            else:
                cells.append(f"{pc(v1)}% → {pc(v2)}% ({100 * (v2 - v1):+.1f})")
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    # 3. plain-SFT safety drift (Stage 0 -> Stage 1)
    L += ["", "## Safety drift from plain Hindi SFT (Stage 0 → Stage 1)", "",
          "| Metric | " + " | ".join(names) + " |", "|" + "---|" * (1 + len(names))]
    for label, key, _ in METRICS[:3] + [METRICS[6], METRICS[7], METRICS[14]]:
        cells = []
        for n in names:
            s0, s1 = S[n][0], S[n][1]
            v0, v1 = (get(s0, key) if s0 else None), (get(s1, key) if s1 else None)
            cells.append("–" if v0 is None or v1 is None else f"{pc(v0)}% → {pc(v1)}%")
        L.append(f"| {label} | " + " | ".join(cells) + " |")

    # 4. checkpoint selection + judge
    for root, name in runs:
        ck = load(root, "stage2_grpo", "checkpoints/checkpoints.json")
        if not ck:
            continue
        L += ["", f"## {name}: GRPO checkpoint selection (mini test set)", "",
              "| Checkpoint | Unsafe | Bad words | Over-refusal | Script | Mean reply |", "|---|---|---|---|---|---|"]
        ref = ck.get("reference_stage1_mini", {})
        L.append(f"| Stage 1 (reference) | {fmt(ref.get('unsafe'), '')} | {fmt(ref.get('badword'), '')} | "
                 f"{fmt(ref.get('over_refusal'), '')} | {fmt(ref.get('script'), '')} | {ref.get('mean_chars', '–')} |")
        best = (ck.get("best") or {}).get("step")
        for c in ck.get("checkpoints", []):
            tag = " ✓ chosen" if c["step"] == best else ""
            L.append(f"| step {c['step']}{tag} | {fmt(c['unsafe'], '')} | {fmt(c['badword'], '')} | "
                     f"{fmt(c['over_refusal'], '')} | {fmt(c['script'], '')} | {c['mean_chars']} |")
        L.append(f"\nPasses constraints: {ck.get('best_passes_constraints')}")
    L += ["", "## Helpfulness judge (Aya Expanse 8B, Stage 2 vs its own Stage 1, both orders)", "",
          "| Model | Win rate | 95% CI | Wins / ties / losses | Aya-Eval | Angry users |", "|---|---|---|---|---|---|"]
    for root, name in runs:
        j = load(root, "stage2_grpo", "judge/vs_stage1_sft.json")
        if j:
            ci = j.get("ci95", [None, None])
            L.append(f"| {name} | {pc(j['win_rate'])}% | {pc(ci[0])}–{pc(ci[1])} | "
                     f"{j['wins']} / {j['ties']} / {j['losses']} | {pc(j['by_set'].get('aya_eval', 0))}% | "
                     f"{pc(j['by_set'].get('angry_test', 0))}% |")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
