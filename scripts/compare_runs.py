"""Compare two Tameez runs (e.g. Llama-3.2-3B vs Tiny Aya Global) stage by stage. CPU only.

    python -I scripts/compare_runs.py --a <run A root> --a-name "Llama-3.2-3B" \
                                      --b <run B root> --b-name "Tiny Aya Global" --out COMPARISON.md

A run root is a folder holding stages/stage{0_base,1_sft,2_grpo}/... as written by the pipeline.
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
    ("MILU Hindi accuracy", "milu_hindi_accuracy.accuracy", "up"),
    ("Mean reply length (chars)", "mean_reply_chars", ""),
]


def get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    if isinstance(d, dict) and "rate" in d:
        return d["rate"]
    return d


def load(root, stage, sub="eval/metrics.json"):
    p = Path(root) / "stages" / stage / sub
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, key):
    if v is None:
        return "–"
    if key == "mean_reply_chars":
        return f"{v:.0f}"
    return f"{100 * v:.1f}%"


def winner(a, b, better):
    if a is None or b is None or not better or a == b:
        return ""
    return "B" if (b < a) == (better == "down") else "A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--out", default="COMPARISON.md")
    args = ap.parse_args()
    A, B = args.a_name, args.b_name
    L = [f"# {A} vs {B}: Tameez pipeline, same data, rewards, settings and scorecard", ""]

    # 1. final models side by side
    a2, b2 = load(args.a, "stage2_grpo"), load(args.b, "stage2_grpo")
    a0, b0 = load(args.a, "stage0_base"), load(args.b, "stage0_base")
    a1, b1 = load(args.a, "stage1_sft"), load(args.b, "stage1_sft")
    L += ["## Every stage, both models", "",
          f"| Metric | Better | {A} S0 | {A} S1 | {A} S2 | {B} S0 | {B} S1 | {B} S2 | Better final |",
          "|---|---|---|---|---|---|---|---|---|"]
    for label, key, better in METRICS:
        av = [get(m, key) if m else None for m in (a0, a1, a2)]
        bv = [get(m, key) if m else None for m in (b0, b1, b2)]
        w = winner(av[2], bv[2], better)
        arrow = {"down": "↓", "up": "↑"}.get(better, "")
        L.append(f"| {label} | {arrow} | " + " | ".join(fmt(v, key) for v in av) + " | "
                 + " | ".join(fmt(v, key) for v in bv) + f" | {({'A': A, 'B': B}).get(w, '')} |")

    # 2. what each stage changed (Stage 1 -> Stage 2) per model
    L += ["", "## What GRPO changed (Stage 1 → Stage 2), per model", "",
          f"| Metric | {A}: S1 → S2 | {B}: S1 → S2 |", "|---|---|---|"]
    for label, key, _better in METRICS:
        cells = []
        for s1, s2 in ((a1, a2), (b1, b2)):
            v1, v2 = (get(s1, key) if s1 else None), (get(s2, key) if s2 else None)
            if v1 is None or v2 is None:
                cells.append("–")
            elif key == "mean_reply_chars":
                cells.append(f"{v1:.0f} → {v2:.0f}")
            else:
                cells.append(f"{100 * v1:.1f}% → {100 * v2:.1f}% ({100 * (v2 - v1):+.1f})")
        L.append(f"| {label} | {cells[0]} | {cells[1]} |")

    # 3. plain-SFT safety drift (Stage 0 -> Stage 1)
    L += ["", "## Safety drift from plain Hindi SFT (Stage 0 → Stage 1)", "",
          f"| Metric | {A} | {B} |", "|---|---|---|"]
    for label, key, _ in METRICS[:3] + [METRICS[6], METRICS[7], METRICS[14]]:
        cells = []
        for s0, s1 in ((a0, a1), (b0, b1)):
            v0, v1 = (get(s0, key) if s0 else None), (get(s1, key) if s1 else None)
            cells.append("–" if v0 is None or v1 is None else f"{100 * v0:.1f}% → {100 * v1:.1f}%")
        L.append(f"| {label} | {cells[0]} | {cells[1]} |")

    # 4. checkpoint selection + judge
    for root, name in ((args.a, A), (args.b, B)):
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
    for root, name in ((args.a, A), (args.b, B)):
        j = load(root, "stage2_grpo", "judge/vs_stage1_sft.json")
        if j:
            ci = j.get("ci95", [None, None])
            L.append(f"| {name} | {100 * j['win_rate']:.1f}% | {100 * ci[0]:.1f}–{100 * ci[1]:.1f} | "
                     f"{j['wins']} / {j['ties']} / {j['losses']} | {100 * j['by_set'].get('aya_eval', 0):.1f}% | "
                     f"{100 * j['by_set'].get('angry_test', 0):.1f}% |")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
