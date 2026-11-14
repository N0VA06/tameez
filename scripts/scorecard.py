"""Collect stages/<stage>/eval/metrics.json (+ judge results) into one table.

Columns: stage0_base, stage1_sft, stage2_grpo, then any stages/ablations/<name>.
Outputs: results/scorecard.md and results/scorecard.csv (rows = metrics, columns = stages).

Usage:
    python -I scripts/scorecard.py
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tameez.common import ABLATIONS, RESULTS, STAGES  # noqa: E402

ORDER = ["stage0_base", "stage1_sft", "stage2_grpo"]
ROWS = [  # (label, path into metrics.json, better)
    ("Unsafe reply rate, all harmful (Guard-3-8B)", "unsafe_rate.all_harmful", "↓"),
    ("  Aya Red-teaming Hindi", "unsafe_rate.aya_redteaming", "↓"),
    ("  Aya Red-teaming Hinglish (rewritten)", "unsafe_rate.aya_redteaming_hinglish", "↓"),
    ("  IndicSafe Hindi harmful", "unsafe_rate.indicsafe", "↓"),
    ("  SurakshaEval Hindi", "unsafe_rate.suraksha", "↓"),
    ("Bad-word rate, safety + angry prompts (lexicon)", "badword_rate", "↓"),
    ("  angry-user prompts only", "badword_rate_angry", "↓"),
    ("Abusive rate (MuRIL)", "toxicity.abusive_rate", "↓"),
    ("Over-refusal, all benign", "over_refusal.all_benign", "↓"),
    ("  IndicSafe harmless + tricky", "over_refusal.indicsafe", "↓"),
    ("  own tricky test", "over_refusal.tricky_test", "↓"),
    ("  angry-user test", "over_refusal.angry_test", "↓"),
    ("Refusal on harmful (detector)", "refusal_on_harmful", "↑"),
    ("Script fidelity, all", "script_fidelity.all", "↑"),
    ("  Hindi (Devanagari)", "script_fidelity.hi", "↑"),
    ("  Hinglish (Roman)", "script_fidelity.hin", "↑"),
    ("  English", "script_fidelity.en", "↑"),
    ("MILU Hindi accuracy (0-shot)", "milu_hindi_accuracy", "↑"),
    ("Helpfulness win rate vs Stage 1 (Aya Expanse judge)", "helpfulness_vs_stage1", "↑"),
    ("Mean reply length (chars)", "mean_reply_chars", ""),
]


MILU_DOMAIN_N = {"Arts & Humanities": 2862, "Engineering & Tech": 2248, "Science": 2624, "Social Sciences": 1777,
                 "Environmental Sciences": 1993, "Health & Medicine": 598, "Business Studies": 1422, "Law & Governance": 1307}


def get(d, path):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, dict):
        val = v.get("rate", v.get("accuracy", v.get("win_rate")))
        if "rate" in v and v.get("n"):          # metrics.json keeps 4 decimals: use the exact count / n
            val = round(v["rate"] * v["n"]) / v["n"]
        dom = v.get("by_domain")
        if "accuracy" in v and isinstance(dom, dict) and set(dom) == set(MILU_DOMAIN_N):
            val = sum(round(a * MILU_DOMAIN_N[k]) for k, a in dom.items()) / sum(MILU_DOMAIN_N.values())
        if val is None:
            return "—"
        ci = v.get("ci95")
        ci_s = f" [{ci[0] + 0.0:.1%}–{ci[1] + 0.0:.1%}]" if ci and ci[0] is not None else ""   # + 0.0: no "-0.0%"
        return f"{val:.1%}{ci_s} (n={v.get('n', '?')})"
    return f"{v:.1f}" if isinstance(v, float) else str(v)


def main():
    stages = {}
    folders = [STAGES / s for s in ORDER] + sorted(ABLATIONS.glob("*"))
    for folder in folders:
        p = folder / "eval" / "metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        judge = folder / "judge" / "vs_stage1_sft.json"
        if judge.exists():
            m["helpfulness_vs_stage1"] = json.loads(judge.read_text())
        name = folder.name if folder.parent == STAGES else f"ablation:{folder.name}"
        stages[name] = m
    cols = [s for s in ORDER if s in stages] + sorted(s for s in stages if s not in ORDER)
    if not cols:
        print("no results yet")
        return

    with open(RESULTS / "scorecard.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "better"] + cols)
        for label, path, better in ROWS:
            w.writerow([label.strip(), better] + [fmt(get(stages[c], path)) for c in cols])

    lines = ["# Tameez scorecard", "",
             "Same prompts, greedy decoding, system prompt \"You are a helpful assistant.\" "
             "Safety is judged by models that are NOT used in the reward (Guard-3-8B, MuRIL). "
             "Brackets = 95% Wilson interval.", "",
             "| Metric | | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for label, path, better in ROWS:
        lines.append(f"| {label} | {better} | " + " | ".join(fmt(get(stages[c], path)) for c in cols) + " |")
    (RESULTS / "scorecard.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
