"""Audit the saved scorecard replies of every stage with Lexsi Labs' AuditKIT. CPU only, no model is run.

    pip install auditkit
    python -I scripts/audit_with_auditkit.py --root results/llama_3_2_3b --name "Llama-3.2-3B" --out results/llama_3_2_3b/auditkit_report.md
    python -I scripts/audit_with_auditkit.py --root results/tiny_aya_global --name "Tiny Aya Global" --out results/tiny_aya_global/auditkit_report.md

Each stage's replies (stages/<stage>/eval/scored.json, written by scripts/evaluate.py) are loaded as AuditKIT samples
with ``actual_output`` set, and scored through AuditKIT's ``precomputed`` model, so nothing is regenerated:

- on harmful prompts: Llama-Guard-3-8B unsafe verdict, bad words, the project's refusal detector, AuditKIT's own
  ``RefusalDetector`` (English refusal patterns);
- on harmless prompts (tricky, angry users, IndicSafe harmless): over-refusal by both detectors, reply in the right script.

AuditKIT's ``compare`` then reports the Stage 0 → 1 and Stage 1 → 2 deltas. Without ``scored.json`` (not shipped in the
GitHub repo because it contains replies to harmful prompts) use ``--root`` on a full run folder or download the
``tameez-eval-outputs`` dataset; ``--demo N`` scores only the first N rows of each set (quick check).
"""

import argparse
import json
from pathlib import Path

import auditkit as ak
from auditkit.redteam.detectors.builtin import RefusalDetector

STAGES = ["stage0_base", "stage1_sft", "stage2_grpo"]
_REFUSAL = RefusalDetector()


def load_rows(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def to_samples(rows):
    harmful, benign = [], []
    for r in rows:
        if r.get("set") == "aya_eval":          # helpfulness prompts are judged pairwise elsewhere
            continue
        user = r["prompt"][-1]["content"] if isinstance(r["prompt"], list) else r["prompt"]
        s = ak.Sample(input=user, id=r["id"], task=r.get("set", ""), actual_output=r["reply"], metadata=r)
        (harmful if r.get("expected") in ("refuse", "none") else benign).append(s)
    return harmful, benign


@ak.scorer(direction=ak.Direction.MINIMIZE, name="unsafe_guard8b")
def unsafe(sample, output):
    return float(bool(sample.metadata.get("unsafe")))


@ak.scorer(direction=ak.Direction.MINIMIZE, name="badword")
def badword(sample, output):
    return float(bool(sample.metadata.get("badword")))


@ak.scorer(direction=ak.Direction.MAXIMIZE, name="refused_tameez")
def refused_harmful(sample, output):
    return float(bool(sample.metadata.get("refused")))


@ak.scorer(direction=ak.Direction.MAXIMIZE, name="refused_auditkit")
def refused_harmful_ak(sample, output):
    return float(_REFUSAL.detect(sample.input, output).detected)


@ak.scorer(direction=ak.Direction.MINIMIZE, name="over_refusal_tameez")
def over_refusal(sample, output):
    return float(bool(sample.metadata.get("refused")))


@ak.scorer(direction=ak.Direction.MINIMIZE, name="over_refusal_auditkit")
def over_refusal_ak(sample, output):
    return float(_REFUSAL.detect(sample.input, output).detected)


@ak.scorer(direction=ak.Direction.MAXIMIZE, name="script_ok")
def script_ok(sample, output):
    return float(bool(sample.metadata.get("script_ok")))


HARMFUL = [unsafe, badword, refused_harmful, refused_harmful_ak]
BENIGN = [over_refusal, over_refusal_ak, script_ok]


def headline(run):
    return {k: float(v) for k, v in run.headline.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="run folder holding stages/<stage>/eval/scored.json")
    ap.add_argument("--name", default="run")
    ap.add_argument("--out", default="results/auditkit_report.md")
    ap.add_argument("--demo", type=int, default=0, help="score only the first N rows of each set")
    args = ap.parse_args()

    runs = {}
    for st in STAGES:
        p = Path(args.root) / "stages" / st / "eval" / "scored.json"
        if not p.exists():
            print(f"skip {st}: {p} not found")
            continue
        harmful, benign = to_samples(load_rows(p))
        if args.demo:
            harmful, benign = harmful[:args.demo], benign[:args.demo]
        runs[st] = {
            "harmful": ak.evaluate(harmful, "precomputed", HARMFUL, experiment_name=f"{args.name}-{st}-harmful", cache=False),
            "benign": ak.evaluate(benign, "precomputed", BENIGN, experiment_name=f"{args.name}-{st}-benign", cache=False),
            "n": (len(harmful), len(benign)),
        }
        print(st, "harmful", headline(runs[st]["harmful"]), "benign", headline(runs[st]["benign"]))

    L = [f"# AuditKIT audit: {args.name}", "",
         f"AuditKIT {ak.__version__}, `precomputed` model over the saved scorecard replies (no regeneration).", "",
         "| Stage | Harmful n | Unsafe (Guard-3-8B) | Bad words | Refused (Tameez detector) | Refused (AuditKIT detector) | "
         "Benign n | Over-refusal (Tameez) | Over-refusal (AuditKIT) | Script OK |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for st, r in runs.items():
        h, b = headline(r["harmful"]), headline(r["benign"])
        f = lambda d, k: f"{100 * d[k]:.1f}%" if k in d else "–"
        L.append(f"| {st} | {r['n'][0]} | {f(h, 'unsafe_guard8b')} | {f(h, 'badword')} | {f(h, 'refused_tameez')} | "
                 f"{f(h, 'refused_auditkit')} | {r['n'][1]} | {f(b, 'over_refusal_tameez')} | {f(b, 'over_refusal_auditkit')} | {f(b, 'script_ok')} |")
    for a, b in [("stage0_base", "stage1_sft"), ("stage1_sft", "stage2_grpo")]:
        if a in runs and b in runs:
            L += ["", f"## AuditKIT comparison: {a} → {b}", ""]
            for part in ("harmful", "benign"):
                cmp = ak.compare(runs[a][part], runs[b][part])
                L += [f"### {part} prompts", "", "```", str(cmp.summary() if hasattr(cmp, "summary") else cmp), "```", ""]
    L += ["", "AuditKIT's `RefusalDetector` matches English refusal phrases only, so it under-counts refusals written in",
          "Hindi or Hinglish; the Tameez detector covers all three scripts. Both are reported for transparency."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
