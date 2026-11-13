"""Build base -> Hindi SFT -> Tameez tables from the raw lm-evaluation-harness JSON files.

    python make_tables.py <results_dir> > ../RESULTS.md

<results_dir>/<run_name>/**/results_*.json, one run per model. Each entry in STAGES is
(base model, Stage 1 Hindi SFT model, Stage 2 Tameez model). The change is split in two:
SFT effect = SFT - base, GRPO effect = Tameez - SFT. Any change of more than 2 points down is flagged.
"""
import glob
import json
import math
import sys
from pathlib import Path

STAGES = [
    ("Tiny Aya Global", "tiny_aya", "tiny_aya_base", "tiny_aya_sft", "tiny_aya_tameez"),
    ("Llama-3.2-3B-Instruct", "llama", "llama_base", "llama_sft", "llama_tameez"),
]
# (label, task key in results, metric key, language, what it measures)
METRICS = [
    ("XNLI", "xnli_en", "acc,none", "English", "language understanding (entailment)"),
    ("XNLI", "xnli_hi", "acc,none", "Hindi", "language understanding (entailment)"),
    ("Belebele", "belebele_eng_Latn", "acc,none", "English", "reading comprehension"),
    ("Belebele", "belebele_hin_Deva", "acc,none", "Hindi", "reading comprehension"),
    ("Global-MMLU Lite", "global_mmlu_en", "acc,none", "English", "knowledge (same questions as Hindi row)"),
    ("Global-MMLU Lite", "global_mmlu_hi", "acc,none", "Hindi", "knowledge (same questions as English row)"),
    ("GSM8K", "gsm8k", "exact_match,flexible-extract", "English", "grade-school maths (5-shot)"),
    ("GSM8K (strict format)", "gsm8k", "exact_match,strict-match", "English", "maths, answer in the exact '####' format"),
    ("IFEval (prompt, strict)", "ifeval", "prompt_level_strict_acc,none", "English", "instruction following"),
    ("IFEval (instruction, strict)", "ifeval", "inst_level_strict_acc,none", "English", "instruction following"),
]
# MILU Hindi, measured earlier in the project on all 14,831 questions (scorecard): base, Hindi SFT, Tameez
MILU = {"tiny_aya": (45.4, 41.3, 43.2), "llama": (38.5, 39.3, 39.5)}


def load(results_dir, run):
    files = sorted(glob.glob(f"{results_dir}/{run}/**/results_*.json", recursive=True))
    if not files:
        return None
    data = json.loads(Path(files[-1]).read_text())
    merged = {**data.get("results", {}), **data.get("groups", {})}
    return {"results": merged, "n": data.get("n-samples", {}), "config": data.get("config", {}),
            "versions": data.get("versions", {}), "file": files[-1]}


def get(run, task, metric):
    if not run or task not in run["results"]:
        return None, None
    r = run["results"][task]
    v = r.get(metric)
    se = r.get(metric.replace(",", "_stderr,"))
    return (100 * v if v is not None else None), (100 * se if isinstance(se, (int, float)) and not math.isnan(se) else None)


def samples(run, task):
    if not run:
        return None
    n = run["n"].get(task)
    if n:
        return n.get("effective")
    parts = [v.get("effective", 0) for k, v in run["n"].items() if k.startswith(task + "_")]
    return sum(parts) if parts else None


def delta(a, b):
    """b - a, formatted, flagged when more than 2 points down."""
    if a is None or b is None:
        return "–"
    d = b - a
    return f"{d:+.1f}" + (" ⚠️" if d < -2 else "")


def main(results_dir):
    out = ["# General-capability benchmarks: base → Hindi SFT → Tameez", "",
           "lm-evaluation-harness 0.4.13 with the vLLM backend, chat template applied, few-shot examples as chat turns, "
           "greedy decoding, at most 500 examples per task or subtask, identical settings for every model. Scores are "
           "percentages; ± is lm-eval's standard error. **SFT effect** = Hindi SFT minus base, **GRPO effect** = Tameez "
           "minus Hindi SFT, **Total** = Tameez minus base, all in points; ⚠️ marks a drop of more than 2 points.", ""]
    for title, key, base_name, sft_name, tam_name in STAGES:
        base, sft, tam = (load(results_dir, n) for n in (base_name, sft_name, tam_name))
        if not (base or sft or tam):
            continue
        out += [f"## {title}", "",
                "| Benchmark | Language | Measures | Base | Hindi SFT | Tameez | SFT effect | GRPO effect | Total | n |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for label, task, metric, lang, what in METRICS:
            (b, bse), (s, sse), (t, tse) = (get(r, task, metric) for r in (base, sft, tam))
            fmt = lambda v, se: "–" if v is None else (f"{v:.1f} ± {se:.1f}" if se is not None else f"{v:.1f}")
            n = samples(tam, task) or samples(base, task) or ""
            out.append(f"| {label} | {lang} | {what} | {fmt(b, bse)} | {fmt(s, sse)} | {fmt(t, tse)} | "
                       f"{delta(b, s)} | {delta(s, t)} | {delta(b, t)} | {n} |")
        mb, ms, mt = MILU[key]
        out.append(f"| MILU (project scorecard) | Hindi | knowledge (14,831 questions) | {mb:.1f} | {ms:.1f} | {mt:.1f} | "
                   f"{delta(mb, ms)} | {delta(ms, mt)} | {delta(mb, mt)} | 14831 |")
        out.append("")
        for run, nm in ((base, base_name), (sft, sft_name), (tam, tam_name)):
            if run:
                out.append(f"- `{nm}`: `{Path(run['file']).relative_to(results_dir)}`")
        out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
