"""Turn the raw lm-eval results into the published artefacts.

    python finalize.py <results_dir> <out_dir>

Writes <out_dir>/RESULTS.md (base -> Hindi SFT -> Tameez tables), <out_dir>/benchmarks.json (for the Space) and
<out_dir>/general_capability.png (chart for the write-ups).
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_tables as mt  # noqa: E402

ROWS = [  # (label, task, metric, lang)
    ("XNLI", "xnli_en", "acc,none", "en"),
    ("XNLI", "xnli_hi", "acc,none", "hi"),
    ("Belebele", "belebele_eng_Latn", "acc,none", "en"),
    ("Belebele", "belebele_hin_Deva", "acc,none", "hi"),
    ("Global-MMLU Lite", "global_mmlu_en", "acc,none", "en"),
    ("Global-MMLU Lite", "global_mmlu_hi", "acc,none", "hi"),
    ("GSM8K", "gsm8k", "exact_match,flexible-extract", "en"),
    ("IFEval (prompt, strict)", "ifeval", "prompt_level_strict_acc,none", "en"),
]

# Written after reading the tables; regenerated into RESULTS.md and benchmarks.json with them.
MODEL_SUMMARIES = {
    "tiny-aya": ("The losses came from the Hindi SFT stage, not GRPO. SFT cost English maths (GSM8K −9.0), English "
                 "reading (−2.8) and MILU (−4.1) while lifting Hindi reading (+7.6) and instruction following (+3.8). "
                 "GRPO moved every benchmark by less than 2 points and won back part of MILU (+1.9)."),
    "llama": ("Hindi SFT brought the big Hindi gains (reading +12.6, knowledge +6.8) and the instruction-following "
              "drop (IFEval −3.4). GRPO moved every benchmark by less than 2 points."),
}
SUMMARY_MD = """## Summary: what caused the losses?

**The Hindi SFT stage, not the GRPO safety stage.** Splitting each model into base → Hindi SFT → Tameez:

- **GRPO did not measurably cost general capability.** On both models, every GRPO effect is within ±2 points
  (largest: Tiny Aya MILU +1.9, Llama XNLI English −1.6), which is inside the noise of a 400–500 question sample.
- **Tiny Aya Global: the losses are SFT.** English maths fell during SFT (GSM8K 61.4 → 52.4, −9.0; SFT lost 85
  questions the base model solved and gained 40) and stayed there after GRPO. English reading (Belebele −2.8 of −3.2)
  and MILU Hindi knowledge (−4.1) also dropped at SFT; GRPO won back half of the MILU loss (+1.9). The gains came from
  SFT too: Hindi reading +7.6, IFEval +3.8, XNLI English +3.0.
- **Llama-3.2-3B: gains and the one loss are SFT.** SFT brought the large Hindi gains (Belebele +12.6, Global-MMLU Lite
  +6.8) and the instruction-following drop (IFEval −3.4 prompt-level, −2.7 instruction-level). GSM8K and English
  reading stayed flat through both stages.

**Identical scores are not identical models.** SFT and Tameez tie exactly on three rows (Tiny Aya GSM8K and IFEval,
Llama IFEval). The weights differ, each run's `config.model_args` names the right repository, scores recompute from the
saved samples, and the replies differ (Tiny Aya: 259 of 500 GSM8K replies, 447 of 500 IFEval replies). The per-question
changes cancel out: on Tiny Aya GSM8K, GRPO fixed 21 questions and broke 21.

**What this suggests.** Stage 1 was trained on 5,000 rows of general Hindi and Hinglish instruction data (translated
Dolly and OASST, Aya Dataset), with no English rows and no maths-focused data, so the English maths loss is most likely
forgetting during SFT. Mixing some English and maths data into
Stage 1 is the obvious next thing to try; GRPO itself can stay as it is.

**Caveats.** At 400–500 examples per task the standard error of each score is about ±2 points, so a stage difference
under ~4 points is not significant; GSM8K −9.0 is well outside that, and MILU uses all 14,831 questions (standard error
about ±0.4). The SFT models were run on 13 Nov 2026 in a fresh Colab L4 session with the same pinned stack (vLLM 0.33.0,
transformers 5.20.2, torch 2.14.0, lm-eval 0.4.13) and the same seeds as the base and Tameez runs.
"""


def collect(results_dir):
    models = []
    for title, key, base_name, sft_name, tam_name in mt.STAGES:
        base, sft, tam = (mt.load(results_dir, n) for n in (base_name, sft_name, tam_name))
        if not (base and tam):
            continue
        rows = []
        for label, task, metric, lang in ROWS:
            b, _ = mt.get(base, task, metric)
            s, _ = mt.get(sft, task, metric)
            t, _ = mt.get(tam, task, metric)
            if b is None or t is None:
                continue
            row = {"label": label, "lang": lang, "base": round(b, 1), "tameez": round(t, 1), "delta": round(t - b, 1)}
            if s is not None:
                row.update(sft=round(s, 1), sft_delta=round(s - b, 1), grpo_delta=round(t - s, 1))
            rows.append(row)
        mb, ms, mtz = mt.MILU[key]
        rows.append({"label": "MILU (full)", "lang": "hi", "base": mb, "tameez": mtz, "delta": round(mtz - mb, 1),
                     "sft": ms, "sft_delta": round(ms - mb, 1), "grpo_delta": round(mtz - ms, 1)})
        fam = "tiny-aya" if key == "tiny_aya" else "llama"
        models.append({"name": title.replace("-Instruct", ""), "family": fam, "rows": rows, "summary": MODEL_SUMMARIES.get(fam)})
    return models


def chart(models, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tone = lambda d: dict(color="#c4382a" if d < -2 else ("#2f8a4b" if d > 2 else "#555"),
                          fontweight="bold" if abs(d) > 2 else "normal")
    fig, axes = plt.subplots(1, len(models), figsize=(7.4 * len(models), 6.4), squeeze=False)
    for ax, m in zip(axes[0], models):
        rows = m["rows"]
        has_sft = all("sft" in r for r in rows)
        full, tint = ("#2a78d6", "#a9c9f0") if m["family"] == "tiny-aya" else ("#eb6834", "#f6be9c")
        y = list(range(len(rows)))
        h = 0.27 if has_sft else 0.38
        bars = [("base", "#c9c9cf", "base")] + ([("sft", tint, "Hindi SFT")] if has_sft else []) + [("tameez", full, "Tameez")]
        offs = [-h, 0, h] if has_sft else [-0.2, 0.2]
        for (k, col, lab), off in zip(bars, offs):
            ax.barh([i + off for i in y], [r[k] for r in rows], height=h * 0.92, color=col, label=lab)
        tr = ax.get_yaxis_transform()
        cols = [("SFT", "sft_delta"), ("GRPO", "grpo_delta")] if has_sft else [("Δ", "delta")]
        for j, (head, k) in enumerate(cols):
            x = 1.03 + 0.15 * j
            ax.text(x, -0.95, head, transform=tr, fontsize=8.5, color="#555", ha="left", va="center", clip_on=False)
            for i, r in enumerate(rows):
                ax.text(x, i, f"{r[k]:+.1f}", transform=tr, fontsize=9, ha="left", va="center", clip_on=False, **tone(r[k]))
        ax.set_yticks(y)
        ax.set_yticklabels([f"{r['label']}\n{r['lang'].upper()}" for r in rows], fontsize=8.5)
        ax.set_ylim(len(rows) - 0.5, -1.3)
        ax.set_xlim(0, 100)
        ax.set_xlabel("score (%)")
        ax.set_title(m["name"], fontsize=11, loc="left", pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    fig.suptitle("General capability: base → Hindi SFT → Tameez (change in points per stage; red = more than 2 points lower)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main(results_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = subprocess.run([sys.executable, str(Path(__file__).parent / "make_tables.py"), results_dir],
                            capture_output=True, text=True, check=True).stdout
    (out / "RESULTS.md").write_text(tables.rstrip() + "\n\n" + SUMMARY_MD, encoding="utf-8")
    models = collect(results_dir)
    note = ("lm-evaluation-harness, chat template, greedy decoding, up to 500 examples per task; MILU is the project's "
            "full 14,831-question scorecard. Each stage is benchmarked: base model, Stage 1 Hindi SFT, Stage 2 Tameez (GRPO).")
    (out / "benchmarks.json").write_text(json.dumps({"note": note, "models": models}, ensure_ascii=False, indent=1))
    chart(models, out / "general_capability.png")
    print(tables)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
