"""Turn the raw lm-eval results into the published artefacts.

    python finalize.py <results_dir> <out_dir>

Writes <out_dir>/RESULTS.md (before/after tables), <out_dir>/benchmarks.json (for the Space) and
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


def collect(results_dir):
    models = []
    for title, base_name, tam_name, fam in [("Tiny Aya Global", "tiny_aya_base", "tiny_aya_tameez", "tiny-aya"),
                                            ("Llama-3.2-3B", "llama_base", "llama_tameez", "llama")]:
        base, tam = mt.load(results_dir, base_name), mt.load(results_dir, tam_name)
        if not (base and tam):
            continue
        rows = []
        for label, task, metric, lang in ROWS:
            b, _ = mt.get(base, task, metric)
            t, _ = mt.get(tam, task, metric)
            if b is not None and t is not None:
                rows.append({"label": label, "lang": lang, "base": round(b, 1), "tameez": round(t, 1), "delta": round(t - b, 1)})
        mb, mtz = mt.MILU["tiny_aya" if fam == "tiny-aya" else "llama"]
        rows.append({"label": "MILU (full)", "lang": "hi", "base": mb, "tameez": mtz, "delta": round(mtz - mb, 1)})
        models.append({"name": title, "family": fam, "rows": rows})
    return models


def chart(models, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        labels = [f"{r['label']}\n{r['lang'].upper()}" for r in m["rows"]]
        y = range(len(labels))
        base = [r["base"] for r in m["rows"]]
        tam = [r["tameez"] for r in m["rows"]]
        color = "#2a78d6" if m["family"] == "tiny-aya" else "#eb6834"
        ax.barh([i + 0.2 for i in y], base, height=0.38, color="#c9c9cf", label="base")
        ax.barh([i - 0.2 for i in y], tam, height=0.38, color=color, label="Tameez")
        for i, r in enumerate(m["rows"]):
            d = r["delta"]
            ax.text(max(r["base"], r["tameez"]) + 1.5, i, f"{d:+.1f}", va="center", fontsize=9,
                    color="#c4382a" if d < -2 else ("#2f8a4b" if d > 2 else "#555"), fontweight="bold" if abs(d) > 2 else None)
        ax.set_yticks(list(y))
        ax.set_yticklabels(labels, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_xlabel("score (%)")
        ax.set_title(m["name"], fontsize=11, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.suptitle("General capability: base model vs Tameez (Δ in points, red = more than 2 points lower)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main(results_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = subprocess.run([sys.executable, str(Path(__file__).parent / "make_tables.py"), results_dir],
                            capture_output=True, text=True, check=True).stdout
    (out / "RESULTS.md").write_text(tables, encoding="utf-8")
    models = collect(results_dir)
    note = ("lm-evaluation-harness, chat template, greedy decoding, up to 500 examples per task; MILU is the project's "
            "full 14,831-question scorecard.")
    (out / "benchmarks.json").write_text(json.dumps({"note": note, "models": models}, ensure_ascii=False, indent=1))
    chart(models, out / "general_capability.png")
    print(tables)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
