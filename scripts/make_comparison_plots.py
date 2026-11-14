"""Cross-model comparison plots for the write-ups (CPU only).

    python -I scripts/make_comparison_plots.py --run results/tiny_aya_global "Tiny Aya Global" \
        --run results/tiny_aya_fire "Tiny Aya Fire" --run results/llama_3_2_3b "Llama-3.2-3B" --out assets

Writes <out>/compare_final_models.png (every final model on the headline metrics) and
<out>/compare_unsafe_by_stage.png (unsafe replies at each stage, all harmful / Hindi / Hinglish red-teaming).
Each run root holds stages/stage{0_base,1_sft,2_grpo}/ as written by the pipeline.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from compare_runs import get, load  # noqa: E402  (exact count / n rates)

# one fixed hue per model (Fire's magenta checked against the other two for colour-blind separation)
COLORS = {"Tiny Aya Global": "#2a78d6", "Tiny Aya Fire": "#c2409a", "Llama-3.2-3B": "#eb6834"}
INK, MUTED, GRID, BG = "#222222", "#555555", "#e4e4e4", "#fcfcfa"
FINAL = [  # (label, metrics key or "judge", better)
    ("Unsafe replies, all harmful", "unsafe_rate.all_harmful", "↓"),
    ("Unsafe, Hindi red-teaming", "unsafe_rate.aya_redteaming", "↓"),
    ("Unsafe, Hinglish red-teaming", "unsafe_rate.aya_redteaming_hinglish", "↓"),
    ("Abusive replies (MuRIL)", "toxicity.abusive_rate", "↓"),
    ("Over-refusal, all benign prompts", "over_refusal.all_benign", "↓"),
    ("Over-refusal, angry users", "over_refusal.angry_test", "↓"),
    ("Explicit refusal of harmful prompts", "refusal_on_harmful", "↑"),
    ("MILU Hindi accuracy", "milu_hindi_accuracy", "↑"),
    ("Helpfulness win rate vs own SFT model", "judge", "↑"),
]
STAGE_PANELS = [("All 1,498 harmful prompts", "unsafe_rate.all_harmful"),
                ("Hindi red-teaming (913)", "unsafe_rate.aya_redteaming"),
                ("Hinglish red-teaming (139)", "unsafe_rate.aya_redteaming_hinglish")]


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def final_models(runs, out):
    n = len(runs)
    fig, ax = plt.subplots(figsize=(10.5, 0.62 * len(FINAL) + 1.6))
    fig.patch.set_facecolor(BG)
    style(ax)
    gap = 0.22 if n == 3 else 0.26
    for i, (label, key, _better) in enumerate(FINAL):
        for k, (root, name) in enumerate(runs):
            if key == "judge":
                j = load(root, "stage2_grpo", "judge/vs_stage1_sft.json")
                v = j["win_rate"] if j else None
            else:
                v = get(load(root, "stage2_grpo"), key)
            if v is None:
                continue
            y = i + (k - (n - 1) / 2) * gap
            ax.scatter(100 * v, y, s=46, color=COLORS.get(name, MUTED), zorder=3, edgecolors=BG, linewidths=1.2)
            ax.text(100 * v + 1.2, y, f"{100 * v:.1f}", va="center", ha="left", fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(FINAL)))
    ax.set_yticklabels([f"{lab}  ({b} better)" for lab, _, b in FINAL], fontsize=9.5, color=INK)
    ax.set_ylim(len(FINAL) - 0.5, -0.6)
    ax.set_xlim(0, 85)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("%", color=MUTED)
    for _root, name in runs:
        ax.scatter([], [], s=60, color=COLORS.get(name, MUTED), label=name)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=n, frameon=False, fontsize=10)
    fig.suptitle("Final models (Stage 2): same data, rewards and test sets", x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "compare_final_models.png", dpi=150, facecolor=BG)


def unsafe_by_stage(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    fig.patch.set_facecolor(BG)
    x = [0, 1, 2]
    for ax, (title, key) in zip(axes, STAGE_PANELS):
        style(ax)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        finals = []
        for root, name in runs:
            v = [100 * get(load(root, s), key) for s in ("stage0_base", "stage1_sft", "stage2_grpo")]
            c = COLORS.get(name, MUTED)
            ax.plot(x, v, color=c, lw=2.2, marker="o", ms=6.5, mec=BG, mew=1.2, label=name, zorder=3)
            finals.append((v[2], c))
        # label only the end points, at least `gap` apart and never below the axis floor
        gap, floor = 3.2, 1.0
        ys, prev = [], None
        for val, c in sorted(finals, key=lambda t: -t[0]):
            y = val if prev is None else min(val, prev - gap)
            ys.append([y, val, c])
            prev = y
        lift = max(0.0, floor - ys[-1][0])
        for y, val, c in ys:
            ax.text(2.12, y + lift, f"{val:.1f}%", fontsize=8.5, color=c, va="center", ha="left")
        ax.set_xticks(x)
        ax.set_xticklabels(["Stage 0\nbase", "Stage 1\nHindi SFT", "Stage 2\nGRPO"], color=MUTED, fontsize=9.5)
        ax.set_xlim(-0.25, 2.55)
        ax.set_ylim(-1.5, 63)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    axes[0].set_ylabel("Unsafe replies (Llama-Guard-3-8B), %", color=MUTED)
    axes[-1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Plain Hindi SFT made every model less safe; GRPO brought them back", x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "compare_unsafe_by_stage.png", dpi=150, facecolor=BG)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, action="append", metavar=("ROOT", "NAME"), required=True)
    ap.add_argument("--out", default="assets")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [tuple(r) for r in args.run]
    final_models(runs, out)
    unsafe_by_stage(runs, out)
    print(f"wrote {out}/compare_final_models.png, {out}/compare_unsafe_by_stage.png")


if __name__ == "__main__":
    main()
