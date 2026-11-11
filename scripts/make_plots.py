"""Draw the result and data plots (PNG) from whatever stages have finished.

Re-run after every stage: each chart gains a series per finished stage, and a stage keeps its
colour across all charts (stage0 blue, stage1 orange, stage2 aqua). Palette: the dataviz skill's
validated default (stages = categorical slots 1-3, languages = violet / magenta / green; both sets
pass the all-pairs colour-blindness check). Every bar carries a value label because some slots are
below 3:1 contrast on the light surface.

Output: results/plots/*.png  (index in results/plots/README.md)

Usage:
    python -I scripts/make_plots.py
"""

import ast
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tameez.common import LOGS, MODELS, PROCESSED, RESULTS, STAGES, read_jsonl  # noqa: E402

OUT = RESULTS / "plots"
STAGE_ORDER = ["stage0_base", "stage1_sft", "stage2_grpo"]
STAGE_LABEL = {"stage0_base": "Stage 0 · base", "stage1_sft": "Stage 1 · Hindi SFT", "stage2_grpo": "Stage 2 · Tameez GRPO"}
STAGE_COLOR = {"stage0_base": "#2a78d6", "stage1_sft": "#eb6834", "stage2_grpo": "#1baf7a"}
LANG_COLOR = {"hi": "#6250d6", "hin": "#e87ba4", "en": "#008300"}
LANG_LABEL = {"hi": "Hindi (Devanagari)", "hin": "Hinglish (Roman)", "en": "English"}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BEFORE, AFTER = "#c3c2b7", "#2a78d6"     # "before filter / old run" is recessive grey

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["DejaVu Sans"], "font.size": 10, "text.color": INK,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlelocation": "left", "axes.titlecolor": INK,
})


# ---------------------------------------------------------------- data

def load_stages():
    stages = {}
    for s in STAGE_ORDER:
        m = STAGES / s / "eval" / "metrics.json"
        if not m.exists():
            continue
        d = {"metrics": json.loads(m.read_text())}
        for key, path in {"scored": "eval/scored.json", "rollout": "env/rollout_report.json", "milu": "eval/milu.json"}.items():
            p = STAGES / s / path
            if p.exists():
                d[key] = read_jsonl(p) if p.suffix == ".json" and key == "scored" else json.loads(p.read_text())
        stages[s] = d
    return stages


def get(d, path):
    for k in path.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def rate_ci(rows, key):
    n = len(rows)
    if not n:
        return None, (None, None)
    k = sum(bool(r[key]) for r in rows)
    p, z = k / n, 1.96
    den = 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / den, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, (c - h, c + h)


# ---------------------------------------------------------------- drawing helpers

NOTES = {  # "How to read" caption printed under each chart (plain language, for judges and teammates)
    "01_unsafe_rate_by_test_set.png":
        "How to read: each bar is the share of the model's replies to harmful test prompts that Llama-Guard-3-8B judged "
        "unsafe (shorter is better). The thin line is the 95% confidence interval: if two stages' lines overlap a lot, "
        "the difference may be noise. Guard-3-8B is never used in training, so the model cannot have learned to please it.",
    "02_hindi_vs_hinglish_redteaming.png":
        "How to read: the same red-team prompts, asked in Devanagari Hindi (violet) and in Roman-script Hinglish "
        "(magenta). Unsafe and abusive replies: lower is better. Explicit refusals: higher is better here, because every "
        "prompt in this test is harmful. A big violet/magenta gap means the model is weaker in Hinglish.",
    "03_over_refusal.png":
        "How to read: over-refusal = the model refused a prompt it should have answered (a harmless question, a "
        "harmless question with a scary word such as 'kill a process', or a rude user with a real question). Lower is "
        "better. Safety training often raises this; the project's claim is that it does not.",
    "04_scorecard_overview.png":
        "How to read: ↓ lower is better, ↑ higher is better; each panel has its own scale. Unsafe = Llama-Guard-3-8B "
        "verdict on harmful prompts; bad-word = reply contains a word from the profanity lexicon; abusive = MuRIL "
        "classifier; script = reply written in the user's own script; MILU = Hindi multiple-choice knowledge test.",
    "05_env_reward_by_bucket.png":
        "How to read: rows are the six training rewards, columns are the four prompt types in the RL pool. Each cell is "
        "the average reward the model's replies earned (−1 worst, +1 best; blue good, red penalised). 'total' is the "
        "weighted sum GRPO maximises. Cells that are not blue show where training can still improve the model.",
    "06_milu_by_domain.png":
        "How to read: each bar is the share of MILU Hindi multiple-choice questions answered correctly in that subject "
        "area (higher is better). The dashed line is 25%, the score of random guessing among four options. Training for "
        "safety should not lower these bars.",
    "07_rl_prompt_pool.png":
        "How to read: each row is a prompt type in the RL training pool, split by language. harmful = should be politely "
        "refused; abusive_legit = rude user with a real question (answer it, ignore the abuse); tricky_benign = harmless "
        "question containing a scary word (answer it); benign = ordinary question (answer it well).",
    "08_guard_prompt_filter.png":
        "How to read: grey = candidate prompts before filtering, blue = prompts kept after Llama-Guard-3-1B judged each "
        "one. For the harmful pool we keep only prompts Guard calls unsafe; for the benign pools we drop prompts Guard "
        "calls unsafe. Without this step, harmless prompts would have been trained as 'refuse'.",
    "10_sft_training_loss.png":
        "How to read: training loss of Stage 1 (Hindi SFT) at each logged step: how wrong the model's next-token "
        "predictions are on the SFT data. It should fall and then flatten. A curve that keeps falling steeply means more "
        "training would help; a sudden rise means instability.",
    "11_grpo_training_curves.png":
        "How to read: one panel per signal logged during GRPO, x = training step. Reward panels (−1 worst … +1 best) "
        "should rise or stay high. KL = how far the model has drifted from the Stage 1 model (should grow slowly; a "
        "spike means it is changing too fast). Reply length should not collapse: very short replies are a known "
        "reward hack. Thin line = each step, thick line = moving average.",
    "09_hinglish_rewrite_acceptance.png":
        "How to read: Aya Expanse 8B rewrote Devanagari text into Roman-script Hinglish. A rewrite is kept only if it is "
        "Roman Hinglish, not a refusal, of similar length, with no added greeting and no words copied from the prompt's "
        "examples. The first prompt listed English words to keep, and Aya pasted them into its output; the fixed prompt "
        "gives strict rules and two worked examples and disables sampling. That raised acceptance and cut contaminated "
        "rewrites from about 40% to about 1.5%.",
}


def finish(fig, ax_or_axes, name, title, subtitle, grid="x", bottom=0.0):
    """Recessive grid on the value axis only, title + subtitle above the plot, 'How to read' note below."""
    for ax in np.atleast_1d(ax_or_axes).ravel():
        ax.grid(False)
        if grid:
            ax.grid(axis=grid, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    top = 1 - 0.95 / fig.get_figheight()          # ~0.95 inch for title + subtitle
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.text(0.01, 1 - 0.15 / fig.get_figheight(), title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.5 / fig.get_figheight(), subtitle, ha="left", va="top", fontsize=9, color=INK2)
    if name in NOTES:
        import textwrap
        width = int(fig.get_figwidth() * 15)            # ~15 characters per inch at 8.5 pt
        fig.text(0.01, -0.02, textwrap.fill(NOTES[name], width), ha="left", va="top", fontsize=8.5, color=INK2,
                 linespacing=1.4)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return name


def pct(x):
    return f"{x:.1%}" if x >= 0.001 or x == 0 else "<0.1%"


def hbars(ax, cats, series, value_fmt=pct, xmax=None, ci=None):
    """Grouped horizontal bars. series: list of (label, colour, values); ci: same shape of (lo, hi) or None."""
    n = len(series)
    h = min(0.8 / n, 0.32)
    y = np.arange(len(cats))
    for i, (label, colour, vals) in enumerate(series):
        pos = y - (n - 1) * h / 2 + i * h
        v = [np.nan if x is None else x for x in vals]
        ax.barh(pos, v, height=h * 0.9, color=colour, edgecolor=SURFACE, linewidth=1.5, label=label)
        if ci:
            for p, (lo, hi), val in zip(pos, ci[i], v):
                if lo is not None and not np.isnan(val):
                    ax.plot([lo, hi], [p, p], color=INK2, linewidth=0.9, solid_capstyle="butt")
        for p, val in zip(pos, v):
            if not np.isnan(val):
                ax.text(val + (xmax or 1) * 0.012 if not ci else max(val, ci[i][list(pos).index(p)][1] or val) + (xmax or 1) * 0.012,
                        p, value_fmt(val), va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, cats)
    ax.invert_yaxis()
    if xmax:
        ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.tick_params(axis="y", length=0)


def legend(ax, above=True, **kw):
    """One row above the plot area (never on top of bars); above=False keeps matplotlib placement."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if above:
        kw = {"loc": "lower left", "bbox_to_anchor": (0, 1.01), "ncol": len(handles), "borderaxespad": 0, **kw}
    ax.legend(handles, labels, frameon=False, fontsize=9, labelcolor=INK2, handlelength=1.2, **kw)


# ---------------------------------------------------------------- charts

def chart_safety(stages):
    sets = [("all_harmful", "All harmful prompts"), ("aya_redteaming", "Aya Red-teaming · Hindi"),
            ("aya_redteaming_hinglish", "Aya Red-teaming · Hinglish"), ("indicsafe", "IndicSafe · harmful"),
            ("suraksha", "SurakshaEval · Hindi")]
    series, cis = [], []
    for s, d in stages.items():
        vals = [get(d["metrics"], f"unsafe_rate.{k}.rate") for k, _ in sets]
        series.append((STAGE_LABEL[s], STAGE_COLOR[s], vals))
        cis.append([tuple(get(d["metrics"], f"unsafe_rate.{k}.ci95") or (None, None)) for k, _ in sets])
    xmax = max(max(c[1] or 0 for c in ci) for ci in cis) * 1.25
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    hbars(ax, [l for _, l in sets], series, xmax=xmax, ci=cis)
    ax.set_xlabel("replies judged unsafe by Llama-Guard-3-8B (lower is better) · line = 95% interval")
    legend(ax)
    return finish(fig, ax, "01_unsafe_rate_by_test_set.png", "Unsafe reply rate by safety test set",
                  "Held-out prompts, greedy decoding, neutral system prompt. Judge is not used in training.")


def chart_hindi_vs_hinglish(stages):
    metrics = [("unsafe", "Unsafe replies (Guard-3-8B) ↓"), ("abusive", "Abusive replies (MuRIL) ↓"),
               ("refused", "Explicit refusals ↑")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=False)
    names = list(stages)
    for ax, (key, title) in zip(axes, metrics):
        x = np.arange(len(names))
        w = 0.34
        top = 0
        for j, (lang, setname) in enumerate([("hi", "aya_redteaming"), ("hin", "aya_redteaming_hinglish")]):
            vals = []
            for s in names:
                rows = [r for r in stages[s].get("scored", []) if r["set"] == setname]
                vals.append(rate_ci(rows, key)[0] or 0)
            pos = x + (j - 0.5) * w
            ax.bar(pos, vals, width=w * 0.9, color=LANG_COLOR[lang], edgecolor=SURFACE, linewidth=1.5,
                   label=LANG_LABEL[lang])
            for p, v in zip(pos, vals):
                ax.text(p, v, pct(v), ha="center", va="bottom", fontsize=8.5, color=INK)
            top = max(top, max(vals))
        ax.set_xticks(x, [STAGE_LABEL[s].split(" · ")[0] for s in names])
        ax.set_xlim(-0.6, max(2, len(names) - 1) + 0.6)      # room for 3 stages: bar width stays fixed
        ax.set_ylim(0, max(0.05, top * 1.25))
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", length=0)
    legend(axes[0], above=False, loc="upper left")
    return finish(fig, axes, "02_hindi_vs_hinglish_redteaming.png", "Same harmful prompts, Hindi vs Hinglish",
                  "Aya Red-teaming Hindi (n=913) and its Hinglish rewrite (n=139). The gap Tameez has to close.", grid="y")


def chart_over_refusal(stages):
    sets = [("all_benign", "All benign prompts"), ("indicsafe", "IndicSafe harmless + tricky"),
            ("tricky_test", "Own tricky-benign test"), ("angry_test", "Angry-user test")]
    series, cis = [], []
    for s, d in stages.items():
        series.append((STAGE_LABEL[s], STAGE_COLOR[s], [get(d["metrics"], f"over_refusal.{k}.rate") for k, _ in sets]))
        cis.append([tuple(get(d["metrics"], f"over_refusal.{k}.ci95") or (None, None)) for k, _ in sets])
    xmax = max(0.1, max(max(c[1] or 0 for c in ci) for ci in cis) * 1.3)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    hbars(ax, [l for _, l in sets], series, xmax=xmax, ci=cis)
    ax.set_xlabel("benign prompts the model refused (lower is better) · line = 95% interval")
    legend(ax)
    return finish(fig, ax, "03_over_refusal.png", "Over-refusal on benign prompts",
                  "The headline claim needs this to stay flat while the unsafe rate falls.")


def chart_overview(stages):
    panels = [("unsafe_rate.all_harmful", "Unsafe replies ↓"), ("badword_rate", "Bad-word replies ↓"),
              ("toxicity.abusive_rate", "Abusive replies ↓"), ("over_refusal.all_benign", "Over-refusal ↓"),
              ("script_fidelity.all", "Reply in user's script ↑"), ("milu_hindi_accuracy", "MILU Hindi accuracy ↑")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6))
    names = list(stages)
    for ax, (path, title) in zip(axes.ravel(), panels):
        vals = []
        for s in names:
            v = get(stages[s]["metrics"], path)
            vals.append((v.get("rate", v.get("accuracy")) if isinstance(v, dict) else v) or 0)
        x = np.arange(len(names))
        ax.bar(x, vals, width=0.5, color=[STAGE_COLOR[s] for s in names], edgecolor=SURFACE, linewidth=1.5)
        ax.set_xlim(-0.6, max(2, len(names) - 1) + 0.6)      # room for 3 stages: bar width stays fixed
        for xi, v in zip(x, vals):
            ax.text(xi, v, pct(v), ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_xticks(x, [STAGE_LABEL[s].split(" · ")[0] for s in names])
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", length=0)
    handles = [matplotlib.patches.Patch(color=STAGE_COLOR[s], label=STAGE_LABEL[s]) for s in names]
    fig.legend(handles=handles, loc="lower center", ncol=len(names), frameon=False, fontsize=9, labelcolor=INK2,
               bbox_to_anchor=(0.5, 0.0))
    return finish(fig, axes, "04_scorecard_overview.png", "Scorecard overview",
                  "Each panel has its own scale. Full numbers with intervals: results/scorecard.md", grid="y", bottom=0.06)


def chart_env(stages):
    with_roll = [s for s in stages if "rollout" in stages[s]]
    if not with_roll:
        return None
    rewards = ["no_badwords", "low_toxicity", "safe_reply", "refusal_calibration", "language_match", "format"]
    buckets = ["harmful", "abusive_legit", "tricky_benign", "benign"]
    cmap = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])  # red (bad) - grey - blue (good)
    fig, axes = plt.subplots(1, len(with_roll), figsize=(5.2 * len(with_roll) + 1, 3.9), squeeze=False)
    for ax, s in zip(axes[0], with_roll):
        rep = stages[s]["rollout"]["report"]
        mat = np.array([[rep[f"bucket:{b}"][r] for b in buckets] for r in rewards], dtype=float)
        ax.imshow(mat, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i in range(len(rewards)):
            for j in range(len(buckets)):
                ax.text(j, i, f"{mat[i, j]:+.2f}", ha="center", va="center", fontsize=8.5, color=INK)
        totals = [rep[f"bucket:{b}"]["total"] for b in buckets]
        ax.set_xticks(range(len(buckets)), [f"{b}\ntotal {t:.2f}" for b, t in zip(buckets, totals)], fontsize=8.5)
        ax.set_yticks(range(len(rewards)), rewards)
        ax.set_title(STAGE_LABEL[s], fontsize=10)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
    return finish(fig, axes, "05_env_reward_by_bucket.png", "Mean reward per bucket in the RL environment",
                  "200 stratified pool prompts, greedy replies, scored by the six training rewards. "
                  "Blue = rewarded, red = penalised, grey = 0.", grid=None)


def chart_milu(stages):
    with_milu = [s for s in stages if "milu" in stages[s]]
    if not with_milu:
        return None
    domains = sorted(stages[with_milu[0]]["milu"]["by_domain"])
    series = [(STAGE_LABEL[s], STAGE_COLOR[s], [stages[s]["milu"]["by_domain"].get(d) for d in domains]) for s in with_milu]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    hbars(ax, domains, series, xmax=0.7)
    ax.axvline(0.25, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(0.252, -0.6, "random guess 25%", fontsize=8, color=MUTED, va="bottom")
    ax.set_xlabel("0-shot accuracy on MILU Hindi test (n=14,831)")
    legend(ax)
    return finish(fig, ax, "06_milu_by_domain.png", "Hindi knowledge (MILU) by domain",
                  "No-regression check: SFT and GRPO should not lose Hindi knowledge.")


def chart_data_composition():
    env = STAGES / "stage1_sft" / "env" / "env_check.json"
    if not env.exists():
        return None
    stats = json.loads(env.read_text())["stats"]["bucket_x_lang"]
    buckets = ["harmful", "abusive_legit", "tricky_benign", "benign"]
    fig, ax = plt.subplots(figsize=(8.5, 3.3))
    left = np.zeros(len(buckets))
    for lang in ("hi", "hin", "en"):
        vals = np.array([stats.get(f"{b}/{lang}", 0) for b in buckets])
        ax.barh(buckets, vals, left=left, height=0.5, color=LANG_COLOR[lang], edgecolor=SURFACE, linewidth=2,
                label=LANG_LABEL[lang])
        for y, (l0, v) in enumerate(zip(left, vals)):
            if v >= 60:
                ax.text(l0 + v / 2, y, f"{v:,}", ha="center", va="center", fontsize=8.5, color="white")
        left += vals
    for y, tot in enumerate(left):
        ax.text(tot + 15, y, f"{int(tot):,}", va="center", fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, left.max() * 1.12)
    ax.set_xlabel("prompts in the GRPO pool (Set B)")
    ax.tick_params(axis="y", length=0)
    legend(ax)
    return finish(fig, ax, "07_rl_prompt_pool.png", "RL prompt pool by bucket and language",
                  f"{int(left.sum()):,} prompts; harmful are Guard-confirmed. Hinglish comes from Aya rewrites + natural Anudesh prompts.")


def chart_filters():
    mpath = PROCESSED / "manifest.json"
    if not mpath.exists():
        return None
    g = json.loads(mpath.read_text())["filters"]["guard_1b"]
    pools = [("harmful", "Harmful candidates\n(keep unsafe)"), ("benign", "Benign candidates\n(drop unsafe)"),
             ("question", "Angry-user questions\n(drop unsafe)")]
    fig, ax = plt.subplots(figsize=(8.5, 3.3))
    y = np.arange(len(pools))
    before = [g[p]["before"] for p, _ in pools]
    after = [g[p]["after"] for p, _ in pools]
    ax.barh(y - 0.18, before, height=0.32, color=BEFORE, edgecolor=SURFACE, linewidth=1.5, label="before Guard filter")
    ax.barh(y + 0.18, after, height=0.32, color=AFTER, edgecolor=SURFACE, linewidth=1.5, label="after Guard filter")
    for yi, b, a in zip(y, before, after):
        ax.text(b + 120, yi - 0.18, f"{b:,}", va="center", fontsize=8.5, color=INK2)
        ax.text(a + 120, yi + 0.18, f"{a:,} ({a / b:.0%})", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [l for _, l in pools])
    ax.invert_yaxis()
    ax.set_xlim(0, max(before) * 1.2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("prompts")
    legend(ax)
    return finish(fig, ax, "08_guard_prompt_filter.png", "Llama-Guard-3-1B prompt filter",
                  "Only about half of the 'harmful' IndicAlign-Toxic prompts are actually unsafe.")


def chart_synthetic():
    log = LOGS / "make_synthetic.log"
    if not log.exists():
        return None
    runs = [ast.literal_eval(m) for m in re.findall(r"valid generations per task: (\{.*\})", log.read_text(encoding="utf8"))]
    if not runs:
        return None
    tasks = [("sft_hin", "SFT pairs → Hinglish"), ("benign_hin", "Benign prompts → Hinglish"),
             ("harmful_hin", "Harmful prompts → Hinglish"), ("redteam_hin", "Red-team test → Hinglish"),
             ("seed_para", "Tricky-seed paraphrases")]
    frac = lambda s: (lambda a, b: a / b)(*map(int, s.split("/")))  # noqa: E731
    first, last = runs[0], runs[-1]
    series = [("first prompt (listed words to keep; Aya copied them)", BEFORE,
               [frac(first[t]) if t in first else None for t, _ in tasks]),
              ("fixed prompt (strict rules + 2 examples, no sampling)", AFTER,
               [frac(last[t]) if t in last else None for t, _ in tasks])]
    if len(runs) == 1:
        series = series[1:]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    hbars(ax, [l for _, l in tasks], series, xmax=1.12)
    ax.set_xlabel("rewrites accepted by validation")
    legend(ax)
    return finish(fig, ax, "09_hinglish_rewrite_acceptance.png", "Hinglish rewrites that passed the quality checks",
                  "Share of Aya Expanse rewrites kept, first prompt vs fixed prompt, per task.")


def moving_avg(y, k=10):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="valid")


def chart_sft_loss():
    p = STAGES / "stage1_sft" / "train" / "train_log.json"
    if not p.exists():
        return None
    hist = [h for h in json.loads(p.read_text()).get("log_history", []) if "loss" in h and "step" in h]
    if not hist:
        return None
    steps, loss = [h["step"] for h in hist], [float(h["loss"]) for h in hist]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.plot(steps, loss, color=STAGE_COLOR["stage1_sft"], linewidth=2, marker="o", markersize=4)
    ax.text(steps[0], loss[0], f"  {loss[0]:.2f}", va="bottom", fontsize=8.5, color=INK)
    ax.text(steps[-1], loss[-1], f"  {loss[-1]:.2f}", va="bottom", fontsize=8.5, color=INK)
    ax.set_xlabel("training step (32 examples per step)")
    ax.set_ylabel("training loss")
    return finish(fig, ax, "10_sft_training_loss.png", "Stage 1 · Hindi SFT training loss",
                  f"{len(hist)} logged points; LoRA r=16 on {MODELS['policy'].split('/')[-1]}.", grid="y")


def chart_grpo_curves():
    p = STAGES / "stage2_grpo" / "train" / "train_log.json"
    if not p.exists():
        return None
    hist = [h for h in json.loads(p.read_text()).get("log_history", []) if "step" in h]
    rewards = ["no_badwords", "low_toxicity", "safe_reply", "refusal_calibration", "language_match", "format"]
    keys = {k for h in hist for k in h}
    prefix = "rewards/registry_" if any(k.startswith("rewards/registry_") for k in keys) else "rewards/"
    panels = [(f"{prefix}{r}/mean", r) for r in rewards]
    panels += [("reward", "total weighted reward"), ("kl", "KL drift from Stage 1"),
               ("completions/mean_length", "mean reply length (tokens)")]
    panels = [(k, t) for k, t in panels if any(k in h for h in hist)]
    if not panels:
        return None
    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows), squeeze=False)
    for ax, (key, title) in zip(axes.ravel(), panels):
        pts = [(h["step"], float(h[key])) for h in hist if key in h]
        x, y = zip(*pts)
        ax.plot(x, y, color=STAGE_COLOR["stage2_grpo"], linewidth=0.8, alpha=0.45)
        ma = moving_avg(y)
        ax.plot(x[len(x) - len(ma):], ma, color=STAGE_COLOR["stage2_grpo"], linewidth=2)
        ax.set_title(title, fontsize=10)
        ax.text(x[-1], ma[-1], f" {ma[-1]:.2f}", fontsize=8.5, color=INK, va="center")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    return finish(fig, axes, "11_grpo_training_curves.png", "Stage 2 · GRPO training curves",
                  "Per-step means logged by TRL during GRPO (thin) with a 10-step moving average (thick).", grid="y")


def main():
    stages = load_stages()
    made = []
    if stages:
        made += [chart_safety(stages), chart_hindi_vs_hinglish(stages), chart_over_refusal(stages),
                 chart_overview(stages), chart_env(stages), chart_milu(stages)]
    made += [chart_data_composition(), chart_filters(), chart_synthetic(), chart_sft_loss(), chart_grpo_curves()]
    made = [m for m in made if m]
    lines = ["# Plots", "", f"Stages included: {', '.join(stages) or 'none yet'}. Re-generate with "
             "`python -I scripts/make_plots.py` after each stage.", ""]
    lines += [f"![{m}]({m})" for m in made]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"{len(made)} plots -> {OUT}")
    for m in made:
        print("  ", m)


if __name__ == "__main__":
    main()
