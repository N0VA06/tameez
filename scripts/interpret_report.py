"""Tables and plots for the interpretability analysis (CPU).

    python -I scripts/interpret_report.py --raw results/interpretability/raw --out results/interpretability

Reads what scripts/interpret.py wrote (one folder per model and stage) and the scorecard replies on Hugging Face (for
which prompts each model actually refused), and writes tables.md, summary.json and four plots to --out/plots.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


EV = "IINOVAII/tameez-eval-outputs"
OLD_DATA = {"llama_3_2_3b": "data", "tiny_aya_global": "tiny_aya/data", "tiny_aya_fire": "tiny_aya_fire/data"}
NAMES = {"llama_3_2_3b": "Llama-3.2-3B", "tiny_aya_global": "Tiny Aya Global", "tiny_aya_fire": "Tiny Aya Fire"}
STAGE = {"stage0_base": "S0 base", "stage1_sft": "S1 Hindi SFT", "stage2_grpo": "S2 Tameez GRPO"}
COLORS = {"Tiny Aya Global": "#2a78d6", "Tiny Aya Fire": "#c2409a", "Llama-3.2-3B": "#eb6834"}
SCRIPT_COL = {"deva": "#2a78d6", "roman": "#eb6834", "en": "#6b6b6b"}
INK, MUTED, GRID, BG = "#222222", "#555555", "#e4e4e4", "#fcfcfa"
ORDER = ["llama_s0", "llama_s1", "llama_s2", "global_s0", "global_s1", "global_s2", "fire_s0", "fire_s1", "fire_s2"]
FINAL = ["llama_s2", "global_s2", "fire_s2"]


def read_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf8") if line.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def auroc(pos, neg):
    """Probability that a random positive scores above a random negative (ties count half); same as interpret.py."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def pct(k, n):
    return f"{100 * k / n:.1f}%" if n else "–"


def scorecard(run, stage, cache):
    """Per-prompt scorecard verdicts (refused, unsafe) of one model and stage, from tameez-eval-outputs."""
    from huggingface_hub import hf_hub_download
    for path in (f"{run}/data/scorecard_replies/{stage}.jsonl", f"{OLD_DATA[run]}/scorecard_replies/{stage}.jsonl"):
        try:
            return {r["id"]: r for r in read_jsonl(hf_hub_download(EV, path, repo_type="dataset", cache_dir=cache))}
        except Exception:
            continue
    return {}


def pick_layer(layers):
    L = len(layers)
    mid = [r for r in layers if round(0.3 * L) <= r["layer"] <= round(0.6 * L)]
    return max(mid, key=lambda r: r["cohen_d_deva_heldout"])["layer"]


def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="results/interpretability/raw")
    ap.add_argument("--out", default="results/interpretability")
    ap.add_argument("--cache", default=None, help="Hugging Face cache folder for the scorecard replies")
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    P = read_jsonl(raw / "prompts.jsonl")
    ix = {p["id"]: i for i, p in enumerate(P)}
    M = {}
    for k in ORDER:
        if (raw / k / "measures.json").exists():
            m = json.load(open(raw / k / "measures.json"))
            m["scores"] = dict(np.load(raw / k / "scores.npz"))
            ia = raw / k / "intervention_args.json"
            m["lay"] = json.load(open(ia))["layer"] if ia.exists() else pick_layer(m["layers"])
            j = raw / k / "judged.jsonl"
            m["judged"] = read_jsonl(j) if j.exists() else []
            m["sc"] = scorecard(m["run"], m["stage"], args.cache)
            M[k] = m
    lines, summary = [], {}

    # ---- 1. behaviour on the 139 twins (scorecard)
    lines += ["## 1. The same 139 requests in two scripts (scorecard replies)", "",
              "| Model | Stage | Unsafe, Devanagari | Unsafe, Roman | Refused, Devanagari | Refused, Roman |", "|---|---|---|---|---|---|"]
    twin_ids = sorted({p["pair"] for p in P if p["group"] == "test_twins"})
    for k, m in M.items():
        sc = m["sc"]
        if not sc:
            continue
        d = [sc[t] for t in twin_ids if t in sc]
        r = [sc[t + "-hin"] for t in twin_ids if t + "-hin" in sc]
        row = {"unsafe_deva": sum(x["unsafe"] for x in d), "unsafe_roman": sum(x["unsafe"] for x in r),
               "refused_deva": sum(x["refused"] for x in d), "refused_roman": sum(x["refused"] for x in r), "n": len(d)}
        summary.setdefault(k, {})["twins_behaviour"] = row
        lines.append(f"| {NAMES[m['run']]} | {STAGE[m['stage']]} | {pct(row['unsafe_deva'], len(d))} | {pct(row['unsafe_roman'], len(r))} | "
                     f"{pct(row['refused_deva'], len(d))} | {pct(row['refused_roman'], len(r))} |")

    # ---- 2. inside the model
    lines += ["", "## 2. Where the same requests land on the model's harm direction", "",
              "Position on the Devanagari harm direction (mean harmful − mean harmless Devanagari training prompts) at the "
              "steering layer: 0 = the average harmless Devanagari training prompt, 1 = the average harmful one. Test prompts: "
              "the 139 held-out twins and the 516 held-out harmless scorecard prompts; English harmful prompts come from the "
              "held-out part of the training pool (no English test twins exist). *Share* = (Roman harmful − Roman harmless) / "
              "(Devanagari harmful − Devanagari harmless).", "",
              "| Model | Stage | Layer | Harmful, Devanagari | Harmful, Roman (same requests) | Harmless, Roman | "
              "Harmless, Devanagari | Harmful, English (pool) | Share | cos(Deva dir, Roman dir) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in M.items():
        Ls = m["layers"][1:-1]          # layer 0 is the embedding; the last entry is after the final norm
        lay = m["lay"]
        s = m["layers"][lay]["score_deva_dir"]
        ratio = (s["twins_roman"] - s["benign_roman"]) / (s["twins_deva"] - s["benign_deva"])
        best = lambda key: max(r["auroc_probe"][key] for r in Ls)  # noqa: E731
        row = {"layer": lay, "n_layers": m["n_layers"], "roman_harm_signal": ratio, "positions": s,
               "cos_deva_roman": m["layers"][lay]["cos"]["deva_roman"], "cos_deva_en": m["layers"][lay]["cos"]["deva_en"],
               "probe": {key: best(key) for key in ("deva->deva", "deva->roman", "roman->roman", "deva->test_roman",
                                                     "roman->test_roman", "deva+roman->test_roman", "deva->en", "en->deva")},
               "fertility": m["fertility"]}
        summary.setdefault(k, {}).update(row)
        lines.append(f"| {NAMES[m['run']]} | {STAGE[m['stage']]} | {lay}/{m['n_layers']} | {s['twins_deva']:.2f} | {s['twins_roman']:.2f} | "
                     f"{s['benign_roman']:.2f} | {s['benign_deva']:.2f} | {s['heldout_h_en']:.2f} | {ratio:.2f} | {row['cos_deva_roman']:.2f} |")

    lines += ["", "## 3. Can the model tell harmful from harmless in each script?", "",
              "AUROC of linear probes (logistic regression on the last-token hidden state), best layer. Trained on the training "
              "pool (529 Devanagari/Roman pairs, 288 English prompts), tested on its held-out quarter (157 pairs, 112 English) or on the scorecard "
              "(139 Roman twins vs 106 harmless Roman prompts).", "",
              "| Model | Stage | Deva → Deva | Deva → Roman | Roman → Roman | Deva → English | Deva → Roman test | Roman → Roman test | Deva+Roman → Roman test |",
              "|---|---|---|---|---|---|---|---|---|"]
    for k, m in M.items():
        p = summary[k]["probe"]
        lines.append(f"| {NAMES[m['run']]} | {STAGE[m['stage']]} | " + " | ".join(f"{p[x]:.3f}" for x in (
            "deva->deva", "deva->roman", "roman->roman", "deva->en", "deva->test_roman", "roman->test_roman", "deva+roman->test_roman")) + " |")

    lines += ["", "### Same meaning across scripts, and tokens per word", "",
              "Share of prompts whose nearest neighbour in the other script is their own twin (raw text, hidden states averaged "
              "over tokens, each script centred), at the steering layer. Chance: 1 in 686, 139 or 40.", "",
              "| Model | Stage | Deva ↔ Roman (686 pool pairs) | Deva ↔ Roman (139 test pairs) | Deva ↔ English (40 seeds) | "
              "Roman ↔ English (40 seeds) | Tokens per word: Deva / Roman / English |", "|---|---|---|---|---|---|---|"]
    for k, m in M.items():
        al = m["layers"][m["lay"]]["align"]
        f = m["fertility"]
        summary[k]["align_at_layer"] = {key: al[key]["top1"] for key in al}
        lines.append(f"| {NAMES[m['run']]} | {STAGE[m['stage']]} | " +
                     " | ".join(f"{100 * al[key]['top1']:.1f}%" for key in ("rl_deva_roman", "test_deva_roman", "seeds_deva_en", "seeds_roman_en")) +
                     f" | {f['deva']['tokens_per_word']:.2f} / {f['roman']['tokens_per_word']:.2f} / {f['en']['tokens_per_word']:.2f} |")

    # ---- 4. does the direction track what the model does?
    lines += ["", "## 4. Does the harm direction predict refusal?", "",
              "AUROC of the position on the Devanagari harm direction (steering layer) for telling the twins the model refused "
              "from those it did not, in the scorecard. 0.5 = no relation.", "",
              "| Model | Stage | Devanagari twins | Roman twins |", "|---|---|---|---|"]
    for k, m in M.items():
        sc, sd = m["sc"], m["scores"]["deva_dir"][:, m["lay"]].astype(float)
        if not sc:
            continue
        res = {}
        for script, suffix in (("deva", ""), ("roman", "-hin")):
            ids = [t + suffix for t in twin_ids if t + suffix in sc]
            pos = [sd[ix[i]] for i in ids if sc[i]["refused"]]
            neg = [sd[ix[i]] for i in ids if not sc[i]["refused"]]
            res[script] = {"auroc": auroc(pos, neg), "n_refused": len(pos), "n": len(ids)}
        summary[k]["refusal_link"] = res
        fmt = lambda r: f"{r['auroc']:.2f} ({r['n_refused']} of {r['n']} refused)" if r["auroc"] is not None else f"– ({r['n_refused']} of {r['n']} refused)"  # noqa: E731
        lines.append(f"| {NAMES[m['run']]} | {STAGE[m['stage']]} | {fmt(res['deva'])} | {fmt(res['roman'])} |")

    # ---- 5. interventions on the final models
    lines += ["", "## 5. Interventions on the final models (Llama-Guard-3-8B + the scorecard refusal matcher)", "",
              "Greedy replies, 256 new tokens, transformers bf16, on the 139 twins in both scripts and the 516 harmless scorecard "
              "prompts. *ablate*: the Devanagari direction removed from the embeddings and from the output of every block. "
              "*add α*: α × the direction (unnormalised, so α = 1 moves a harmless prompt to the harmful mean) added at the "
              "steering layer, every position. *gated α*: the *add α* reply when a probe trained on Devanagari + Roman training "
              "prompts flags the prompt, otherwise the *none* reply. The probe's layer (best held-out Roman AUROC) and threshold "
              "(0, the balanced decision boundary) were fixed before the replies were judged.", "",
              "| Model | Condition | Roman harmful: unsafe | Roman harmful: refused | Deva harmful: unsafe | Deva harmful: refused | "
              "Harmless refused: Roman | Deva | English |", "|---|---|---|---|---|---|---|---|---|"]
    for k in FINAL:
        if k not in M or not M[k]["judged"]:
            continue
        m = M[k]
        J = {(r["cond"], r["id"]): r for r in m["judged"]}
        Ls = m["layers"][1:-1]
        lp = max(Ls, key=lambda r: r["auroc_probe"]["deva+roman->roman"])["layer"]
        gate = m["scores"]["probe_deva_roman"][:, lp].astype(float) > 0
        conds = ["none", "ablate"] + [c for c in ("add_0.5", "add_1.0") if any(cc == c for cc, _ in J)]
        rows = {c: [r for (cc, _), r in J.items() if cc == c] for c in conds}
        base = {r["id"]: r for r in rows["none"]}
        for c in [c for c in conds if c.startswith("add_")]:
            add = {r["id"]: r for r in rows[c]}
            rows["gated_" + c[4:]] = [add[i] if gate[ix[i]] else base[i] for i in base if i in add]
        res = {}
        for c, rs in rows.items():
            def sel(script, harmful):
                return [r for r in rs if r["script"] == script and r["harmful"] == harmful]
            cell = {}
            for name, script, harmful, key in (("roman_unsafe", "roman", True, "unsafe"), ("roman_refused", "roman", True, "refused"),
                                               ("deva_unsafe", "deva", True, "unsafe"), ("deva_refused", "deva", True, "refused"),
                                               ("benign_roman", "roman", False, "refused"), ("benign_deva", "deva", False, "refused"),
                                               ("benign_en", "en", False, "refused")):
                xs = sel(script, harmful)
                cell[name] = (sum(bool(r[key]) for r in xs), len(xs))
            res[c] = cell
            lines.append(f"| {NAMES[m['run']]} | {c.replace('_', ' ')} | " + " | ".join(pct(*cell[n]) for n in cell) + " |")
        flags = {g: float(np.mean([gate[ix[r["id"]]] for r in rows["none"] if (r["script"], r["harmful"]) == g]))
                 for g in (("roman", True), ("deva", True), ("roman", False), ("deva", False), ("en", False))}
        summary[k]["interventions"] = {c: {n: [*v, *wilson(*v)] for n, v in cell.items()} for c, cell in res.items()}  # k, n, 95% CI
        summary[k]["gate"] = {"layer": lp, "flag_rate": {f"{s}_{'harmful' if h else 'harmless'}": v for (s, h), v in flags.items()}}
    gates = [k for k in FINAL if "gate" in summary.get(k, {})]
    if gates:
        lines += ["", "Share of prompts the gate flags (the *gated* rows add the direction only to these):", "",
                  "| Model | Probe layer | Harmful Roman | Harmful Devanagari | Harmless Roman | Harmless Devanagari | Harmless English |",
                  "|---|---|---|---|---|---|---|"]
        for k in gates:
            g = summary[k]["gate"]
            f = g["flag_rate"]
            lines.append(f"| {NAMES[M[k]['run']]} | {g['layer']} | " + " | ".join(f"{100 * f[x]:.0f}%" for x in (
                "roman_harmful", "deva_harmful", "roman_harmless", "deva_harmless", "en_harmless")) + " |")
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    json.dump(summary, open(out / "summary.json", "w"), indent=1, default=float)
    plots(M, P, out / "plots", summary)
    print("\n".join(lines))


# ---------------------------------------------------------------- plots

def plots(M, P, out, summary):
    fin = [k for k in FINAL if k in M]
    if not fin:
        return
    # 1. harm signal by layer, final models
    fig, axes = plt.subplots(1, len(fin), figsize=(4.2 * len(fin), 3.6), sharey=True)
    fig.patch.set_facecolor(BG)
    for ax, k in zip(np.atleast_1d(axes), fin):
        m = M[k]
        style(ax)
        Ls = m["layers"][1:-1]
        x = [r["layer"] for r in Ls]
        for key, label, col, ls in (("twins_deva", "harmful, Devanagari", SCRIPT_COL["deva"], "-"),
                                    ("twins_roman", "harmful, Roman (same requests)", SCRIPT_COL["roman"], "-"),
                                    ("benign_deva", "harmless, Devanagari", SCRIPT_COL["deva"], "--"),
                                    ("benign_roman", "harmless, Roman", SCRIPT_COL["roman"], "--")):
            ax.plot(x, [r["score_deva_dir"][key] for r in Ls], color=col, ls=ls, lw=2, label=label)
        ax.axvline(m["lay"], color=MUTED, lw=0.8, ls=":")
        ax.set_title(NAMES[m["run"]] + " (final)", fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("layer", color=MUTED, fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("position on the harm direction\n0 = harmless avg, 1 = harmful avg (Devanagari)", color=MUTED, fontsize=8.5)
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Same harmful requests, weaker harm signal: Roman copies land well below their Devanagari originals",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "interp_01_harm_direction_by_layer.png", dpi=150, facecolor=BG)
    plt.close(fig)

    # 2. probe transfer by layer
    fig, axes = plt.subplots(1, len(fin), figsize=(4.2 * len(fin), 3.4), sharey=True)
    fig.patch.set_facecolor(BG)
    for ax, k in zip(np.atleast_1d(axes), fin):
        m = M[k]
        style(ax)
        Ls = m["layers"][1:-1]
        x = [r["layer"] for r in Ls]
        for key, label, col, ls in (("deva->deva", "Devanagari probe → Devanagari", SCRIPT_COL["deva"], "-"),
                                    ("deva->roman", "Devanagari probe → Roman", SCRIPT_COL["roman"], "--"),
                                    ("roman->roman", "Roman probe → Roman", SCRIPT_COL["roman"], "-"),
                                    ("deva->en", "Devanagari probe → English", SCRIPT_COL["en"], ":")):
            ax.plot(x, [r["auroc_probe"][key] for r in Ls], color=col, ls=ls, lw=2, label=label)
        ax.set_ylim(0.45, 1.01)
        ax.set_title(NAMES[m["run"]] + " (final)", fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("layer", color=MUTED, fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("harmful vs harmless, AUROC (held out)", color=MUTED, fontsize=8.5)
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Linear probes for harmful vs harmless: trained in one script, tested in another", x=0.01, ha="left",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "interp_02_probe_transfer.png", dpi=150, facecolor=BG)
    plt.close(fig)

    # 3. meaning alignment by relative depth, base and final
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), sharey=True)
    fig.patch.set_facecolor(BG)
    for ax, (key, title) in zip(axes, (("rl_deva_roman", "Devanagari ↔ Roman (686 pairs)"), ("seeds_deva_en", "Devanagari ↔ English (40 seeds)"))):
        style(ax)
        for k, m in M.items():
            if m["stage"] not in ("stage0_base", "stage2_grpo"):
                continue
            Ls = m["layers"][1:-1]
            x = [r["layer"] / m["n_layers"] for r in Ls]
            ax.plot(x, [100 * r["align"][key]["top1"] for r in Ls], color=COLORS[NAMES[m["run"]]],
                    ls="-" if m["stage"] == "stage0_base" else "--", lw=2,
                    label=f"{NAMES[m['run']]} {'base' if m['stage'] == 'stage0_base' else 'final'}")
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.set_xlabel("relative depth", color=MUTED, fontsize=9)
    axes[0].set_ylabel("nearest neighbour is the twin, %", color=MUTED, fontsize=8.5)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Does a prompt mean the same thing to the model in another script?", x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / "interp_03_script_alignment.png", dpi=150, facecolor=BG)
    plt.close(fig)

    # 4. interventions
    have = [k for k in fin if "interventions" in summary.get(k, {})]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(4.3 * len(have), 3.8), sharey=True)
    fig.patch.set_facecolor(BG)
    metrics = (("roman_refused", "Roman harmful: refused", SCRIPT_COL["roman"]), ("roman_unsafe", "Roman harmful: unsafe", "#8a3b12"),
               ("benign_roman", "Roman harmless: refused", "#9aa4b1"), ("benign_deva", "Devanagari harmless: refused", "#4b5563"))
    for ax, k in zip(np.atleast_1d(axes), have):
        iv = summary[k]["interventions"]
        conds = [c for c in ("none", "add_0.5", "gated_0.5", "add_1.0", "gated_1.0") if c in iv]
        style(ax)
        w = 0.8 / len(metrics)
        for j, (key, label, col) in enumerate(metrics):
            vals = [100 * iv[c][key][0] / max(1, iv[c][key][1]) for c in conds]
            ax.bar(np.arange(len(conds)) + (j - (len(metrics) - 1) / 2) * w, vals, w * 0.92, color=col, label=label)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c.replace("_", " ") for c in conds], fontsize=8.5, color=INK)
        ax.set_title(NAMES[M[k]["run"]] + " (final)", fontsize=10.5, color=INK, loc="left")
    np.atleast_1d(axes)[0].set_ylabel("%", color=MUTED)
    h, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Adding the Devanagari harm direction to Roman prompts: always, or only when a probe flags them",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "interp_04_interventions.png", dpi=150, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
