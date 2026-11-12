"""Audit the curated Tameez data splits with Lexsi Labs' CuratorKIT. CPU only, nothing is regenerated.

    pip install 'curatorkit[hygiene]'
    hf download IINOVAII/tameez-hindi-safety-data --repo-type dataset --local-dir data/hf_safety_data
    python -I scripts/curate_with_curatorkit.py --data data/hf_safety_data --out results/curatorkit_report.md

The splits were built by scripts/build_data.py (our own MinHash pass via datasketch). This script re-checks them
independently with CuratorKIT:

- exact and near-duplicates inside each training split (Set A `sft`, Set B `rl_prompts`), with CuratorKIT's
  ExactDeduplicator and MinHashDeduplicator;
- leakage: training prompts that are near-duplicates of a held-out test prompt (Set C `safety_test`, Set D
  `aya_eval_hindi` and `milu_hindi`). Test prompts go first into one MinHashDeduplicator run, so any training prompt
  it drops is a near-copy of a test prompt (or of a training prompt already dropped on its own);
- credentials, API keys and high-entropy secrets in every split, with CuratorKIT's SecretsGate.

The report holds counts and ids only, no prompt text.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from curatorkit.hygiene.secrets import SecretsGate
from curatorkit.normalizers.dedup import ExactDeduplicator, MinHashDeduplicator
from curatorkit.schema import DataSample

TRAIN = {"sft": "Set A (SFT data)", "rl_prompts": "Set B (RL prompt pool)"}
TEST = {"safety_test": "Set C (safety test)", "aya_eval_hindi": "Set D (Aya-Eval Hindi)",
        "milu_hindi": "Set D (MILU Hindi)"}
THRESHOLD = 0.7  # the threshold build_data.py used for its own leakage pass


def read_split(data, name):
    files = sorted((Path(data) / "data" / name).glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no split '{name}' under {data}/data/ (download the dataset first, see the docstring)")
    return [json.loads(line) for f in files for line in f.open(encoding="utf-8") if line.strip()]


def user_text(row):
    if "question" in row:  # MILU multiple choice
        opts = " ".join(str(row.get(f"option{i}", "")) for i in range(1, 5))
        return f"{row['question']} {opts}".strip()
    msgs = row.get("prompt") or row.get("messages") or []
    users = [m["content"] for m in msgs if m.get("role") == "user"]
    return users[-1] if users else ""


def reply_text(row):
    msgs = row.get("messages") or []
    replies = [m["content"] for m in msgs if m.get("role") == "assistant"]
    return replies[-1] if replies else None


def to_samples(rows, split, with_reply=False):
    out = []
    for i, r in enumerate(rows):
        reply = reply_text(r) if with_reply else None
        out.append(DataSample(id=f"{split}:{r.get('id', i)}", source_uri=f"tameez-hindi-safety-data/{split}",
                              instruction=user_text(r), output=reply or "",
                              task_type="instruction_following" if reply else "prompt_only"))
    return out


def dropped(before, after):
    kept = {s.id for s in after}
    return [s.id for s in before if s.id not in kept]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/hf_safety_data", help="local copy of IINOVAII/tameez-hindi-safety-data")
    ap.add_argument("--out", default="results/curatorkit_report.md")
    args = ap.parse_args()
    t0 = time.time()

    rows = {name: read_split(args.data, name) for name in [*TRAIN, *TEST]}
    prompts = {name: to_samples(r, name) for name, r in rows.items()}

    # 1. duplicates inside each training split (prompt only, and prompt + reply for SFT)
    within = {}
    for name in TRAIN:
        s = prompts[name]
        within[name] = {"n": len(s),
                        "exact": len(dropped(s, ExactDeduplicator().run(s))),
                        "near": len(dropped(s, MinHashDeduplicator(threshold=THRESHOLD).run(s)))}
    sft_pairs = to_samples(rows["sft"], "sft", with_reply=True)
    within["sft"]["exact_with_reply"] = len(dropped(sft_pairs, ExactDeduplicator().run(sft_pairs)))

    # 2. leakage: test prompts first, then training prompts, one MinHash pass per (train, test) pair
    leak = {}
    for tr in TRAIN:
        self_dropped = set(dropped(prompts[tr], MinHashDeduplicator(threshold=THRESHOLD).run(prompts[tr])))
        for te in TEST:
            combined = prompts[te] + prompts[tr]
            gone = [i for i in dropped(combined, MinHashDeduplicator(threshold=THRESHOLD).run(combined))
                    if i.startswith(f"{tr}:") and i not in self_dropped]
            leak[(tr, te)] = gone

    # 3. secrets in every split (prompt + reply)
    secrets = {}
    for name in rows:
        samples = sft_pairs if name == "sft" else prompts[name]
        _, rejected = SecretsGate().run(samples)
        reasons = Counter(str(getattr(r, "reason", getattr(r, "rejection_reason", "secret")))[:60] for r in rejected)
        secrets[name] = (len(samples), len(rejected), reasons)

    # report
    L = ["# CuratorKIT audit: tameez-hindi-safety-data", "",
         f"CuratorKIT {__import__('curatorkit').__version__}, run on the published splits (CPU, "
         f"{(time.time() - t0) / 60:.1f} min). Near-duplicate threshold: MinHash Jaccard ≥ {THRESHOLD}, character "
         "3-grams, 128 permutations. Counts and ids only, no prompt text.", "",
         "## Duplicates inside the training splits", "",
         "| Split | Rows | Exact duplicate prompts | Near-duplicate prompts | Exact duplicate (prompt + reply) |",
         "|---|---|---|---|---|"]
    for name, label in TRAIN.items():
        w = within[name]
        L.append(f"| {label} | {w['n']:,} | {w['exact']:,} | {w['near']:,} | {w.get('exact_with_reply', '–')} |")
    L += ["", "## Leakage: training prompts that near-duplicate a test prompt", "",
          "| Training split | " + " | ".join(TEST.values()) + " |", "|---|" + "---|" * len(TEST)]
    for tr, label in TRAIN.items():
        L.append(f"| {label} | " + " | ".join(str(len(leak[(tr, te)])) for te in TEST) + " |")
    examples = [i for v in leak.values() for i in v][:20]
    if examples:
        L += ["", "Flagged training ids (first 20): " + ", ".join(f"`{i}`" for i in examples)]
    L += ["", "## Secrets (CuratorKIT SecretsGate)", "", "| Split | Rows scanned | Flagged | Reasons |", "|---|---|---|---|"]
    for name, (n, k, reasons) in secrets.items():
        label = TRAIN.get(name) or TEST.get(name)
        L.append(f"| {label} | {n:,} | {k} | {', '.join(f'{r} ({c})' for r, c in reasons.most_common(3)) or '–'} |")
    L += ["", "Not checked here: PII. CuratorKIT's PII pseudonymiser is built on English entity recognition, so it would "
          "say little about Devanagari Hindi or Roman Hinglish text."]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
