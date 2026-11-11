"""Join the sampled reward logs of a GRPO run and export N completions per bucket for reading.

Output: <dir>/train/samples_by_bucket.md (completion + every reward value) and samples_summary.json
(mean reward per bucket). Completions of harmful prompts may contain unsafe text: read, do not publish.

Usage:
    python -I scripts/show_samples.py --dir stages/stage2_grpo/pilot --per-bucket 30
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tameez.common import SEED, STAGE2, read_jsonl, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(STAGE2), help="GRPO run folder")
    ap.add_argument("--per-bucket", type=int, default=30)
    ap.add_argument("--last-minutes", type=float, default=0, help="only completions from the end of training")
    args = ap.parse_args()

    run = Path(args.dir)
    recs = read_jsonl(run / "train" / "reward_samples.json")
    if args.last_minutes:
        t_end = max(r["t"] for r in recs)
        recs = [r for r in recs if r["t"] >= t_end - 60 * args.last_minutes]
    by_key = {}
    for r in recs:
        e = by_key.setdefault(r["key"], {k: r[k] for k in ("bucket", "lang", "prompt", "completion")} | {"rewards": {}})
        e["rewards"][r["reward"]] = r["value"]

    buckets = defaultdict(list)
    for e in by_key.values():
        buckets[e["bucket"]].append(e)
    rng = random.Random(SEED)
    lines, summary = [f"# Sampled completions: {run}", ""], {}
    for b, items in sorted(buckets.items()):
        names = sorted({k for e in items for k in e["rewards"]})
        summary[b] = {"n": len(items), **{k: round(sum(e["rewards"].get(k) or 0 for e in items) / len(items), 3) for k in names}}
        lines += [f"## {b} ({len(items)} logged)", ""]
        for e in rng.sample(items, min(args.per_bucket, len(items))):
            scores = ", ".join(f"{k}={v:+.2f}" for k, v in sorted(e["rewards"].items()) if v is not None)
            lines += [f"**[{e['lang']}] prompt:** {e['prompt']}", "", f"**reply:** {e['completion']}", "", f"`{scores}`", "", "---", ""]
    out = run / "train"
    out.mkdir(parents=True, exist_ok=True)
    (out / "samples_by_bucket.md").write_text("\n".join(lines), encoding="utf8")
    write_json(summary, out / "samples_summary.json")
    print(json.dumps(summary, indent=1))
    print(f"-> {out / 'samples_by_bucket.md'}")


if __name__ == "__main__":
    main()
