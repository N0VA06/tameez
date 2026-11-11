"""Inspect the GRPO environment (TameezEnv): validate the pool, show its composition, and
measure the reward a model already earns in it (rollout) before any GRPO.

Outputs (stages/<stage>/env/):
    env_check.json       validation problems + composition + reward weights
    rollout_report.json  mean of every reward per bucket / language, refusal + script rates (with --rollout)
    rollout.json         prompts, replies and scores (git-ignored; may contain unsafe text)

Usage:
    python -I scripts/env_check.py                                   # validate + stats (CPU)
    python scripts/env_check.py --rollout --stage stage1_sft         # + rollout of that stage's model (GPU)
    python scripts/env_check.py --rollout --stage stage0_base --model unsloth/Llama-3.2-3B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tameez.common import MODELS, STAGES, acquire_gpu, get_logger, timed, write_json, write_jsonl  # noqa: E402
from tameez.env import TameezEnv  # noqa: E402

log = get_logger("env_check")
DEFAULT_MODEL = {"stage0_base": MODELS["policy"], "stage1_sft": str(STAGES / "stage1_sft" / "model" / "merged"),
                 "stage2_grpo": str(STAGES / "stage2_grpo" / "model" / "best_merged")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="stage1_sft")
    ap.add_argument("--model", help="defaults to the stage's merged model")
    ap.add_argument("--rollout", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    env = TameezEnv()
    out = STAGES / args.stage / "env"
    problems = env.validate()
    check = {**env.describe(), "stats": env.stats(), "problems": problems[:50], "n_problems": len(problems)}
    write_json(check, out / "env_check.json")
    log.info(f"pool: {check['stats']}")
    log.info(f"validation: {'OK' if not problems else f'{len(problems)} problems, first: {problems[:5]}'}")

    if args.rollout:
        model = args.model or DEFAULT_MODEL.get(args.stage, MODELS["policy"])
        acquire_gpu(log)
        with timed(log, f"rollout of {model} on {args.n} stratified prompts"):
            rows, replies, scores, report = env.rollout(model, n=args.n, log=log)
        write_jsonl([{"bucket": r["bucket"], "lang": r["lang"], "prompt": r["prompt"][-1]["content"], "reply": rep, **sc}
                     for r, rep, sc in zip(rows, replies, scores)], out / "rollout.json")
        write_json({"model": model, "n": len(rows), "report": report}, out / "rollout_report.json")
        log.info("rollout report:\n" + json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
