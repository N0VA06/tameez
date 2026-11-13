"""Stage 2b: GRPO (Dr. GRPO loss) on the Set B prompt pool with the six training rewards. GPU.

Starts from the Stage 1 merged model. Prompts and rewards come from the RL environment
(tameez.env.TameezEnv). Every reward call is also logged (sampled) to train/reward_samples.json
so completions can be read per bucket with show_samples.py.

Outputs (<out>/, default stages/stage2_grpo/):
    model/checkpoint-*/        LoRA checkpoints every --save-steps
    model/adapter/, merged/    final adapter and merged model
    train/train_log.json       per-step reward means, KL, loss, completion length (TRL log history)
    train/reward_samples.json  sampled (bucket, lang, reward, value, completion) records

Defaults are sized for one 24 GB L4 (measured: Llama 2 h 17 min at ~27 s/step, Tiny Aya 3 h 35 min at ~44 s/step): 300 steps, 4 prompts x 4 generations per step
(batch 8 x 2 accumulation), completions up to 256 tokens, learning rate 5e-5, checkpoint every 50 steps.
Why: at the plan's 5e-6 (and at 2e-5 over 150 steps) the KL to Stage 1 stayed below 0.003 and held-out changes
were within noise, i.e. the LoRA barely moved; no reward hacking was seen (stable reply length, non-zero
reward variance in every group). The beta = 0.04 KL penalty is kept as the guard against drifting too far.
Plan setting (A100): --max-steps 400 --num-generations 8 --grad-accum 4 --max-completion-length 384.

Usage:
    python scripts/train_grpo.py --out stages/stage2_grpo/pilot --max-steps 20 --save-steps 10   # pilot
    python scripts/train_grpo.py                                                                # Stage 2
    python scripts/train_grpo.py --out stages/ablations/no_refusal --max-steps 60 --exclude refusal_calibration
"""

import argparse
import functools
import json
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tameez import rewards  # noqa: E402
from tameez.env import TameezEnv  # noqa: E402
from tameez.common import PROCESSED, SEED, STAGE1, STAGE2, SYSTEM_PROMPT, acquire_gpu, get_logger, timed, write_json  # noqa: E402

from tameez.merge import merge_adapter  # noqa: E402

log = get_logger("train_grpo")

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class SampleLog:
    """Append a random fraction of reward calls to a JSON Lines file."""

    def __init__(self, path, rate):
        self.path, self.rate, self.lock = Path(path), rate, threading.Lock()
        self.rng = random.Random(SEED)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def wrap(self, fn):
        @functools.wraps(fn)
        def logged(text, **kwargs):
            value = fn(text, **kwargs)
            key = hash(text)
            if (key % 1000) < self.rate * 1000:  # same completions sampled across all rewards
                rec = {"t": round(time.time()), "reward": fn.__name__, "value": value, "bucket": kwargs.get("bucket"),
                       "lang": kwargs.get("lang"), "key": key, "prompt": rewards.last_user_text(kwargs.get("prompt"))[:300],
                       "completion": text[:1500]}
                with self.lock, open(self.path, "a", encoding="utf8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return value
        return logged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(STAGE2), help="run folder (stage2_grpo, its pilot/, or an ablation)")
    ap.add_argument("--model", default=str(STAGE1 / "model" / "merged"))
    ap.add_argument("--data", default=str(PROCESSED / "B_pool.json"))
    # trl + hf: Unsloth patches Llama attention globally, which breaks the Llama-Guard reward model
    # loaded in the same process ("no attribute apply_qkv"); TRL's vLLM rollout is not detected inside
    # this process. HF rollouts measured at ~27-44 s/step on an L4 (16 completions, 256 tokens).
    ap.add_argument("--backend", default="trl", choices=["trl", "unsloth"])
    ap.add_argument("--rollout", default="hf", choices=["hf", "vllm", "unsloth_fast"],
                    help="generation engine for rollouts (vllm = TRL colocated vLLM)")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--num-generations", type=int, default=4, help="8 on an A100, 4 on a 24 GB GPU")
    ap.add_argument("--batch-size", type=int, default=8, help="completions per micro-batch (= 2 prompts x 4)")
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5, help="plan: 5e-6; see module docstring for why 5e-5")
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--max-completion-length", type=int, default=256)
    ap.add_argument("--vllm-mem", type=float, default=0.35,
                    help="GPU share for colocated vLLM rollouts (3B weights alone need ~6.4 GB)")
    ap.add_argument("--exclude", nargs="*", default=[], help="reward names to drop (ablations)")
    ap.add_argument("--sample-rate", type=float, default=0.05, help="fraction of completions logged")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--resume-from", help="checkpoint folder (with optimizer.pt, scheduler.pt, rng_state.pth, trainer_state.json) to continue a run that was cut off")
    args = ap.parse_args()

    log.info(f"args: {vars(args)}")
    acquire_gpu(log)
    from aligntune.core.backend_factory import create_rl_trainer

    out = Path(args.out)
    env = TameezEnv(args.data, exclude=args.exclude)
    problems = env.validate()
    if problems:
        raise SystemExit(f"prompt pool failed validation ({len(problems)} problems), e.g. {problems[:3]}")
    log.info(f"environment: {env.stats()}")
    sample_log = SampleLog(out / "train" / "reward_samples.json", args.sample_rate)
    specs = env.reward_specs(wrap=sample_log.wrap)
    log.info(f"rewards: {[(s['params']['function'].__name__, s['weight']) for s in specs]}")

    rollout = {"vllm": {"rollout_backend": "vllm", "vllm_mode": "colocate", "vllm_gpu_memory_utilization": args.vllm_mem},
               "hf": {"rollout_backend": "hf"},
               "unsloth_fast": {"fast_inference": True, "vllm_gpu_memory_utilization": args.vllm_mem}}[args.rollout]
    trainer = create_rl_trainer(
        model_name=args.model,
        dataset_name=str(env.pool_path),
        algorithm="grpo",
        backend=args.backend,
        output_dir=str(out / "model"),
        system_prompt=SYSTEM_PROMPT,
        loss_type="dr_grpo",
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=0.9,
        top_p=0.95,
        max_prompt_length=512,
        max_completion_length=args.max_completion_length,
        max_seq_length=768,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        save_total_limit=None,
        logging_steps=1,
        use_peft=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_target_modules=LORA_TARGETS,
        bf16=True,
        gradient_checkpointing=True,
        rewards=specs,
        report_to="tensorboard",
        seed=SEED,
        **rollout,
    )
    if args.resume_from:
        # AlignTune calls TRL's trainer.train() without arguments; pass the checkpoint through TRL's own resume path
        import trl
        _train = trl.GRPOTrainer.train

        def _resume(self, *a, **k):
            k.setdefault("resume_from_checkpoint", args.resume_from)
            return _train(self, *a, **k)
        trl.GRPOTrainer.train = _resume
        log.info(f"resuming from {args.resume_from}")
    t0 = time.time()
    with timed(log, f"GRPO {out} ({args.max_steps} steps; per-reward means logged every step)"):
        result = trainer.train()
    trainer.save_model(str(out / "model" / "adapter"))
    log.info(f"adapter saved to {out / 'model' / 'adapter'}")

    hf_trainer = getattr(trainer, "trainer", None)
    history = getattr(getattr(hf_trainer, "state", None), "log_history", [])
    write_json(json.loads(json.dumps({"args": vars(args), "minutes": round((time.time() - t0) / 60, 1),
                                      "result": result if isinstance(result, dict) else str(result),
                                      "log_history": history}, default=str)),
               out / "train" / "train_log.json")

    if not args.no_merge:
        with timed(log, "merge LoRA into Stage 1 model"):
            merge_adapter(args.model, out / "model" / "adapter", out / "model" / "merged")
        log.info(f"merged model -> {out / 'model' / 'merged'}")


if __name__ == "__main__":
    main()
