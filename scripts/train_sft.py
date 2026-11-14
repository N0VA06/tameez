"""Stage 1: normal Hindi SFT of the base model on Set A with AlignTune, then merge the adapter. GPU.

Outputs (stages/stage1_sft/):
    model/adapter/          LoRA adapter (+ intermediate checkpoints, TensorBoard logs)
    model/merged/           merged bf16 model (input to Stage 2 and to evaluate.py)
    train/train_log.json    loss curve (trainer log history) + run config

Defaults are sized for one 24 GB L4 (~50 min): a 5,000-row stratified sample of Set A (every
Hinglish and Aya row + a random mix of Dolly-T and OASST-T), max length 768, batch 4 x 8 accumulation.
Use --max-samples 0 for all ~14K rows (plan setting, ~2.5 h on an L4).

Usage:
    python scripts/train_sft.py --max-steps 5 --out stages/sft_smoke --no-merge   # smoke test
    python scripts/train_sft.py                                                    # Stage 1
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tameez.common import MODELS, PROCESSED, SEED, STAGE1, acquire_gpu, get_logger, read_jsonl, timed, write_json, write_jsonl  # noqa: E402

from tameez.merge import merge_adapter  # noqa: E402

log = get_logger("train_sft")

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def stratified_subset(path, n, out_path):
    """Keep every Hinglish and Aya row (scarce), fill the rest randomly from Dolly-T / OASST-T.

    A_sft.json is stored in source order, so taking the first n rows would be Dolly only.
    """
    rows = read_jsonl(path)
    rng = random.Random(SEED)
    keep = [r for r in rows if r["lang"] == "hin" or r["source"] == "aya_dataset"]
    rest = [r for r in rows if r not in keep]
    rng.shuffle(rest)
    rng.shuffle(keep)
    subset = keep[:n] + rest[:max(0, n - len(keep))]       # never more than n rows
    rng.shuffle(subset)
    write_jsonl(subset, out_path)
    from collections import Counter
    log.info(f"SFT subset: {len(subset)} of {len(rows)} rows -> {dict(Counter(r['source'] for r in subset))}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODELS["policy"])
    ap.add_argument("--data", default=str(PROCESSED / "A_sft.json"))
    ap.add_argument("--out", default=str(STAGE1))
    ap.add_argument("--backend", default="trl", choices=["trl", "unsloth"])
    ap.add_argument("--epochs", type=float, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-samples", type=int, default=5000, help="stratified sample of Set A (0 = all rows)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq-length", type=int, default=768)
    ap.add_argument("--no-merge", action="store_true")
    args = ap.parse_args()

    log.info(f"args: {vars(args)}")
    acquire_gpu(log)
    from aligntune.core.backend_factory import create_sft_trainer

    out = Path(args.out)
    adapter_dir = out / "model" / "adapter"
    data = args.data
    if args.max_samples:
        data = str(stratified_subset(args.data, args.max_samples, out / "train" / "sft_subset.json"))
    trainer = create_sft_trainer(
        model_name=args.model,
        dataset_name=data,
        backend=args.backend,
        output_dir=str(adapter_dir),
        task_type="instruction_following",
        dataset_text_field="messages",
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_seq_length=args.max_seq_length,
        use_peft=True,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        lora_target_modules=LORA_TARGETS,
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        report_to="tensorboard",
        seed=SEED,
    )
    t0 = time.time()
    with timed(log, "SFT training (progress bar + loss every 10 steps below)"):
        result = trainer.train()
    trainer.save_model(str(adapter_dir))
    log.info(f"adapter saved to {adapter_dir}")

    hf_trainer = getattr(trainer, "trainer", None)
    record = {"args": vars(args), "minutes": round((time.time() - t0) / 60, 1),
           "result": result if isinstance(result, dict) else str(result),
           "log_history": getattr(getattr(hf_trainer, "state", None), "log_history", [])}
    write_json(json.loads(json.dumps(record, default=str)), out / "train" / "train_log.json")

    if not args.no_merge:
        with timed(log, "merge LoRA into base"):
            merge_adapter(args.model, adapter_dir, out / "model" / "merged")
        log.info(f"merged model -> {out / 'model' / 'merged'}")


if __name__ == "__main__":
    main()
