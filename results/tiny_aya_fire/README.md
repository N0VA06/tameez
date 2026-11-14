# Tiny Aya Fire run

Policy: [`CohereLabs/tiny-aya-fire`](https://huggingface.co/CohereLabs/tiny-aya-fire), Cohere's South Asia variant. Final model: [IINOVAII/Tiny-Aya-Fire-Tameez-GRPO](https://huggingface.co/IINOVAII/Tiny-Aya-Fire-Tameez-GRPO) (GRPO step 300 of 300, the same fallback as Global: every checkpoint was above the 2.0% over-refusal cap). The same code, data, rewards and settings as the Global run.

## Results

| | Stage 0 · tiny-aya-fire | Stage 1 · Hindi fine-tune | **Stage 2 · Tameez (GRPO)** |
|---|---|---|---|
| Unsafe replies to harmful prompts ↓ | 5.1% | 19.9% | **4.3%** |
| … Hindi red-team prompts ↓ | 3.5% | 18.3% | **1.8%** |
| … Hinglish red-team prompts ↓ | 25.9% | 56.1% | 31.7% ⚠️ |
| Replies with bad words ↓ | 2.0% | 2.0% | **0.8%** |
| Refuses harmful prompts ↑ | 47.0% | 31.8% | **73.1%** |
| Over-refusal, all harmless prompts ↓ | 3.1% | 0.4% | 3.5% |
| Over-refusal, angry users with a real question ↓ | 7.5% | 1.0% | 8.0% |
| Answers Hinglish in Hinglish ↑ | 55.1% | 85.7% | **95.1%** |
| Hindi knowledge (MILU, 14,831 questions) ↑ | 46.9% | 43.6% | 45.0% |
| Helpfulness win rate vs Stage 1 (ties count ½) ↑ | – | – | **62.9%** |

The same story on Cohere's South Asia variant: plain SFT nearly quadrupled the unsafe replies, GRPO brought them back
just below the base level, more than doubled explicit refusals and was judged *more* helpful than the SFT model. Fire
keeps more Hindi knowledge than Global at every stage; Hinglish red-teaming is again the weak spot.

## What is in this folder

| Path | What it is |
|---|---|
| [`scorecard.md`](scorecard.md) / [`scorecard.csv`](scorecard.csv) | every metric of every stage with 95% Wilson intervals |
| [`plots/`](plots/) | the result and training plots of this run ([index](plots/README.md)) |
| [`auditkit_report.md`](auditkit_report.md) | AuditKIT audit of the saved scorecard replies |
| `stages/stage0_base/eval/` | base model scorecard: `metrics.json`, `milu.json`, `eval_args.json` |
| `stages/*/env/` | the RL environment before and after training: `env_check.json`, `rollout_report.json` |
| `stages/stage1_sft/train/` | Stage 1 (Hindi SFT) training log |
| `stages/stage1_sft/eval/`, `eval_mini/` | Stage 1 scorecard, and Stage 1 on the 200-prompt mini set (the checkpoint-selection reference) |
| `stages/stage2_grpo/train/` | Stage 2 (Tameez GRPO) training log and reward-sample summary |
| `stages/stage2_grpo/checkpoints/` | every GRPO checkpoint scored on the mini set; `checkpoints.json` holds the selection |
| `stages/stage2_grpo/eval/`, `judge/` | the final scorecard, and the helpfulness judge (Stage 2 vs Stage 1) |

Replies to harmful prompts are kept out of git. Every reply, judge verdict and log of this run is on Hugging Face:
[tameez-eval-outputs/tiny_aya_fire](https://huggingface.co/datasets/IINOVAII/tameez-eval-outputs/tree/main/tiny_aya_fire) and
[tameez-training-logs/tiny_aya_fire](https://huggingface.co/datasets/IINOVAII/tameez-training-logs/tree/main/tiny_aya_fire).
