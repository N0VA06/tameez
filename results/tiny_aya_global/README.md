# Tiny Aya Global run (main model)

Policy: [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global). Final model: [IINOVAII/Tiny-Aya-Global-Tameez-GRPO](https://huggingface.co/IINOVAII/Tiny-Aya-Global-Tameez-GRPO) (GRPO step 300 of 300, the lowest-unsafe checkpoint after every checkpoint missed the over-refusal cap).

## Results

| | Stage 0 · tiny-aya-global | Stage 1 · Hindi fine-tune | **Stage 2 · Tameez (GRPO)** |
|---|---|---|---|
| Unsafe replies to harmful prompts ↓ | 4.5% | 20.7% | **3.9%** |
| … Hindi red-team prompts ↓ | 3.9% | 19.8% | **1.5%** |
| … Hinglish red-team prompts ↓ | 18.7% | 57.6% | 29.5% ⚠️ |
| Replies with bad words ↓ | 2.5% | 2.1% | **0.6%** |
| Refuses harmful prompts ↑ | 39.7% | 33.1% | **73.6%** |
| Over-refusal, all harmless prompts ↓ | 2.5% | 0.8% | 3.9% |
| Over-refusal, angry users with a real question ↓ | 5.5% | 2.0% | 9.0% |
| Answers Hinglish in Hinglish ↑ | 50.2% | 89.4% | **94.3%** |
| Hindi knowledge (MILU, 14,831 questions) ↑ | 45.4% | 41.3% | 43.2% |
| Helpfulness win rate vs Stage 1 (ties count ½) ↑ | – | – | **61.5%** |

Plain SFT more than quadrupled Tiny Aya's unsafe replies; GRPO brought them back to about the base level, doubled
explicit refusals and was judged *more* helpful than the SFT model. Hinglish red-teaming remains the weak spot.

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
[tameez-eval-outputs/tiny_aya_global](https://huggingface.co/datasets/IINOVAII/tameez-eval-outputs/tree/main/tiny_aya_global) and
[tameez-training-logs/tiny_aya_global](https://huggingface.co/datasets/IINOVAII/tameez-training-logs/tree/main/tiny_aya_global).
