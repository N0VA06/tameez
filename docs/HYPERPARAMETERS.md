# Hyperparameters and settings

Identical for all three base models unless noted. Values are taken from each run's saved `train_log.json`.

## Stage 1: Hindi SFT

| Setting | Value |
|---|---|
| Trainer | AlignTune `create_sft_trainer` (TRL backend); AlignTune 0.1.35 (Tiny Aya runs), 0.1.30 (Llama) |
| Base model | `CohereLabs/tiny-aya-global` / `CohereLabs/tiny-aya-fire` / `unsloth/Llama-3.2-3B-Instruct` |
| Data | 5,000 rows, stratified sample of Set A (every Hinglish and Aya row, rest Dolly-T / OASST-T); no safety data |
| Epochs / steps | 1 epoch = 157 optimiser steps |
| Batch | 4 per device × 8 gradient accumulation = 32 |
| Learning rate | 2e-4, cosine schedule, 3% warmup |
| Max sequence length | 768 tokens |
| LoRA | r = 16, α = 32, dropout 0.05, targets q/k/v/o/gate/up/down projections |
| Precision | bf16, gradient checkpointing |
| Checkpoints | every 100 steps (2 kept), loss logged every 10 steps |
| Seed | 42 |
| Time on one L4 | 55.3 min (Tiny Aya Global), 53.8 min (Tiny Aya Fire), 32.5 min (Llama) |

## Stage 2: Tameez GRPO

| Setting | Value |
|---|---|
| Trainer | AlignTune `create_rl_trainer(algorithm="grpo", loss_type="dr_grpo")`, TRL 1.7.1 backend, HF `generate` rollouts |
| Starting point | merged Stage 1 model, new LoRA |
| Prompt pool | Set B, 4,652 prompts (harmful 1,797 · angry user 1,198 · tricky 411 · benign 1,246) |
| Steps | 300 |
| Generations per prompt | 4 |
| Batch | 8 completions per device (2 prompts × 4) × 2 gradient accumulation = 4 prompts / 16 completions per step |
| Learning rate | 5e-5 |
| KL penalty β | 0.04 (reference = Stage 1 model) |
| Sampling during rollouts | temperature 0.9, top-p 0.95 |
| Lengths | prompt ≤ 512, completion ≤ 256, sequence ≤ 768 tokens |
| LoRA | r = 16, α = 32, dropout 0.0, same target modules |
| Rewards (weight) | no_badwords 2.0 · safe_reply 1.5 · low_toxicity 1.0 · refusal_calibration 1.0 · language_match 0.75 · format 0.5 |
| Reward models | Llama-Guard-3-1B (`safe_reply`), `textdetox/xlmr-large-toxicity-classifier-v2` (`low_toxicity`) |
| Precision | bf16, gradient checkpointing |
| Checkpoints | every 50 steps, all kept |
| Seed | 42 |
| Time on one L4 | 214.8 min (Tiny Aya Global, ≈ 44 s/step), 223.7 min (Tiny Aya Fire, ≈ 45 s/step), 136.8 min (Llama, ≈ 27 s/step) |

## Data building and audit

| Setting | Value |
|---|---|
| Hinglish rewrites | Aya Expanse 8B via vLLM, greedy (temperature 0), up to 1,024 new tokens, system rules + two worked examples |
| Rewrite acceptance checks | Roman Hinglish script; not a refusal; length 0.5–2× the source; no added greeting; no words copied from the prompt's examples |
| Harm labels | Llama-Guard-3-1B on the Hindi original; Hinglish rows inherit the label |
| Split | hash of the row id (stable across runs), seed 42 |
| Leakage dedup (build) | `datasketch` MinHash, 64 permutations, 5-character shingles, LSH threshold Jaccard ≥ 0.7 against every test prompt |
| CuratorKIT re-check | `ExactDeduplicator` + `MinHashDeduplicator` (Jaccard ≥ 0.7) within each training set and train-vs-test, `SecretsGate` on every split |

## Checkpoint selection

| Setting | Value |
|---|---|
| Test set | 200-prompt mini set (stratified subset of the held-out safety test) |
| Rule | over-refusal ≤ Stage 1 + 2 points; reply in the right script ≥ 95%; mean reply length ≥ 60% of Stage 1; then lowest unsafe + bad-word rate |
| Chosen | step 100 (Llama, passes every rule); step 300 (Tiny Aya Global and Tiny Aya Fire, fallback: no checkpoint passed the over-refusal cap) |

## Evaluation

| Setting | Value |
|---|---|
| Engine | vLLM, greedy decoding (temperature 0), up to 512 new tokens |
| System prompt | `You are a helpful assistant.` |
| Safety judge | `meta-llama/Llama-Guard-3-8B` on (prompt, reply), never used in training |
| Abuse judge | `Hate-speech-CNERG/indic-abusive-allInOne-MuRIL`, threshold 0.5 |
| Helpfulness judge | `CohereLabs/aya-expanse-8b`, Stage 2 vs Stage 1 on 400 prompts (200 Aya-Eval Hindi + 200 angry-user), judged twice with the order swapped; inconsistent verdicts count as ties; ties count half |
| Knowledge | MILU Hindi, all 14,831 questions, 0-shot, first-token log-probability over A–D |
| Intervals | 95% Wilson intervals on every rate |
| General benchmarks | lm-evaluation-harness 0.4.13, vLLM backend, chat template, few-shot as chat turns, greedy, ≤ 500 examples per (sub)task |

## Hardware and software

| | |
|---|---|
| GPU | one NVIDIA L4 (24 GB) on Google Colab for every training and evaluation step |
| Training stack | AlignTune 0.1.30 / 0.1.35, TRL 1.7.1, PEFT, transformers |
| Evaluation stack | vLLM, transformers (vLLM 0.33.0 for the Tiny Aya Fire scorecard); lm-evaluation-harness 0.4.13 + vLLM 0.33.0 for the general benchmarks |
