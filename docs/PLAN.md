# Tameez: GRPO Plan for a Polite, Safe Indic LLM

Nov 3, 2026 · @Vijay

> This is the original plan. Changes required after looking at the real data are listed in
> [DATA_ANALYSIS.md](DATA_ANALYSIS.md#required-edits-to-the-plan-planmd), and they were applied before the sets were built.

## Goal and success criteria

Train a 3B model with GRPO so that in Hindi and Hinglish it never uses abusive words, refuses harmful requests in the
user's own language and script, and still answers benign questions, including ones written by angry or swearing users.

The headline claim for judges: **safer in Hindi with no rise in over-refusal.** Most safety tuning trades one for the
other. Recent Indic benchmarks report that models over-refuse benign prompts in low-resource scripts while missing
unsafe ones (IndicSafe).

| Metric | Test set | Target vs Stage 1 (Hindi SFT) |
|---|---|---|
| Bad-word rate (% replies with a lexicon hit) | All eval sets | Under 0.5% |
| Unsafe reply rate (held-out judge) | Aya Red-teaming Hindi, IndicSafe Hindi | Halve it |
| Over-refusal rate on benign prompts | IndicSafe Hindi harmless + tricky, own tricky set | No increase; ideally lower |
| Script/language fidelity | All eval sets | Over 95% reply in the user's script |
| Helpfulness win rate vs Stage 1 | Aya Evaluation Suite Hindi, own angry-user set | At least 50% (no regression) |

Targets are proposals; set the final numbers after the Day 1 baseline run.

## Environment setup

One 24 GB GPU (L4 or A10) runs everything with LoRA. One A100 40/80 GB allows 8 generations per prompt and a bigger batch.

**Policy model:** `meta-llama/Llama-3.2-3B-Instruct` (Hindi officially supported; AlignTune's GRPO examples use it).
Optional second run: `CohereLabs/tiny-aya-global` (gated).

**Reward models (inference only, same GPU):**

| Role | Model | Why |
|---|---|---|
| Toxicity score | `textdetox/xlmr-large-toxicity-classifier-v2` | 15 languages incl. Hindi and Hinglish; idx 1 = toxic |
| Harmful-compliance check | `meta-llama/Llama-Guard-3-1B` (gated) | Supports Hindi |
| Eval-only judge | `meta-llama/Llama-Guard-3-8B` | Different, larger judge at eval time |
| Optional helpfulness judge | `CohereLabs/aya-expanse-8b` via vLLM | Strong Hindi; CuratorKIT supports it |

```bash
conda create -n tameez python=3.11 -y && conda activate tameez
git clone https://github.com/Lexsi-Labs/aligntune.git && pip install -e ./aligntune
pip install "curatorkit[all]" datasets accelerate peft bitsandbytes vllm lm-eval
hf auth login   # accept Llama 3.2, Llama Guard 3 and Tiny Aya licences first
aligntune info
```

AlignTune pins `trl[vllm]==1.7.1` and `transformers>=5.15,<6`; do not upgrade TRL separately.

## Datasets: four separate sets

| Set | Dataset | Used for | Notes |
|---|---|---|---|
| A. Hindi SFT | Aya Dataset, Hindi rows | Stage 1 fine-tune | Human-written, Apache 2.0 |
| A. Hindi SFT | IndicAlign-Instruct (Dolly-T, OpenAssistant-T) | Stage 1 | Translated + romanised versions |
| B. Alignment pool | IndicAlign-Toxic (HHRLHF-T + Toxic-Matrix) | Harmful prompts | 123K pairs, CC BY 4.0. Prompts only |
| B. Alignment pool | textdetox/multilingual_toxicity_dataset | Angry-user rants | toxic/non-toxic, 15 languages |
| B. Alignment pool | Own tricky-benign seeds + CuratorKIT paraphrases | Over-refusal training | 30–50 seeds × ~5 |
| B. Lexicons | AI4Bharat Setu NSFW + textdetox/multilingual_toxic_lexicon | Bad-word reward | Native-speaker whitelist |
| C. Safety test | Aya Red-teaming, Hindi | Unsafe-reply rate | global vs local harm |
| C. Safety test | IndicSafe, Hindi | Unsafe rate + over-refusal | 500/lang incl. 125 harmless + 45 tricky |
| C. Safety test | SurakshaEval, hindi config | Second safety benchmark | Gated |
| D. Skill test | MILU, Hindi test | Knowledge MCQ (no regression) | Gated; lm-eval-harness |
| D. Skill test | Aya Evaluation Suite, dolly-human-edited Hindi | Open-ended quality, judge win rate | 200 prompts |

**Leakage rules**
- Never use a set C or D prompt as a seed for generation.
- Run CuratorKIT MinHash dedup of set B against sets C and D before training.
- Split synthetic prompts train/test by seed.

## The RL environment: prompt pool (~5,000 prompts)

| Bucket | Share | Good behaviour | Source |
|---|---|---|---|
| harmful | 35% | Decline politely in the user's language, offer a safe alternative | IndicAlign-Toxic |
| abusive_legit | 25% | Ignore the abuse, solve the real problem, never copy bad words | Aya Hindi questions + rants; CuratorKIT evol |
| tricky_benign | 15% | Answer normally | Seeds + paraphrases |
| benign | 25% | Answer well in the same language and script | Aya Hindi, IndicAlign-Instruct (held out from A) |

Language mix target: ~50% Devanagari, 40% Roman Hinglish, 10% English.

**Row columns:** `prompt` (chat messages), `reference` (`refuse`/`help`), `bucket`, `lang` (`hi`/`hin`/`en`),
`allowed_terms`, `source`, `seed_id`. Save as JSON Lines with a `.json` extension (AlignTune's loader accepts only `.json`/`.csv`).

## Reward design (six functions, logged separately)

| Reward | Checks | Values | Weight |
|---|---|---|---|
| no_badwords | Lexicon hit or masked profanity after normalisation | 0 clean, −1 hit | 2.0 |
| low_toxicity | 1 − 2·p(toxic), textdetox XLM-R | −1…+1 | 1.0 |
| safe_reply | Llama Guard 3 1B on (prompt, reply) | +1 / −1 | 1.5 |
| refusal_calibration | Refusal phrase in first ~220 chars | harmful: +0.5 refused, else 0; benign: −1 refused, else +0.5 | 1.0 |
| language_match | Reply script vs `lang` | +1 / 0 / −0.5 / −1 | 0.75 |
| format | <20 chars, repetition, leaked tokens, >2,500 chars | 0 / −1 / −0.5 | 0.5 |
| helpfulness_judge (v2) | Aya Expanse 8B, 1–5 rubric | −1…+1, None for harmful | 1.0 |

Start with v1 (no LLM judge). Anti-hacks: `format` beats empty replies, `refusal_calibration` beats refuse-everything,
`language_match` beats switching to English, the masked-word rule beats "b***", and there is no quoting exemption.

## Training recipe: three checkpoints, one scorecard

- **Stage 0:** base Llama-3.2-3B-Instruct.
- **Stage 1:** Hindi SFT on set A (10–20K rows), LoRA r16/α32 on all attn+MLP, lr 2e-4, 1–2 epochs, max len 1,024, merge.
- **Stage 2a (optional):** CuratorKIT `preference`/`two_pass` with Aya Expanse 8B, ~2–3K pairs; SFT 1 epoch on chosen; keep pairs for DPO.
- **Stage 2b:** GRPO via AlignTune `create_rl_trainer`.

| Setting | Value |
|---|---|
| algorithm / backend | grpo / unsloth (fallback trl) |
| loss_type | dr_grpo |
| num_generations | 8 (4 on 24 GB) |
| batch × grad accum | 8 × 4 |
| learning_rate | 5e-6 |
| beta (KL) | 0.04 |
| temperature / top_p | 0.9 / 0.95 |
| max_completion_length | 384 |
| max_steps / save_steps | 400 / 50 |
| system_prompt | "You are a helpful assistant." |
| rewards | `[{"type": "custom", "weight": w, "params": {"function": fn}}, …]` |

## Evaluation scorecard (same on all stages, greedy, neutral system prompt)

| Metric | Test set | Measured with |
|---|---|---|
| Unsafe reply rate ↓ | Aya Red-teaming Hindi; IndicSafe harmful; SurakshaEval Hindi | Llama Guard 3 8B |
| Bad-word rate ↓ | Safety + angry-user test prompts | Lexicon |
| Toxicity ↓ | Same | Hate-speech-CNERG/indic-abusive-allInOne-MuRIL |
| Over-refusal ↓ | IndicSafe harmless + tricky; own tricky split | Refusal detector + human check of 100 |
| Script fidelity ↑ | All | Script + Hinglish detector |
| MILU Hindi accuracy ↑ | MILU Hindi test | lm-eval-harness, 0-shot |
| Helpfulness win rate ↑ | Aya Eval Suite Hindi; angry-user split | Aya Expanse 8B pairwise vs Stage 1 |

Grading rule: measure safety with different models from the ones in the reward.

**Ablations (150 steps each):** prompt-only "be polite" baseline; GRPO without refusal_calibration; without language_match; DPO instead of GRPO.
**Human check:** two native speakers rate 100 blind samples per stage (politeness, helpfulness, naturalness, 1–5); report agreement.

## Day-by-day

1. Setup + Stage 0 score.
2. Build data (A, B, seeds, lexicon whitelist, dedup).
3. Stage 1 SFT + score.
4. Reward sanity check on 50 hand-labelled replies; 50-step GRPO pilot.
5. Full GRPO (400 steps); score every 50-step checkpoint on a 200-prompt mini-set.
6. Stage 2 scorecard + ablations + human check.
7. Write-up, demo (Gradio).

## Risks

| Risk | Signal | Fix |
|---|---|---|
| Refuses everything | refusal_calibration falls | Raise its weight / benign share |
| Short generic replies | Length collapses | v2 judge; raise beta |
| Drifts to English | language_match falls | Weight → 1.0 |
| Lexicon flags innocent words | Benign replies scored −1 | Whitelist + allowed_terms |
| Weak classifier on Hinglish | Toxic Hinglish scored clean | Lexicon carries hard penalty |
| Hindi skill drops | MILU −1 pt | Raise beta; more benign; earlier checkpoint |
| Slow reward step | Llama Guard dominates | Sample 30% of benign rows; 4 generations |
| Gated models late | Download fails | Request access early |

## Sources

AlignTune · CuratorKIT · IndicLLMSuite / IndicAlign · Aya Red-teaming · IndicSafe · SurakshaEval · MILU ·
Qi et al., "Fine-tuning Aligned Language Models Compromises Safety" (ICLR 2024) · textdetox models and datasets
