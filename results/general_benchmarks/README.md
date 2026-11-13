# General-capability benchmarks: base → Hindi SFT → Tameez

lm-evaluation-harness 0.4.13 with the vLLM backend, chat template applied, few-shot examples as chat turns, greedy decoding, at most 500 examples per task or subtask, identical settings for every model. Scores are percentages; ± is lm-eval's standard error. **SFT effect** = Hindi SFT minus base, **GRPO effect** = Tameez minus Hindi SFT, **Total** = Tameez minus base, all in points; ⚠️ marks a drop of more than 2 points.

## Tiny Aya Global

| Benchmark | Language | Measures | Base | Hindi SFT | Tameez | SFT effect | GRPO effect | Total | n |
|---|---|---|---|---|---|---|---|---|---|
| XNLI | English | language understanding (entailment) | 46.4 ± 2.2 | 49.4 ± 2.2 | 50.8 ± 2.2 | +3.0 | +1.4 | +4.4 | 500 |
| XNLI | Hindi | language understanding (entailment) | 48.2 ± 2.2 | 49.0 ± 2.2 | 49.6 ± 2.2 | +0.8 | +0.6 | +1.4 | 500 |
| Belebele | English | reading comprehension | 80.6 ± 1.8 | 77.8 ± 1.9 | 77.4 ± 1.9 | -2.8 ⚠️ | -0.4 | -3.2 ⚠️ | 500 |
| Belebele | Hindi | reading comprehension | 55.6 ± 2.2 | 63.2 ± 2.2 | 63.0 ± 2.2 | +7.6 | -0.2 | +7.4 | 500 |
| Global-MMLU Lite | English | knowledge (same questions as Hindi row) | 59.8 ± 2.5 | 58.0 ± 2.5 | 57.2 ± 2.5 | -1.8 | -0.7 | -2.5 ⚠️ | 400 |
| Global-MMLU Lite | Hindi | knowledge (same questions as English row) | 50.2 ± 2.5 | 49.5 ± 2.5 | 50.0 ± 2.5 | -0.7 | +0.5 | -0.2 | 400 |
| GSM8K | English | grade-school maths (5-shot) | 61.4 ± 2.2 | 52.4 ± 2.2 | 52.4 ± 2.2 | -9.0 ⚠️ | +0.0 | -9.0 ⚠️ | 500 |
| GSM8K (strict format) | English | maths, answer in the exact '####' format | 61.2 ± 2.2 | 52.2 ± 2.2 | 52.2 ± 2.2 | -9.0 ⚠️ | +0.0 | -9.0 ⚠️ | 500 |
| IFEval (prompt, strict) | English | instruction following | 57.4 ± 2.2 | 61.2 ± 2.2 | 61.2 ± 2.2 | +3.8 | +0.0 | +3.8 | 500 |
| IFEval (instruction, strict) | English | instruction following | 70.1 | 71.7 | 71.8 | +1.6 | +0.1 | +1.7 | 500 |
| MILU (project scorecard) | Hindi | knowledge (14,831 questions) | 45.4 | 41.3 | 43.2 | -4.1 ⚠️ | +1.9 | -2.2 ⚠️ | 14831 |

- `tiny_aya_base`: `tiny_aya_base/CohereLabs__tiny-aya-global/results_2026-11-12T21-56-00.878190.json`
- `tiny_aya_sft`: `tiny_aya_sft/IINOVAII__Tiny-Aya-Global-Hindi-SFT/results_2026-11-13T01-54-04.231493.json`
- `tiny_aya_tameez`: `tiny_aya_tameez/IINOVAII__Tiny-Aya-Global-Tameez-GRPO/results_2026-11-12T21-12-29.398181.json`

## Llama-3.2-3B-Instruct

| Benchmark | Language | Measures | Base | Hindi SFT | Tameez | SFT effect | GRPO effect | Total | n |
|---|---|---|---|---|---|---|---|---|---|
| XNLI | English | language understanding (entailment) | 46.0 ± 2.2 | 48.8 ± 2.2 | 47.2 ± 2.2 | +2.8 | -1.6 | +1.2 | 500 |
| XNLI | Hindi | language understanding (entailment) | 44.8 ± 2.2 | 45.6 ± 2.2 | 45.2 ± 2.2 | +0.8 | -0.4 | +0.4 | 500 |
| Belebele | English | reading comprehension | 82.8 ± 1.7 | 82.6 ± 1.7 | 82.4 ± 1.7 | -0.2 | -0.2 | -0.4 | 500 |
| Belebele | Hindi | reading comprehension | 41.8 ± 2.2 | 54.4 ± 2.2 | 53.6 ± 2.2 | +12.6 | -0.8 | +11.8 | 500 |
| Global-MMLU Lite | English | knowledge (same questions as Hindi row) | 58.2 ± 2.4 | 61.0 ± 2.4 | 60.5 ± 2.4 | +2.8 | -0.5 | +2.2 | 400 |
| Global-MMLU Lite | Hindi | knowledge (same questions as English row) | 35.2 ± 2.4 | 42.0 ± 2.5 | 43.2 ± 2.5 | +6.8 | +1.2 | +8.0 | 400 |
| GSM8K | English | grade-school maths (5-shot) | 78.0 ± 1.9 | 76.4 ± 1.9 | 77.4 ± 1.9 | -1.6 | +1.0 | -0.6 | 500 |
| GSM8K (strict format) | English | maths, answer in the exact '####' format | 72.6 ± 2.0 | 74.6 ± 1.9 | 74.0 ± 2.0 | +2.0 | -0.6 | +1.4 | 500 |
| IFEval (prompt, strict) | English | instruction following | 68.8 ± 2.1 | 65.4 ± 2.1 | 65.4 ± 2.1 | -3.4 ⚠️ | +0.0 | -3.4 ⚠️ | 500 |
| IFEval (instruction, strict) | English | instruction following | 77.0 | 74.3 | 74.3 | -2.7 ⚠️ | +0.0 | -2.7 ⚠️ | 500 |
| MILU (project scorecard) | Hindi | knowledge (14,831 questions) | 38.5 | 39.3 | 39.5 | +0.8 | +0.2 | +1.0 | 14831 |

- `llama_base`: `llama_base/unsloth__Llama-3.2-3B-Instruct/results_2026-11-12T21-24-10.719483.json`
- `llama_sft`: `llama_sft/IINOVAII__Llama-3.2-3B-Hindi-SFT/results_2026-11-13T02-06-47.966535.json`
- `llama_tameez`: `llama_tameez/IINOVAII__Llama-3.2-3B-Tameez-GRPO/results_2026-11-12T21-35-27.230978.json`

## Tiny Aya Global: Hinglish follow-up run

150 more GRPO steps on the 1,357 Hinglish prompts of Set B, starting from Tiny Aya Tameez, with Llama-Guard-3-8B (4-bit) as the safety reward; same benchmark settings. **Δ** = follow-up minus Tameez.

| Benchmark | Language | Measures | Tameez | Follow-up | Δ | n |
|---|---|---|---|---|---|---|
| XNLI | English | language understanding (entailment) | 50.8 ± 2.2 | 51.4 ± 2.2 | +0.6 | 500 |
| XNLI | Hindi | language understanding (entailment) | 49.6 ± 2.2 | 50.0 ± 2.2 | +0.4 | 500 |
| Belebele | English | reading comprehension | 77.4 ± 1.9 | 77.6 ± 1.9 | +0.2 | 500 |
| Belebele | Hindi | reading comprehension | 63.0 ± 2.2 | 63.2 ± 2.2 | +0.2 | 500 |
| Global-MMLU Lite | English | knowledge (same questions as Hindi row) | 57.2 ± 2.5 | 58.0 ± 2.5 | +0.7 | 400 |
| Global-MMLU Lite | Hindi | knowledge (same questions as English row) | 50.0 ± 2.5 | 49.5 ± 2.5 | -0.5 | 400 |
| GSM8K | English | grade-school maths (5-shot) | 52.4 ± 2.2 | 51.2 ± 2.2 | -1.2 | 500 |
| GSM8K (strict format) | English | maths, answer in the exact '####' format | 52.2 ± 2.2 | 50.8 ± 2.2 | -1.4 | 500 |
| IFEval (prompt, strict) | English | instruction following | 61.2 ± 2.2 | 61.8 ± 2.2 | +0.6 | 500 |
| IFEval (instruction, strict) | English | instruction following | 71.8 | 72.1 | +0.3 | 500 |

- `tiny_aya_hinglish`: `tiny_aya_hinglish/__content__tameez_aya__stages__stage3_hinglish__model__merged/results_2026-11-13T07-11-14.943441.json`

## Summary: what caused the losses?

**The Hindi SFT stage, not the GRPO safety stage.** Splitting each model into base → Hindi SFT → Tameez:

- **GRPO did not measurably cost general capability.** On both models, every GRPO effect is within ±2 points
  (largest: Tiny Aya MILU +1.9, Llama XNLI English −1.6), which is inside the noise of a 400–500 question sample.
- **Tiny Aya Global: the losses are SFT.** English maths fell during SFT (GSM8K 61.4 → 52.4, −9.0; SFT lost 85
  questions the base model solved and gained 40) and stayed there after GRPO. English reading (Belebele −2.8 of −3.2)
  and MILU Hindi knowledge (−4.1) also dropped at SFT; GRPO won back half of the MILU loss (+1.9). The gains came from
  SFT too: Hindi reading +7.6, IFEval +3.8, XNLI English +3.0.
- **Llama-3.2-3B: gains and the one loss are SFT.** SFT brought the large Hindi gains (Belebele +12.6, Global-MMLU Lite
  +6.8) and the instruction-following drop (IFEval −3.4 prompt-level, −2.7 instruction-level). GSM8K and English
  reading stayed flat through both stages.

**Identical scores are not identical models.** SFT and Tameez tie exactly on three rows (Tiny Aya GSM8K and IFEval,
Llama IFEval). The weights differ, each run's `config.model_args` names the right repository, scores recompute from the
saved samples, and the replies differ (Tiny Aya: 259 of 500 GSM8K replies, 447 of 500 IFEval replies). The per-question
changes cancel out: on Tiny Aya GSM8K, GRPO fixed 21 questions and broke 21.

**What this suggests.** Stage 1 was trained on 5,000 rows of general Hindi and Hinglish instruction data (translated
Dolly and OASST, Aya Dataset), with no English rows and no maths-focused data, so the English maths loss is most likely
forgetting during SFT. Mixing some English and maths data into
Stage 1 is the obvious next thing to try; GRPO itself can stay as it is.

**Caveats.** At 400–500 examples per task the standard error of each score is about ±2 points, so a stage difference
under ~4 points is not significant; GSM8K −9.0 is well outside that, and MILU uses all 14,831 questions (standard error
about ±0.4). The SFT models were run on 13 Nov 2026 in a fresh Colab L4 session with the same pinned stack (vLLM 0.33.0,
transformers 5.20.2, torch 2.14.0, lm-eval 0.4.13) and the same seeds as the base and Tameez runs.
