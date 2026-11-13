# General-capability benchmarks: base vs Tameez

lm-evaluation-harness 0.4.13 with the vLLM backend, chat template applied, few-shot examples as chat turns, greedy decoding, at most 500 examples per task or subtask, identical settings for every model. Scores are percentages; ± is lm-eval's standard error. **Δ** = Tameez minus base, in points; ⚠️ marks a drop of more than 2 points.

## Tiny Aya Global

| Benchmark | Language | Measures | Base | Tameez | Δ | n |
|---|---|---|---|---|---|---|
| XNLI | English | language understanding (entailment) | 46.4 ± 2.2 | 50.8 ± 2.2 | +4.4 | 500 |
| XNLI | Hindi | language understanding (entailment) | 48.2 ± 2.2 | 49.6 ± 2.2 | +1.4 | 500 |
| Belebele | English | reading comprehension | 80.6 ± 1.8 | 77.4 ± 1.9 | -3.2 ⚠️ | 500 |
| Belebele | Hindi | reading comprehension | 55.6 ± 2.2 | 63.0 ± 2.2 | +7.4 | 500 |
| Global-MMLU Lite | English | knowledge (same questions as Hindi row) | 59.8 ± 2.5 | 57.2 ± 2.5 | -2.5 ⚠️ | 400 |
| Global-MMLU Lite | Hindi | knowledge (same questions as English row) | 50.2 ± 2.5 | 50.0 ± 2.5 | -0.2 | 400 |
| GSM8K | English | grade-school maths (5-shot) | 61.4 ± 2.2 | 52.4 ± 2.2 | -9.0 ⚠️ | 500 |
| GSM8K (strict format) | English | maths, answer in the exact '####' format | 61.2 ± 2.2 | 52.2 ± 2.2 | -9.0 ⚠️ | 500 |
| IFEval (prompt, strict) | English | instruction following | 57.4 ± 2.2 | 61.2 ± 2.2 | +3.8 | 500 |
| IFEval (instruction, strict) | English | instruction following | 70.1 | 71.8 | +1.7 | 500 |
| MILU (project scorecard) | Hindi | knowledge (14,831 questions) | 45.4 | 43.2 | -2.2 ⚠️ | 14831 |

- `tiny_aya_base`: `tiny_aya_base/CohereLabs__tiny-aya-global/results_2026-11-12T21-56-00.878190.json`
- `tiny_aya_tameez`: `tiny_aya_tameez/IINOVAII__Tiny-Aya-Global-Tameez-GRPO/results_2026-11-12T21-12-29.398181.json`

## Llama-3.2-3B-Instruct

| Benchmark | Language | Measures | Base | Tameez | Δ | n |
|---|---|---|---|---|---|---|
| XNLI | English | language understanding (entailment) | 46.0 ± 2.2 | 47.2 ± 2.2 | +1.2 | 500 |
| XNLI | Hindi | language understanding (entailment) | 44.8 ± 2.2 | 45.2 ± 2.2 | +0.4 | 500 |
| Belebele | English | reading comprehension | 82.8 ± 1.7 | 82.4 ± 1.7 | -0.4 | 500 |
| Belebele | Hindi | reading comprehension | 41.8 ± 2.2 | 53.6 ± 2.2 | +11.8 | 500 |
| Global-MMLU Lite | English | knowledge (same questions as Hindi row) | 58.2 ± 2.4 | 60.5 ± 2.4 | +2.2 | 400 |
| Global-MMLU Lite | Hindi | knowledge (same questions as English row) | 35.2 ± 2.4 | 43.2 ± 2.5 | +8.0 | 400 |
| GSM8K | English | grade-school maths (5-shot) | 78.0 ± 1.9 | 77.4 ± 1.9 | -0.6 | 500 |
| GSM8K (strict format) | English | maths, answer in the exact '####' format | 72.6 ± 2.0 | 74.0 ± 2.0 | +1.4 | 500 |
| IFEval (prompt, strict) | English | instruction following | 68.8 ± 2.1 | 65.4 ± 2.1 | -3.4 ⚠️ | 500 |
| IFEval (instruction, strict) | English | instruction following | 77.0 | 74.3 | -2.7 ⚠️ | 500 |
| MILU (project scorecard) | Hindi | knowledge (14,831 questions) | 38.5 | 39.5 | +1.0 | 14831 |

- `llama_base`: `llama_base/unsloth__Llama-3.2-3B-Instruct/results_2026-11-12T21-24-10.719483.json`
- `llama_tameez`: `llama_tameez/IINOVAII__Llama-3.2-3B-Tameez-GRPO/results_2026-11-12T21-35-27.230978.json`


## Summary: did general capability hold up?

**Mostly yes in Hindi, partly in English.** Comparing each base model with its final Tameez model (Hindi SFT + GRPO):

- **Tiny Aya Global.** Hindi reading comprehension improved (Belebele +7.4) and Hindi knowledge held (Global-MMLU Lite
  −0.2; MILU −2.2 on the full scorecard). Instruction following improved (IFEval +3.8). English slipped: reading
  comprehension −3.2, Global-MMLU Lite −2.5, and **maths dropped clearly (GSM8K 61.4 → 52.4, −9.0)**. The maths loss is
  not just formatting: the Tameez model gave an unreadable final answer slightly more often (34 vs 5 of 1,000 extractions),
  but most of the gap is wrong answers.
- **Llama-3.2-3B.** Hindi improved a lot (Belebele +11.8, Global-MMLU Lite +8.0), English held (Belebele −0.4,
  GSM8K −0.6, Global-MMLU Lite +2.2), and instruction following dropped (IFEval −3.4 prompt-level).

**Caveats.** These compare base with final, so the change includes the Hindi SFT stage as well as GRPO; the SFT models
were not benchmarked, so the two cannot be separated yet. At 400–500 examples per task the standard error is about
±2 points, so differences under ~4 points are within noise.
