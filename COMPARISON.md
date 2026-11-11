# Llama-3.2-3B vs Tiny Aya Global: Tameez pipeline, same data, rewards, settings and scorecard

## Every stage, both models

| Metric | Better | Llama-3.2-3B S0 | Llama-3.2-3B S1 | Llama-3.2-3B S2 | Tiny Aya Global S0 | Tiny Aya Global S1 | Tiny Aya Global S2 | Better final |
|---|---|---|---|---|---|---|---|---|
| Unsafe replies, all harmful prompts | ↓ | 10.0% | 15.6% | 3.7% | 4.5% | 20.7% | 3.9% | Llama-3.2-3B |
| • Hindi red-teaming | ↓ | 10.0% | 16.0% | 2.6% | 3.9% | 19.8% | 1.5% | Tiny Aya Global |
| • Hinglish red-teaming | ↓ | 21.6% | 30.9% | 15.8% | 18.7% | 57.6% | 29.5% | Llama-3.2-3B |
| • IndicSafe harmful | ↓ | 6.4% | 11.2% | 2.7% | 1.8% | 11.2% | 0.6% | Tiny Aya Global |
| • SurakshaEval | ↓ | 6.9% | 6.9% | 0.0% | 0.0% | 10.3% | 0.9% | Llama-3.2-3B |
| Bad-word replies | ↓ | 2.8% | 3.0% | 0.7% | 2.5% | 2.1% | 0.6% | Tiny Aya Global |
| Abusive replies (MuRIL) | ↓ | 3.6% | 6.9% | 7.5% | 3.0% | 8.4% | 4.3% | Tiny Aya Global |
| Explicit refusal of harmful prompts | ↑ | 31.9% | 28.3% | 58.5% | 39.7% | 33.1% | 73.6% | Tiny Aya Global |
| Over-refusal, all benign | ↓ | 2.1% | 3.3% | 7.0% | 2.5% | 0.8% | 3.9% | Tiny Aya Global |
| • IndicSafe harmless + tricky | ↓ | 0.0% | 0.0% | 1.2% | 1.2% | 0.0% | 0.0% | Tiny Aya Global |
| • own tricky test | ↓ | 0.7% | 0.7% | 2.1% | 0.0% | 0.0% | 1.4% | Tiny Aya Global |
| • angry users | ↓ | 5.0% | 8.0% | 15.5% | 5.5% | 2.0% | 9.0% | Tiny Aya Global |
| Reply in user's script | ↑ | 99.6% | 99.8% | 99.7% | 94.0% | 98.6% | 99.3% | Llama-3.2-3B |
| • Hinglish prompts | ↑ | 97.5% | 98.4% | 98.0% | 50.2% | 89.4% | 94.3% | Llama-3.2-3B |
| MILU Hindi accuracy | ↑ | 38.5% | 39.3% | 39.5% | 45.4% | 41.3% | 43.2% | Tiny Aya Global |
| Mean reply length (chars) |  | 828 | 646 | 376 | 1180 | 635 | 388 |  |

## What GRPO changed (Stage 1 → Stage 2), per model

| Metric | Llama-3.2-3B: S1 → S2 | Tiny Aya Global: S1 → S2 |
|---|---|---|
| Unsafe replies, all harmful prompts | 15.6% → 3.7% (-11.9) | 20.7% → 3.9% (-16.8) |
| • Hindi red-teaming | 16.0% → 2.6% (-13.4) | 19.8% → 1.5% (-18.3) |
| • Hinglish red-teaming | 30.9% → 15.8% (-15.1) | 57.6% → 29.5% (-28.1) |
| • IndicSafe harmful | 11.2% → 2.7% (-8.5) | 11.2% → 0.6% (-10.6) |
| • SurakshaEval | 6.9% → 0.0% (-6.9) | 10.3% → 0.9% (-9.5) |
| Bad-word replies | 3.0% → 0.7% (-2.3) | 2.1% → 0.6% (-1.5) |
| Abusive replies (MuRIL) | 6.9% → 7.5% (+0.7) | 8.4% → 4.3% (-4.1) |
| Explicit refusal of harmful prompts | 28.3% → 58.5% (+30.3) | 33.1% → 73.6% (+40.5) |
| Over-refusal, all benign | 3.3% → 7.0% (+3.7) | 0.8% → 3.9% (+3.1) |
| • IndicSafe harmless + tricky | 0.0% → 1.2% (+1.2) | 0.0% → 0.0% (+0.0) |
| • own tricky test | 0.7% → 2.1% (+1.4) | 0.0% → 1.4% (+1.4) |
| • angry users | 8.0% → 15.5% (+7.5) | 2.0% → 9.0% (+7.0) |
| Reply in user's script | 99.8% → 99.7% (-0.1) | 98.6% → 99.3% (+0.6) |
| • Hinglish prompts | 98.4% → 98.0% (-0.4) | 89.4% → 94.3% (+4.9) |
| MILU Hindi accuracy | 39.3% → 39.5% (+0.2) | 41.3% → 43.2% (+1.9) |
| Mean reply length (chars) | 646 → 376 | 635 → 388 |

## Safety drift from plain Hindi SFT (Stage 0 → Stage 1)

| Metric | Llama-3.2-3B | Tiny Aya Global |
|---|---|---|
| Unsafe replies, all harmful prompts | 10.0% → 15.6% | 4.5% → 20.7% |
| • Hindi red-teaming | 10.0% → 16.0% | 3.9% → 19.8% |
| • Hinglish red-teaming | 21.6% → 30.9% | 18.7% → 57.6% |
| Abusive replies (MuRIL) | 3.6% → 6.9% | 3.0% → 8.4% |
| Explicit refusal of harmful prompts | 31.9% → 28.3% | 39.7% → 33.1% |
| MILU Hindi accuracy | 38.5% → 39.3% | 45.4% → 41.3% |

## Llama-3.2-3B: GRPO checkpoint selection (mini test set)

| Checkpoint | Unsafe | Bad words | Over-refusal | Script | Mean reply |
|---|---|---|---|---|---|
| Stage 1 (reference) | 13.8% | 2.5% | 4.2% | 100.0% | 793.1 |
| step 50 | 6.2% | 2.0% | 7.5% | 99.5% | 591.8 |
| step 100 ✓ chosen | 5.0% | 2.0% | 5.8% | 100.0% | 602.5 |
| step 150 | 1.2% | 1.0% | 6.7% | 100.0% | 522.0 |
| step 200 | 1.2% | 1.0% | 11.7% | 100.0% | 401.2 |
| step 250 | 1.2% | 1.0% | 8.3% | 99.5% | 412.2 |
| step 300 | 1.2% | 1.0% | 6.7% | 100.0% | 401.6 |

Passes constraints: True

## Tiny Aya Global: GRPO checkpoint selection (mini test set)

| Checkpoint | Unsafe | Bad words | Over-refusal | Script | Mean reply |
|---|---|---|---|---|---|
| Stage 1 (reference) | 27.5% | 2.0% | 0.8% | 96.5% | 738.1 |
| step 50 | 18.8% | 1.5% | 3.3% | 96.5% | 677.5 |
| step 100 | 7.5% | 1.0% | 7.5% | 97.5% | 593.3 |
| step 150 | 8.8% | 1.5% | 5.0% | 97.5% | 547.6 |
| step 200 | 11.2% | 2.0% | 5.0% | 97.0% | 515.0 |
| step 250 | 8.8% | 0.5% | 5.0% | 97.5% | 454.5 |
| step 300 ✓ chosen | 5.0% | 1.0% | 5.8% | 98.0% | 414.6 |

Passes constraints: False

## Helpfulness judge (Aya Expanse 8B, Stage 2 vs its own Stage 1, both orders)

| Model | Win rate | 95% CI | Wins / ties / losses | Aya-Eval | Angry users |
|---|---|---|---|---|---|
| Llama-3.2-3B | 42.1% | 37.3–47.0 | 48 / 241 / 111 | 44.5% | 39.8% |
| Tiny Aya Global | 61.5% | 56.7–66.3 | 139 / 214 / 47 | 62.5% | 60.5% |

## Notes

- **Tiny Aya "Passes constraints: False".** The selection rule allows over-refusal of at most Stage 1 + 2 points. Tiny Aya's
  Stage 1 over-refusal on the mini set was very low (0.8%), so the cap was 2.8%, and every GRPO checkpoint was above it
  (3.3–7.5%). The script then fell back to the lowest unsafe + bad-word rate, which is step 300. On the full test set
  that model's over-refusal is 3.9%, still lower than Llama's final 7.0%.
- Llama numbers come from `stages/`, Tiny Aya numbers from `tiny_aya_run/stages/` (Colab L4, Nov 11 2026,
  same code, data, rewards and settings; only `TAMEEZ_POLICY=CohereLabs/tiny-aya-global` changed).
- Regenerate: `python -I scripts/compare_runs.py --a . --a-name "Llama-3.2-3B" --b tiny_aya_run --b-name "Tiny Aya Global" --out COMPARISON.md`
