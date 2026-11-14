# Llama-3.2-3B, Tiny Aya Global and Tiny Aya Fire: Tameez pipeline, same data, rewards, settings and scorecard

## Every stage, every model

| Metric | Better | Llama-3.2-3B S0 | Llama-3.2-3B S1 | Llama-3.2-3B S2 | Tiny Aya Global S0 | Tiny Aya Global S1 | Tiny Aya Global S2 | Tiny Aya Fire S0 | Tiny Aya Fire S1 | Tiny Aya Fire S2 | Best final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Unsafe replies, all harmful prompts | ↓ | 10.0% | 15.6% | 3.7% | 4.5% | 20.7% | 3.9% | 5.1% | 19.9% | 4.3% | Llama-3.2-3B |
| • Hindi red-teaming | ↓ | 10.0% | 16.0% | 2.6% | 3.9% | 19.8% | 1.5% | 3.5% | 18.3% | 1.8% | Tiny Aya Global |
| • Hinglish red-teaming | ↓ | 21.6% | 30.9% | 15.8% | 18.7% | 57.6% | 29.5% | 25.9% | 56.1% | 31.7% | Llama-3.2-3B |
| • IndicSafe harmful | ↓ | 6.4% | 11.2% | 2.7% | 1.8% | 11.2% | 0.6% | 2.4% | 13.6% | 0.6% | tie |
| • SurakshaEval | ↓ | 6.9% | 6.9% | 0.0% | 0.0% | 10.3% | 0.9% | 0.9% | 6.9% | 1.7% | Llama-3.2-3B |
| Bad-word replies | ↓ | 2.8% | 3.0% | 0.7% | 2.5% | 2.1% | 0.6% | 2.0% | 2.0% | 0.8% | Tiny Aya Global |
| Abusive replies (MuRIL) | ↓ | 3.6% | 6.9% | 7.5% | 3.0% | 8.4% | 4.3% | 2.2% | 8.2% | 3.1% | Tiny Aya Fire |
| Explicit refusal of harmful prompts | ↑ | 31.9% | 28.3% | 58.5% | 39.7% | 33.1% | 73.6% | 47.0% | 31.8% | 73.1% | Tiny Aya Global |
| Over-refusal, all benign | ↓ | 2.1% | 3.3% | 7.0% | 2.5% | 0.8% | 3.9% | 3.1% | 0.4% | 3.5% | Tiny Aya Fire |
| • IndicSafe harmless + tricky | ↓ | 0.0% | 0.0% | 1.2% | 1.2% | 0.0% | 0.0% | 0.6% | 0.0% | 0.0% | tie |
| • own tricky test | ↓ | 0.7% | 0.7% | 2.1% | 0.0% | 0.0% | 1.4% | 0.0% | 0.0% | 1.4% | tie |
| • angry users | ↓ | 5.0% | 8.0% | 15.5% | 5.5% | 2.0% | 9.0% | 7.5% | 1.0% | 8.0% | Tiny Aya Fire |
| Reply in user's script | ↑ | 99.6% | 99.8% | 99.7% | 93.9% | 98.6% | 99.3% | 94.7% | 98.2% | 99.3% | Llama-3.2-3B |
| • Hinglish prompts | ↑ | 97.6% | 98.4% | 98.0% | 50.2% | 89.4% | 94.3% | 55.1% | 85.7% | 95.1% | Llama-3.2-3B |
| MILU Hindi accuracy | ↑ | 38.5% | 39.3% | 39.5% | 45.4% | 41.3% | 43.2% | 46.9% | 43.6% | 45.0% | Tiny Aya Fire |
| Mean reply length (chars) |  | 828 | 646 | 376 | 1180 | 635 | 388 | 1133 | 656 | 526 |  |

## What GRPO changed (Stage 1 → Stage 2), per model

| Metric | Llama-3.2-3B: S1 → S2 | Tiny Aya Global: S1 → S2 | Tiny Aya Fire: S1 → S2 |
|---|---|---|---|
| Unsafe replies, all harmful prompts | 15.6% → 3.7% (-11.9) | 20.7% → 3.9% (-16.8) | 19.9% → 4.3% (-15.6) |
| • Hindi red-teaming | 16.0% → 2.6% (-13.4) | 19.8% → 1.5% (-18.3) | 18.3% → 1.8% (-16.5) |
| • Hinglish red-teaming | 30.9% → 15.8% (-15.1) | 57.6% → 29.5% (-28.1) | 56.1% → 31.7% (-24.5) |
| • IndicSafe harmful | 11.2% → 2.7% (-8.5) | 11.2% → 0.6% (-10.6) | 13.6% → 0.6% (-13.0) |
| • SurakshaEval | 6.9% → 0.0% (-6.9) | 10.3% → 0.9% (-9.5) | 6.9% → 1.7% (-5.2) |
| Bad-word replies | 3.0% → 0.7% (-2.3) | 2.1% → 0.6% (-1.5) | 2.0% → 0.8% (-1.2) |
| Abusive replies (MuRIL) | 6.9% → 7.5% (+0.7) | 8.4% → 4.3% (-4.1) | 8.2% → 3.1% (-5.1) |
| Explicit refusal of harmful prompts | 28.3% → 58.5% (+30.2) | 33.1% → 73.6% (+40.4) | 31.8% → 73.1% (+41.2) |
| Over-refusal, all benign | 3.3% → 7.0% (+3.7) | 0.8% → 3.9% (+3.1) | 0.4% → 3.5% (+3.1) |
| • IndicSafe harmless + tricky | 0.0% → 1.2% (+1.2) | 0.0% → 0.0% (+0.0) | 0.0% → 0.0% (+0.0) |
| • own tricky test | 0.7% → 2.1% (+1.4) | 0.0% → 1.4% (+1.4) | 0.0% → 1.4% (+1.4) |
| • angry users | 8.0% → 15.5% (+7.5) | 2.0% → 9.0% (+7.0) | 1.0% → 8.0% (+7.0) |
| Reply in user's script | 99.8% → 99.7% (-0.1) | 98.6% → 99.3% (+0.6) | 98.2% → 99.3% (+1.1) |
| • Hinglish prompts | 98.4% → 98.0% (-0.4) | 89.4% → 94.3% (+4.9) | 85.7% → 95.1% (+9.4) |
| MILU Hindi accuracy | 39.3% → 39.5% (+0.2) | 41.3% → 43.2% (+1.9) | 43.6% → 45.0% (+1.4) |
| Mean reply length (chars) | 646 → 376 | 635 → 388 | 656 → 526 |

## Safety drift from plain Hindi SFT (Stage 0 → Stage 1)

| Metric | Llama-3.2-3B | Tiny Aya Global | Tiny Aya Fire |
|---|---|---|---|
| Unsafe replies, all harmful prompts | 10.0% → 15.6% | 4.5% → 20.7% | 5.1% → 19.9% |
| • Hindi red-teaming | 10.0% → 16.0% | 3.9% → 19.8% | 3.5% → 18.3% |
| • Hinglish red-teaming | 21.6% → 30.9% | 18.7% → 57.6% | 25.9% → 56.1% |
| Abusive replies (MuRIL) | 3.6% → 6.9% | 3.0% → 8.4% | 2.2% → 8.2% |
| Explicit refusal of harmful prompts | 31.9% → 28.3% | 39.7% → 33.1% | 47.0% → 31.8% |
| MILU Hindi accuracy | 38.5% → 39.3% | 45.4% → 41.3% | 46.9% → 43.6% |

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

## Tiny Aya Fire: GRPO checkpoint selection (mini test set)

| Checkpoint | Unsafe | Bad words | Over-refusal | Script | Mean reply |
|---|---|---|---|---|---|
| Stage 1 (reference) | 25.0% | 2.0% | 0.0% | 96.5% | 754.0 |
| step 50 | 15.0% | 1.0% | 4.2% | 99.0% | 853.1 |
| step 100 | 13.8% | 1.5% | 5.8% | 98.0% | 789.2 |
| step 150 | 8.8% | 1.5% | 6.7% | 99.0% | 710.4 |
| step 200 | 8.8% | 1.0% | 5.0% | 98.0% | 689.5 |
| step 250 | 10.0% | 1.5% | 5.8% | 98.5% | 656.1 |
| step 300 ✓ chosen | 7.5% | 2.0% | 5.0% | 98.5% | 637.1 |

Passes constraints: False

## Helpfulness judge (Aya Expanse 8B, Stage 2 vs its own Stage 1, both orders)

| Model | Win rate | 95% CI | Wins / ties / losses | Aya-Eval | Angry users |
|---|---|---|---|---|---|
| Llama-3.2-3B | 42.1% | 37.3–47.0 | 48 / 241 / 111 | 44.5% | 39.8% |
| Tiny Aya Global | 61.5% | 56.7–66.3 | 139 / 214 / 47 | 62.5% | 60.5% |
| Tiny Aya Fire | 62.9% | 58.1–67.6 | 146 / 211 / 43 | 63.5% | 62.3% |

## Notes

- **Both Tiny Aya models: "Passes constraints: False".** The selection rule allows over-refusal of at most Stage 1 + 2
  points on the mini set. Both Tiny Aya Stage 1 models almost never refused a harmless mini-set prompt (Global 0.8%,
  Fire 0.0%), so the caps were 2.8% and 2.0%, and every GRPO checkpoint of both runs was above them (3.3–7.5% and
  4.2–6.7%). The script then fell back to the lowest unsafe + bad-word rate, which is step 300 in both runs. On the full
  test set their over-refusal is 3.9% (Global) and 3.5% (Fire), lower than Llama's 7.0%. Fire's step 300 met the other
  two rules (script 98.5%, replies 84% of Stage 1's length); Global's was also just under the length rule (56%).
- Llama numbers come from `stages/`, Tiny Aya Global from `tiny_aya_run/stages/` (Colab L4, 11 Nov 2026) and Tiny Aya
  Fire from `tiny_aya_fire_run/stages/` (Colab L4, 13 Nov 2026). Same code, data, rewards, settings and scorecard; only
  the policy changed (`TAMEEZ_POLICY=CohereLabs/tiny-aya-global` or `CohereLabs/tiny-aya-fire`). The Fire run was scored
  with the newer evaluation stack of the later runs (vLLM 0.33.0); re-running the published Global model on that stack
  moved its Hinglish unsafe rate from 29.5% to 28.1%, within the margin.
- Rates are recomputed exactly as count / n (metrics.json stores them to 4 decimals) and MILU from its per-domain counts.
- Regenerate: `python -I scripts/compare_runs.py --run . "Llama-3.2-3B" --run tiny_aya_run "Tiny Aya Global" --run tiny_aya_fire_run "Tiny Aya Fire" --out COMPARISON.md`
