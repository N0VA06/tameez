## 1. The same 139 requests in two scripts (scorecard replies)

| Model | Stage | Unsafe, Devanagari | Unsafe, Roman | Refused, Devanagari | Refused, Roman |
|---|---|---|---|---|---|
| Llama-3.2-3B | S0 base | 7.2% | 21.6% | 24.5% | 5.8% |
| Llama-3.2-3B | S1 Hindi SFT | 12.9% | 30.9% | 21.6% | 6.5% |
| Llama-3.2-3B | S2 Tameez GRPO | 2.9% | 15.8% | 56.1% | 8.6% |
| Tiny Aya Global | S0 base | 2.2% | 18.7% | 30.2% | 5.0% |
| Tiny Aya Global | S1 Hindi SFT | 20.9% | 57.6% | 23.0% | 0.0% |
| Tiny Aya Global | S2 Tameez GRPO | 1.4% | 29.5% | 75.5% | 10.1% |
| Tiny Aya Fire | S0 base | 2.2% | 25.9% | 37.4% | 5.0% |
| Tiny Aya Fire | S1 Hindi SFT | 19.4% | 56.1% | 23.0% | 0.7% |
| Tiny Aya Fire | S2 Tameez GRPO | 1.4% | 31.7% | 73.4% | 5.8% |

## 2. Where the same requests land on the model's harm direction

Position on the Devanagari harm direction (mean harmful − mean harmless Devanagari training prompts) at the steering layer: 0 = the average harmless Devanagari training prompt, 1 = the average harmful one. Test prompts: the 139 held-out twins and the 516 held-out harmless scorecard prompts; English harmful prompts come from the held-out part of the training pool (no English test twins exist). *Share* = (Roman harmful − Roman harmless) / (Devanagari harmful − Devanagari harmless).

| Model | Stage | Layer | Harmful, Devanagari | Harmful, Roman (same requests) | Harmless, Roman | Harmless, Devanagari | Harmful, English (pool) | Share | cos(Deva dir, Roman dir) |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | S0 base | 13/28 | 0.83 | 0.44 | 0.20 | 0.13 | 1.38 | 0.34 | 0.85 |
| Llama-3.2-3B | S1 Hindi SFT | 13/28 | 0.82 | 0.32 | 0.20 | 0.14 | 1.46 | 0.18 | 0.79 |
| Llama-3.2-3B | S2 Tameez GRPO | 13/28 | 0.82 | 0.33 | 0.16 | 0.12 | 1.40 | 0.25 | 0.82 |
| Tiny Aya Global | S0 base | 17/36 | 0.78 | 0.53 | 0.34 | 0.16 | 1.25 | 0.31 | 0.88 |
| Tiny Aya Global | S1 Hindi SFT | 17/36 | 0.69 | 0.49 | 0.30 | 0.10 | 1.11 | 0.34 | 0.91 |
| Tiny Aya Global | S2 Tameez GRPO | 18/36 | 0.72 | 0.51 | 0.18 | 0.07 | 1.17 | 0.51 | 0.93 |
| Tiny Aya Fire | S0 base | 17/36 | 0.78 | 0.51 | 0.33 | 0.16 | 1.25 | 0.29 | 0.86 |
| Tiny Aya Fire | S1 Hindi SFT | 17/36 | 0.70 | 0.52 | 0.32 | 0.12 | 1.14 | 0.36 | 0.91 |
| Tiny Aya Fire | S2 Tameez GRPO | 18/36 | 0.74 | 0.50 | 0.18 | 0.07 | 1.18 | 0.47 | 0.93 |

## 3. Can the model tell harmful from harmless in each script?

AUROC of linear probes (logistic regression on the last-token hidden state), best layer. Trained on the training pool (529 Devanagari/Roman pairs, 288 English prompts), tested on its held-out quarter (157 pairs, 112 English) or on the scorecard (139 Roman twins vs 106 harmless Roman prompts).

| Model | Stage | Deva → Deva | Deva → Roman | Roman → Roman | Deva → English | Deva → Roman test | Roman → Roman test | Deva+Roman → Roman test |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | S0 base | 0.999 | 0.975 | 0.986 | 1.000 | 0.748 | 0.856 | 0.865 |
| Llama-3.2-3B | S1 Hindi SFT | 0.999 | 0.973 | 0.983 | 1.000 | 0.710 | 0.817 | 0.819 |
| Llama-3.2-3B | S2 Tameez GRPO | 0.999 | 0.979 | 0.983 | 1.000 | 0.755 | 0.829 | 0.832 |
| Tiny Aya Global | S0 base | 1.000 | 0.983 | 0.990 | 1.000 | 0.826 | 0.926 | 0.915 |
| Tiny Aya Global | S1 Hindi SFT | 1.000 | 0.995 | 0.996 | 1.000 | 0.892 | 0.910 | 0.912 |
| Tiny Aya Global | S2 Tameez GRPO | 1.000 | 0.990 | 0.995 | 1.000 | 0.874 | 0.926 | 0.921 |
| Tiny Aya Fire | S0 base | 1.000 | 0.982 | 0.989 | 1.000 | 0.835 | 0.937 | 0.925 |
| Tiny Aya Fire | S1 Hindi SFT | 1.000 | 0.996 | 0.997 | 1.000 | 0.924 | 0.925 | 0.926 |
| Tiny Aya Fire | S2 Tameez GRPO | 1.000 | 0.993 | 0.998 | 1.000 | 0.897 | 0.922 | 0.921 |

### Same meaning across scripts, and tokens per word

Share of prompts whose nearest neighbour in the other script is their own twin (raw text, hidden states averaged over tokens, each script centred), at the steering layer. Chance: 1 in 686, 139 or 40.

| Model | Stage | Deva ↔ Roman (686 pool pairs) | Deva ↔ Roman (139 test pairs) | Deva ↔ English (40 seeds) | Roman ↔ English (40 seeds) | Tokens per word: Deva / Roman / English |
|---|---|---|---|---|---|---|
| Llama-3.2-3B | S0 base | 98.5% | 98.6% | 100.0% | 100.0% | 2.56 / 2.03 / 1.28 |
| Llama-3.2-3B | S1 Hindi SFT | 98.7% | 97.1% | 100.0% | 100.0% | 2.56 / 2.03 / 1.28 |
| Llama-3.2-3B | S2 Tameez GRPO | 98.7% | 97.1% | 100.0% | 100.0% | 2.56 / 2.03 / 1.28 |
| Tiny Aya Global | S0 base | 94.6% | 92.8% | 100.0% | 97.5% | 1.35 / 1.90 / 1.21 |
| Tiny Aya Global | S1 Hindi SFT | 97.2% | 97.1% | 100.0% | 100.0% | 1.35 / 1.90 / 1.21 |
| Tiny Aya Global | S2 Tameez GRPO | 97.4% | 97.1% | 100.0% | 100.0% | 1.35 / 1.90 / 1.21 |
| Tiny Aya Fire | S0 base | 95.5% | 94.2% | 100.0% | 100.0% | 1.35 / 1.90 / 1.21 |
| Tiny Aya Fire | S1 Hindi SFT | 97.8% | 96.4% | 100.0% | 100.0% | 1.35 / 1.90 / 1.21 |
| Tiny Aya Fire | S2 Tameez GRPO | 97.8% | 96.4% | 100.0% | 100.0% | 1.35 / 1.90 / 1.21 |

## 4. Does the harm direction predict refusal?

AUROC of the position on the Devanagari harm direction (steering layer) for telling the twins the model refused from those it did not, in the scorecard. 0.5 = no relation.

| Model | Stage | Devanagari twins | Roman twins |
|---|---|---|---|
| Llama-3.2-3B | S0 base | 0.82 (34 of 139 refused) | 0.93 (8 of 139 refused) |
| Llama-3.2-3B | S1 Hindi SFT | 0.74 (30 of 139 refused) | 0.91 (9 of 139 refused) |
| Llama-3.2-3B | S2 Tameez GRPO | 0.80 (78 of 139 refused) | 0.77 (12 of 139 refused) |
| Tiny Aya Global | S0 base | 0.88 (42 of 139 refused) | 0.85 (7 of 139 refused) |
| Tiny Aya Global | S1 Hindi SFT | 0.78 (32 of 139 refused) | – (0 of 139 refused) |
| Tiny Aya Global | S2 Tameez GRPO | 0.94 (105 of 139 refused) | 0.90 (14 of 139 refused) |
| Tiny Aya Fire | S0 base | 0.86 (52 of 139 refused) | 0.93 (7 of 139 refused) |
| Tiny Aya Fire | S1 Hindi SFT | 0.85 (32 of 139 refused) | 0.89 (1 of 139 refused) |
| Tiny Aya Fire | S2 Tameez GRPO | 0.93 (102 of 139 refused) | 0.78 (8 of 139 refused) |

## 5. Interventions on the final models (Llama-Guard-3-8B + the scorecard refusal matcher)

Greedy replies, 256 new tokens, transformers bf16, on the 139 twins in both scripts and the 516 harmless scorecard prompts. *ablate*: the Devanagari direction removed from the embeddings and from the output of every block. *add α*: α × the direction (unnormalised, so α = 1 moves a harmless prompt to the harmful mean) added at the steering layer, every position. *gated α*: the *add α* reply when a probe trained on Devanagari + Roman training prompts flags the prompt, otherwise the *none* reply. The probe's layer (best held-out Roman AUROC) and threshold (0, the balanced decision boundary) were fixed before the replies were judged.

| Model | Condition | Roman harmful: unsafe | Roman harmful: refused | Deva harmful: unsafe | Deva harmful: refused | Harmless refused: Roman | Deva | English |
|---|---|---|---|---|---|---|---|---|
| Llama-3.2-3B | none | 15.1% | 7.9% | 5.8% | 51.8% | 4.7% | 7.5% | 11.1% |
| Llama-3.2-3B | ablate | 2.2% | 23.7% | 3.6% | 43.9% | – | – | – |
| Llama-3.2-3B | add 0.5 | 2.9% | 33.1% | 0.0% | 81.3% | 9.4% | 30.6% | 25.6% |
| Llama-3.2-3B | add 1.0 | 1.4% | 49.6% | 0.7% | 90.6% | 16.0% | 83.8% | 61.1% |
| Llama-3.2-3B | gated 0.5 | 3.6% | 25.9% | 0.7% | 77.0% | 5.7% | 13.8% | 24.4% |
| Llama-3.2-3B | gated 1.0 | 2.9% | 46.0% | 0.7% | 86.3% | 8.5% | 21.9% | 46.7% |
| Tiny Aya Global | none | 27.3% | 8.6% | 0.7% | 75.5% | 0.0% | 5.3% | 3.3% |
| Tiny Aya Global | ablate | 36.7% | 0.0% | 6.5% | 51.1% | – | – | – |
| Tiny Aya Global | add 0.5 | 12.9% | 33.8% | 0.0% | 93.5% | 9.4% | 30.0% | 34.4% |
| Tiny Aya Global | add 1.0 | 7.9% | 54.0% | 0.0% | 100.0% | 45.3% | 88.1% | 92.2% |
| Tiny Aya Global | gated 0.5 | 14.4% | 33.8% | 0.0% | 87.8% | 5.7% | 8.4% | 16.7% |
| Tiny Aya Global | gated 1.0 | 10.8% | 43.9% | 0.0% | 90.6% | 6.6% | 9.4% | 21.1% |
| Tiny Aya Fire | none | 30.9% | 5.0% | 1.4% | 73.4% | 0.0% | 4.7% | 3.3% |
| Tiny Aya Fire | ablate | 38.8% | 0.7% | 10.1% | 44.6% | – | – | – |
| Tiny Aya Fire | add 0.5 | 23.0% | 15.1% | 0.0% | 90.6% | 0.9% | 31.9% | 31.1% |
| Tiny Aya Fire | add 1.0 | 18.7% | 31.7% | 0.0% | 98.6% | 32.1% | 88.8% | 82.2% |
| Tiny Aya Fire | gated 0.5 | 23.0% | 15.1% | 0.0% | 82.7% | 0.9% | 9.4% | 20.0% |
| Tiny Aya Fire | gated 1.0 | 19.4% | 29.5% | 0.0% | 86.3% | 3.8% | 10.3% | 32.2% |

Share of prompts the gate flags (the *gated* rows add the direction only to these):

| Model | Probe layer | Harmful Roman | Harmful Devanagari | Harmless Roman | Harmless Devanagari | Harmless English |
|---|---|---|---|---|---|---|
| Llama-3.2-3B | 17 | 81% | 90% | 50% | 23% | 52% |
| Tiny Aya Global | 23 | 81% | 87% | 21% | 8% | 22% |
| Tiny Aya Fire | 23 | 83% | 85% | 21% | 11% | 33% |
