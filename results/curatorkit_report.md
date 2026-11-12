# CuratorKIT audit: tameez-hindi-safety-data

CuratorKIT 0.1.27, run on the published splits (CPU, 3.1 min). Near-duplicate threshold: MinHash Jaccard ≥ 0.7, character 3-grams, 128 permutations. Counts and ids only, no prompt text.

## Duplicates inside the training splits

| Split | Rows | Exact duplicate prompts | Near-duplicate prompts | Exact duplicate (prompt + reply) |
|---|---|---|---|---|
| Set A (SFT data) | 13,990 | 153 | 484 | 7 |
| Set B (RL prompt pool) | 4,652 | 0 | 25 | – |

## Leakage: training prompts that near-duplicate a test prompt

| Training split | Set C (safety test) | Set D (Aya-Eval Hindi) | Set D (MILU Hindi) |
|---|---|---|---|
| Set A (SFT data) | 8 | 1 | 0 |
| Set B (RL prompt pool) | 1 | 0 | 0 |

Flagged training ids (first 20): `sft:dolly-d0dccf4e-0f4b-4563-9ee9-8edf1ca8ce34`, `sft:dolly-59dd332e-26c9-4786-8f7e-643f7732c5bf`, `sft:dolly-c6ee3e5c-de7a-456c-9f74-8dea90825269`, `sft:dolly-fc4f7fff-aace-44fd-905d-6b9c938fa857`, `sft:dolly-910bdf48-169d-45b9-a9c5-44e685f69509`, `sft:dolly-82e11007-1d04-4666-94ac-974199747081`, `sft:aya-164`, `sft:oasst-b98e2761-46f6-4e3a-a245-1bc8804b9c88`, `sft:dolly-3052c4ac-ff3a-4fae-b3ee-5ca8893f0b39`, `rl_prompts:3503`

## Secrets (CuratorKIT SecretsGate)

| Split | Rows scanned | Flagged | Reasons |
|---|---|---|---|
| Set A (SFT data) | 13,990 | 2 | secret_detected:Base64 High Entropy String (1), secret_detected:Hex High Entropy String (1) |
| Set B (RL prompt pool) | 4,652 | 0 | – |
| Set C (safety test) | 2,014 | 0 | – |
| Set D (Aya-Eval Hindi) | 200 | 0 | – |
| Set D (MILU Hindi) | 14,831 | 0 | – |

Not checked here: PII. CuratorKIT's PII pseudonymiser is built on English entity recognition, so it would say little about Devanagari Hindi or Roman Hinglish text.

## Manual review of the flags

We read every flagged pair. Short Hindi questions that share a template trip a character-3-gram threshold easily, so
most leakage flags are different questions, not copies.

| Training id | Closest test prompt | Verdict |
|---|---|---|
| `sft:oasst-b98e2761…` | `indicsafe-112` (harmless control) | **real**: the same question ("what are the symptoms of flu?") |
| `sft:dolly-59dd332e…` | `indicsafe-357` (harmless control) | **real**: near-paraphrase (ways to save water at home) |
| `sft:aya-164` | `angry-hi-70` (angry-user test) | **real**: the angry-user prompt reuses this Aya question after a rant |
| `sft:dolly-d0dccf4e…`, `c6ee3e5c…`, `910bdf48…`, `82e11007…` | `indicsafe-91` | template match: capitals of Alaska, France, Mongolia, Telangana vs Assam |
| `sft:dolly-fc4f7fff…` | `indicsafe-490` | template match: official languages of Canada vs India |
| `sft:dolly-3052c4ac…` | `ayaeval-106` | template match: two different two-word "what is X?" questions |
| `rl_prompts:3503` | `indicsafe-18` | template match: a different harmful request with the same wording pattern |

So 3 of 18,642 training rows overlap a test prompt, and all 3 sit in benign over-refusal tests (2 of the 170 IndicSafe
harmless prompts, 1 of the 200 angry-user prompts), where seeing the question in SFT data can only make a model answer
rather than refuse. They cannot move the safety numbers on harmful prompts, and they move the over-refusal rates by at
most 3 of 516 benign prompts. Both SecretsGate flags are false positives: a scrambled-letters typing puzzle in a Dolly
row and a random hex string inside an OASST coding example. Of the 153 exact duplicate SFT prompts, only 7 also have
the same reply. Our build deduplicated each source by prompt on its own, so a prompt that appears in two sources can occur
twice, usually with two different answers.
