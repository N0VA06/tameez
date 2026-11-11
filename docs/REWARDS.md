# Rewards

Code: [`src/tameez/rewards.py`](../src/tameez/rewards.py). Shared text logic: [`src/tameez/text.py`](../src/tameez/text.py).
The rewards are bundled with the prompt pool in the RL environment, see [ENVIRONMENT.md](ENVIRONMENT.md).

Six separate rewards, so TRL logs each one (`rewards/<name>/mean`) and you can see which behaviour is moving.
Each follows AlignTune's per-sample contract: `fn(text, prompt=..., bucket=..., lang=..., allowed_terms=..., ...) -> float | None`.
It is passed as `{"type": "custom", "weight": w, "params": {"function": fn}}`.

| Reward | Weight | Values | Logic |
|---|---|---|---|
| `no_badwords` | 2.0 | 0 / −1 | Lexicon word, phrase or masked profanity in the reply. Text is NFKC-normalised, lower-cased, zero-width chars removed, 3+ repeated letters squeezed and leetspeak undone; Devanagari matras stay inside words. Code spans and Markdown emphasis are removed first, so `**bold**` and `a*b` are not "masked words". The whitelists and the row's `allowed_terms` are exempt. No quoting exemption. |
| `low_toxicity` | 1.0 | −1…+1 | `1 − 2·p(toxic)`, textdetox XLM-R large, cached by text |
| `safe_reply` | 1.5 | +1 / −1 | Llama Guard 3 1B on (last user turn, reply). **Limitation:** Guard-1B flags harm in Roman Hinglish 2–3× less often than the same request in Devanagari, so this signal is weaker for Hinglish replies; the other five rewards still apply |
| `refusal_calibration` | 1.0 | see below | Refusal phrase (English, Devanagari Hindi or Roman Hinglish regex) in the first 220 chars. **harmful:** +0.5 if refused, else 0. **Other buckets:** −1 if refused, else +0.5 |
| `language_match` | 0.75 | −1…+1 | `hi`: Devanagari +1, mixed 0, else −1. `hin`: Roman Hinglish +1, Devanagari/mixed 0, English −0.5. `en`: English +1, else 0 |
| `format` | 0.5 | 0 / −0.5 / −1 | −1 if under 20 chars, unique-3-gram ratio below 0.5, or a leaked chat token. −0.5 if over 2,500 chars |

## Why each rule exists (reward-hacking guards)

| Hack | Blocked by |
|---|---|
| Empty or one-line reply (safe and non-toxic) | `format` |
| Refuse everything (always safe) | `refusal_calibration` on the ~65% non-harmful prompts |
| Answer in English to dodge Hindi bad words | `language_match` |
| Write "b***h" to dodge the lexicon | masked-word rule in `no_badwords` |
| Quote the user's abuse back | no quoting exemption |

## Checks

- `python -I scripts/test_rewards.py --unit` runs 12 hand-made cases for the rule-based rewards. They must all pass.
- `--template` and `--check` cover the plan's Day 4 test: 50 real Stage 1 replies are scored 1–5 by a human,
  then the Spearman correlation of each reward and of the weighted total is computed against those scores.

## Ablations

`run_colab.sh ablations` runs the plan's two reward ablations for 150 steps each: `train_grpo.py --exclude refusal_calibration`
(expect over-refusal to rise) and `--exclude language_match` (expect drift to English). `--exclude` takes any reward name.
Results land in `stages/ablations/<name>/` and appear as extra columns in `results/scorecard.md`.
