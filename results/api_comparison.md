# Tameez vs a large API model

Same 200 held-out prompts, same system prompt (`You are a helpful assistant.`), greedy decoding, up to 512 new tokens. The API model was called through Hugging Face Inference Providers (`CohereLabs/c4ai-command-a-03-2025:cohere`). Every reply, including the Tameez models' original scorecard replies, was judged again on one machine with the same judges as the scorecard: Llama-Guard-3-8B (unsafe), MuRIL (abusive), the bad-word lexicon, the refusal phrase matcher and the script check. Brackets are 95% Wilson intervals: with this many prompts, differences of a few points are within noise.

| Metric | c4ai-command-a-03-2025 (API) | Tiny Aya Global + Tameez | Llama-3.2-3B + Tameez |
|---|---|---|---|
| Unsafe replies, all harmful prompts ↓ | 6.2% [3–14] (n=80) | 3.8% [1–10] (n=80) | 2.5% [1–9] (n=80) |
| … Hindi red-teaming ↓ | 10.0% [3–26] (n=30) | 0.0% [0–11] (n=30) | 6.7% [2–21] (n=30) |
| … Hinglish red-teaming ↓ | 5.0% [1–24] (n=20) | 15.0% [5–36] (n=20) | 0.0% [0–16] (n=20) |
| … IndicSafe harmful ↓ | 3.3% [1–17] (n=30) | 0.0% [0–11] (n=30) | 0.0% [0–11] (n=30) |
| Explicit refusal of harmful prompts ↑ | 31.2% [22–42] (n=80) | 67.5% [57–77] (n=80) | 48.8% [38–60] (n=80) |
| Explicit refusal, Hinglish attacks ↑ | 35.0% [18–57] (n=20) | 20.0% [8–42] (n=20) | 10.0% [3–30] (n=20) |
| Wrongly refuses harmless prompts ↓ | 0.8% [0–5] (n=120) | 5.8% [3–12] (n=120) | 5.0% [2–10] (n=120) |
| … angry users ↓ | 2.0% [0–10] (n=50) | 14.0% [7–26] (n=50) | 12.0% [6–24] (n=50) |
| Bad-word replies ↓ | 4.5% [2–8] (n=200) | 1.0% [0–4] (n=200) | 2.0% [1–5] (n=200) |
| Abusive replies (MuRIL) ↓ | 4.5% [2–8] (n=200) | 7.5% [5–12] (n=200) | 10.0% [7–15] (n=200) |
| Reply in the user's script ↑ | 91.5% [87–95] (n=200) | 98.5% [96–99] (n=200) | 99.5% [97–100] (n=200) |
| Mean reply length (characters) | 996 | 432 | 599 |

Judge check: the local Llama-Guard-3-8B verdicts on the Tameez replies agree with the labels in the original scorecard for Tiny Aya Global + Tameez 99.5%, Llama-3.2-3B + Tameez 100.0% of prompts.

## Cost

| | Cost per 1,000 replies | How it is measured |
|---|---|---|
| c4ai-command-a-03-2025 (API) | **$4.71** | 200 replies used 10,890 input and 91,401 output tokens ($0.941) at $2.5 / $10 per million tokens (Cohere list price) |
| Tiny Aya + Tameez, self-hosted on Colab L4 (~$0.17/h) | **$0.003** | the final scorecard generated 2,214 replies in 130 s with vLLM on one L4 (59 ms per reply when batched) |
| Tiny Aya + Tameez, self-hosted on AWS g6.xlarge L4 (~$0.80/h on demand) | **$0.013** | the final scorecard generated 2,214 replies in 130 s with vLLM on one L4 (59 ms per reply when batched) |

API latency: median 14.0 s per reply (one request at a time per worker).

The self-hosted figure assumes a busy GPU running batched requests; an idle GPU still costs money, so at low traffic a pay-per-token API can be cheaper overall. The one-off training cost of Tiny Aya + Tameez was about 6 h on a Colab L4 (roughly 11 compute units, about $1), not counting data generation and earlier runs.

## What this shows

**Safety is comparable.** On these 80 harmful prompts the two 3B Tameez models give unsafe replies about as often as
Command A, a 111B model behind a paid API (3.8% and 2.5% against 6.2%). The intervals overlap, so the honest reading is
"about the same", not "better". Command A is stronger on the 20 Hinglish attacks (5.0% unsafe against Tiny Aya's
15.0%), which matches the weakness reported in the main results; 20 prompts is too few to say more.

**They decline in different styles.** Tameez answers harmful requests with a short, polite refusal in the user's
script (67.5% of harmful prompts for Tiny Aya). Command A usually declines by explaining why the request is wrong and
redirecting, which Llama-Guard rates as safe but the refusal phrase matcher does not count; its 31.2% "explicit
refusal" understates how often it declines. Its replies are also more than twice as long (996 against 432 characters).

**Command A is better at not over-refusing.** It wrongly refused 0.8% of harmless prompts (1 of 120) and 2% of
angry-user prompts, against 5.8% and 14% for Tiny Aya + Tameez. Over-refusal on angry users is the main cost of the
Tameez reward design, as the main results already say.

**Tameez keeps to the user's script and avoids repeating abuse.** Tiny Aya + Tameez replied in the user's script
98.5% of the time (Command A 91.5%) and used a lexicon bad word in 1.0% of replies (Command A 4.5%, mostly from quoting
or explaining the user's abuse while declining, and one gardening answer about "weed"; the lexicon is strict).

**The cost difference is large.** At Cohere's list price, Command A cost $4.71 per 1,000 replies here. Tiny Aya +
Tameez self-hosted on one L4 costs about $0.003 per 1,000 replies on Colab or $0.013 on AWS when the GPU is kept busy
with batched requests, a few hundred times less, plus about $1 of one-off training compute. The fine-tuned model is also
small enough to run on one 24 GB GPU, offline, with user data staying on your own machine.

**Caveats.** 200 prompts (80 harmful, 120 harmless) is a small sample; Command A was run once with greedy decoding
through one provider; the self-hosted cost ignores idle time, engineering and data-generation compute; and the phrase
matcher and lexicon were built for the Tameez models' style, so they are stricter with a model that writes long
explanatory answers. The replies and every judge verdict are in the `tameez-eval-outputs` dataset under
`api_baseline/`, so every number here can be checked.
