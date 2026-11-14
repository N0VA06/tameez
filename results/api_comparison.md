# Tameez vs a large API model

Same 200 held-out prompts, same system prompt (`You are a helpful assistant.`), greedy decoding, up to 512 new tokens. The API model was called through Hugging Face Inference Providers (`CohereLabs/c4ai-command-a-03-2025:cohere`). Every reply, including the Tameez models' original scorecard replies, was judged again on one machine with the same judges as the scorecard: Llama-Guard-3-8B (unsafe), MuRIL (abusive), the bad-word lexicon, the refusal phrase matcher and the script check. Brackets are 95% Wilson intervals: with this many prompts, differences of a few points are within noise.

| Metric | c4ai-command-a-03-2025 (API) | Tiny Aya Global + Tameez | Tiny Aya Fire + Tameez | Llama-3.2-3B + Tameez |
|---|---|---|---|---|
| Unsafe replies, all harmful prompts ↓ | 6.2% [3–14] (n=80) | 3.8% [1–10] (n=80) | 8.8% [4–17] (n=80) | 2.5% [1–9] (n=80) |
| … Hindi red-teaming ↓ | 10.0% [3–26] (n=30) | 0.0% [0–11] (n=30) | 3.3% [1–17] (n=30) | 6.7% [2–21] (n=30) |
| … Hinglish red-teaming ↓ | 5.0% [1–24] (n=20) | 15.0% [5–36] (n=20) | 30.0% [15–52] (n=20) | 0.0% [0–16] (n=20) |
| … IndicSafe harmful ↓ | 3.3% [1–17] (n=30) | 0.0% [0–11] (n=30) | 0.0% [0–11] (n=30) | 0.0% [0–11] (n=30) |
| Explicit refusal of harmful prompts ↑ | 31.2% [22–42] (n=80) | 67.5% [57–77] (n=80) | 61.2% [50–71] (n=80) | 48.8% [38–60] (n=80) |
| Explicit refusal, Hinglish attacks ↑ | 35.0% [18–57] (n=20) | 20.0% [8–42] (n=20) | 5.0% [1–24] (n=20) | 10.0% [3–30] (n=20) |
| Wrongly refuses harmless prompts ↓ | 0.8% [0–5] (n=120) | 5.8% [3–12] (n=120) | 5.0% [2–10] (n=120) | 5.0% [2–10] (n=120) |
| … angry users ↓ | 2.0% [0–10] (n=50) | 14.0% [7–26] (n=50) | 12.0% [6–24] (n=50) | 12.0% [6–24] (n=50) |
| Bad-word replies ↓ | 4.5% [2–8] (n=200) | 1.0% [0–4] (n=200) | 1.5% [1–4] (n=200) | 2.0% [1–5] (n=200) |
| Abusive replies (MuRIL) ↓ | 4.5% [2–8] (n=200) | 7.5% [5–12] (n=200) | 3.0% [1–6] (n=200) | 10.0% [7–15] (n=200) |
| Reply in the user's script ↑ | 91.5% [87–95] (n=200) | 98.5% [96–99] (n=200) | 99.0% [96–100] (n=200) | 99.5% [97–100] (n=200) |
| Mean reply length (characters) | 996 | 432 | 666 | 599 |

Judge check: the local Llama-Guard-3-8B verdicts on the Tameez replies agree with the labels in the original scorecard for Tiny Aya Global + Tameez 99.5%, Llama-3.2-3B + Tameez 100.0% of prompts. Tiny Aya Fire + Tameez was added later and uses the labels from its own scorecard run (the same judges, run on the training GPU) for these 200 prompts, since the re-judged labels of the other Tameez models matched their stored ones that closely.

## Cost

| | Cost per 1,000 replies | How it is measured |
|---|---|---|
| c4ai-command-a-03-2025 (API) | **$4.71** | 200 replies used 10,890 input and 91,401 output tokens ($0.941) at $2.5 / $10 per million tokens (Cohere list price) |
| Tiny Aya + Tameez, self-hosted on Colab L4 (~$0.17/h) | **$0.003** | the final scorecard generated 2,214 replies in 130 s with vLLM on one L4 (59 ms per reply when batched) |
| Tiny Aya + Tameez, self-hosted on AWS g6.xlarge L4 (~$0.80/h on demand) | **$0.013** | the final scorecard generated 2,214 replies in 130 s with vLLM on one L4 (59 ms per reply when batched) |

API latency: median 14.0 s per reply (one request at a time per worker).

The self-hosted figure assumes a busy GPU running batched requests; an idle GPU still costs money, so at low traffic a pay-per-token API can be cheaper overall. The one-off training cost of Tiny Aya + Tameez was about 6 h on a Colab L4 (roughly 11 compute units, about $1), not counting data generation and earlier runs.
