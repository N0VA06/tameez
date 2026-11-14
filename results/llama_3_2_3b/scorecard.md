# Tameez scorecard

Same prompts, greedy decoding, system prompt "You are a helpful assistant." Safety is judged by models that are NOT used in the reward (Guard-3-8B, MuRIL). Brackets = 95% Wilson interval.

| Metric | | stage0_base | stage1_sft | stage2_grpo |
|---|---|---|---|---|
| Unsafe reply rate, all harmful (Guard-3-8B) | ↓ | 10.0% [8.6%–11.6%] (n=1498) | 15.6% [13.9%–17.5%] (n=1498) | 3.7% [2.8%–4.8%] (n=1498) |
|   Aya Red-teaming Hindi | ↓ | 10.0% [8.2%–12.1%] (n=913) | 16.0% [13.8%–18.5%] (n=913) | 2.6% [1.8%–3.9%] (n=913) |
|   Aya Red-teaming Hinglish (rewritten) | ↓ | 21.6% [15.6%–29.1%] (n=139) | 30.9% [23.8%–39.1%] (n=139) | 15.8% [10.7%–22.8%] (n=139) |
|   IndicSafe Hindi harmful | ↓ | 6.4% [4.2%–9.5%] (n=330) | 11.2% [8.2%–15.1%] (n=330) | 2.7% [1.4%–5.1%] (n=330) |
|   SurakshaEval Hindi | ↓ | 6.9% [3.5%–13.0%] (n=116) | 6.9% [3.5%–13.0%] (n=116) | 0.0% [0.0%–3.2%] (n=116) |
| Bad-word rate, safety + angry prompts (lexicon) | ↓ | 2.8% [2.2%–3.6%] (n=2014) | 3.0% [2.4%–3.9%] (n=2014) | 0.7% [0.4%–1.2%] (n=2014) |
|   angry-user prompts only | ↓ | 3.0% [1.4%–6.4%] (n=200) | 5.0% [2.7%–9.0%] (n=200) | 1.5% [0.5%–4.3%] (n=200) |
| Abusive rate (MuRIL) | ↓ | 3.6% [2.9%–4.5%] (n=2014) | 6.9% [5.8%–8.0%] (n=2014) | 7.5% [6.5%–8.8%] (n=2014) |
| Over-refusal, all benign | ↓ | 2.1% [1.2%–3.8%] (n=516) | 3.3% [2.1%–5.2%] (n=516) | 7.0% [5.1%–9.5%] (n=516) |
|   IndicSafe harmless + tricky | ↓ | 0.0% [0.0%–2.2%] (n=170) | 0.0% [0.0%–2.2%] (n=170) | 1.2% [0.3%–4.2%] (n=170) |
|   own tricky test | ↓ | 0.7% [0.1%–3.8%] (n=146) | 0.7% [0.1%–3.8%] (n=146) | 2.1% [0.7%–5.9%] (n=146) |
|   angry-user test | ↓ | 5.0% [2.7%–9.0%] (n=200) | 8.0% [5.0%–12.6%] (n=200) | 15.5% [11.1%–21.2%] (n=200) |
| Refusal on harmful (detector) | ↑ | 31.9% [29.5%–34.4%] (n=1382) | 28.3% [26.0%–30.7%] (n=1382) | 58.5% [55.9%–61.1%] (n=1382) |
| Script fidelity, all | ↑ | 99.6% [99.3%–99.8%] (n=2214) | 99.8% [99.5%–99.9%] (n=2214) | 99.7% [99.4%–99.9%] (n=2214) |
|   Hindi (Devanagari) | ↑ | 99.9% [99.6%–100.0%] (n=1879) | 100.0% [99.8%–100.0%] (n=1879) | 100.0% [99.8%–100.0%] (n=1879) |
|   Hinglish (Roman) | ↑ | 97.6% [94.8%–98.9%] (n=245) | 98.4% [95.9%–99.4%] (n=245) | 98.0% [95.3%–99.1%] (n=245) |
|   English | ↑ | 100.0% [95.9%–100.0%] (n=90) | 100.0% [95.9%–100.0%] (n=90) | 98.9% [94.0%–99.8%] (n=90) |
| MILU Hindi accuracy (0-shot) | ↑ | 38.5% [37.8%–39.3%] (n=14831) | 39.3% [38.6%–40.1%] (n=14831) | 39.5% [38.7%–40.3%] (n=14831) |
| Helpfulness win rate vs Stage 1 (Aya Expanse judge) | ↑ | — | — | 42.1% [37.3%–47.0%] (n=400) |
| Mean reply length (chars) |  | 827.5 | 645.6 | 376.0 |
