# Results

Every number in the project write-ups comes from a file in this folder. The three runs use the same code, data,
rewards, settings and held-out scorecard; only the base model differs.

| Folder or file | What it holds |
|---|---|
| [`COMPARISON.md`](COMPARISON.md) | the three models, every stage side by side, with exact counts |
| [`tiny_aya_global/`](tiny_aya_global/) | **main run**: Cohere Labs' Tiny Aya Global → Hindi SFT → Tameez GRPO |
| [`tiny_aya_fire/`](tiny_aya_fire/) | the same pipeline on Tiny Aya Fire, Cohere's South Asia variant |
| [`llama_3_2_3b/`](llama_3_2_3b/) | the first run and control, on Llama-3.2-3B-Instruct |
| [`general_benchmarks/`](general_benchmarks/) | XNLI, Belebele, Global-MMLU Lite, GSM8K and IFEval (English and Hindi) for every stage of every model |
| [`api_comparison.md`](api_comparison.md) | Tameez vs Cohere's Command A (111B, paid API) on 200 held-out prompts |
| [`curatorkit_report.md`](curatorkit_report.md) | CuratorKIT audit of the published data splits: duplicates, train-test overlap, secrets |

## Inside each run folder

```
<model>/
├── README.md               headline table and a map of the folder
├── scorecard.md / .csv     every metric of every stage, with 95% Wilson intervals
├── auditkit_report.md      AuditKIT audit of the saved replies
├── plots/                  result and training plots (index in plots/README.md)
└── stages/
    ├── stage0_base/        eval/ (metrics.json, milu.json), env/ (RL environment report)
    ├── stage1_sft/         train/ (training log), eval/, eval_mini/ (checkpoint-selection reference), env/
    └── stage2_grpo/        train/, checkpoints/ (every checkpoint scored), eval/, judge/ (helpfulness), env/
```

## How to read a scorecard

- **Unsafe rate:** share of replies to the 1,498 held-out harmful prompts that Llama-Guard-3-8B judges unsafe. Lower is
  better. The judge is never used in training.
- **Refusal on harmful:** share of harmful prompts refused outright (a phrase matcher). Higher is better.
- **Over-refusal:** share of the 516 harmless prompts (benign, tricky words such as "kill a process", angry users with a
  real question) that the model refuses. Lower is better.
- **Bad words / abusive:** replies with a lexicon swear word, and replies the MuRIL abuse classifier flags. MuRIL
  over-flags Roman Hinglish, so its Hinglish numbers are not reliable.
- **Script fidelity:** the reply is in the same script as the prompt (Devanagari, Roman Hinglish or English).
- **MILU:** 14,831 Hindi multiple-choice questions, zero-shot, first-token log-probability over A–D.
- **Helpfulness:** Aya Expanse 8B compares Stage 2 with Stage 1 on 400 prompts, judged twice in both orders. Ties count
  half.

Every rate carries a 95% Wilson interval in `scorecard.md`. With 139 Hinglish red-teaming prompts, the interval is
about ±7 points, so small differences there are noise.

## Where the replies are

Model replies to harmful prompts are kept out of git. Every reply, judge verdict, checkpoint reply and log is on
Hugging Face, in the same per-model folders:

- [tameez-eval-outputs](https://huggingface.co/datasets/IINOVAII/tameez-eval-outputs): scorecard replies, judge
  verdicts, metrics, checkpoint replies, general-benchmark samples, the API comparison.
- [tameez-training-logs](https://huggingface.co/datasets/IINOVAII/tameez-training-logs): SFT and GRPO curves, reward
  samples, RL rollouts, pipeline logs, TensorBoard.
