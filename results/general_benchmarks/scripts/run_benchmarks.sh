#!/usr/bin/env bash
# General-capability benchmarks for the Tameez models, before vs after (lm-evaluation-harness + vLLM).
# Same settings for every model: chat template, greedy decoding, at most 500 examples per (sub)task.
# Knowledge is measured with Global-MMLU Lite in English and Hindi (the same ~400 MMLU questions in both languages);
# the full 57-subject MMLU was dropped because it needs ~56k scoring requests per model (~1.5 GPU hours on an L4).
#
#   bash run_benchmarks.sh <hf_model_id> <short_name> [limit]
#
# Results: /content/results/<short_name>/  (lm-eval JSON + per-sample logs), log: /content/<short_name>.log
set -euo pipefail
MODEL="$1"; NAME="$2"; LIMIT="${3:-500}"
TASKS="xnli_en,xnli_hi,belebele_eng_Latn,belebele_hin_Deva,global_mmlu_en,global_mmlu_hi,gsm8k,ifeval"
mkdir -p /content/results/"$NAME"
lm_eval \
  --model vllm \
  --model_args "pretrained=${MODEL},dtype=bfloat16,gpu_memory_utilization=0.85,max_model_len=4096,seed=1234,trust_remote_code=False" \
  --tasks "$TASKS" \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --gen_kwargs "temperature=0,do_sample=False" \
  --limit "$LIMIT" \
  --batch_size auto \
  --seed 1234 \
  --log_samples \
  --output_path /content/results/"$NAME"
