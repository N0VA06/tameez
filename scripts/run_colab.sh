#!/usr/bin/env bash
# Run one pipeline step on the Colab VM, log it, and copy everything it produced to Google Drive.
#
#   bash scripts/run_colab.sh <step>
#
# Steps, in order (see docs/PIPELINE.md):
#   setup      install AlignTune/CuratorKIT/vLLM, restore data/stages/results from Drive
#   lexicon    merge bad-word lists                              -> data/lexicons/
#   synthetic  Aya Expanse Hinglish rewrites + paraphrases       -> data/processed/cache/synthetic.json
#   data       build sets A-D with Llama Guard 1B filter         -> data/processed/
#   env        validate the RL environment (prompt pool)         -> stages/stage1_sft/env/env_check.json
#   stage0     base model: scorecard + env rollout               -> stages/stage0_base/{eval,env}/
#   sft        Stage 1 Hindi SFT + merge                         -> stages/stage1_sft/{model,train}/
#   stage1     Stage 1: scorecard + env rollout                  -> stages/stage1_sft/{eval,env}/
#   rewards    reward unit tests + labelling sheet               -> results/reward_check/
#   pilot      20-step GRPO pilot + samples per bucket           -> stages/stage2_grpo/pilot/
#   envs       re-run the RL-environment rollout of stage0 + stage1 (after the pool changes)
#   grpo       300-step GRPO (L4 settings, ~2.7 h)               -> stages/stage2_grpo/{model,train}/
#   ckpts      score every checkpoint on the mini-set, keep best -> stages/stage2_grpo/checkpoints/, model/best_merged
#   stage2     Stage 2: scorecard + env rollout + judge          -> stages/stage2_grpo/{eval,env,judge}/
#   ablations  plan's ablations, 60 steps each + their scorecards -> stages/ablations/<name>/
#   plots      redraw all result/data plots (also runs after every stage step) -> results/plots/
set -euo pipefail
cd "$(dirname "$0")/.."
DRIVE=${DRIVE:-/content/drive/MyDrive/tameez}
STEP=${1:?usage: run_colab.sh <step>}
mkdir -p results/logs

sync_up() {
  [ -d "$DRIVE" ] || { echo "Drive not mounted at $DRIVE: results stay on the VM only"; return; }
  mkdir -p data/processed data/lexicons results stages "$DRIVE/data" "$DRIVE/results" "$DRIVE/stages"
  rsync -a data/processed data/lexicons "$DRIVE/data/"
  rsync -a results/ "$DRIVE/results/"
  rsync -a --exclude '_tmp_merged/' stages/ "$DRIVE/stages/"
  echo "synced to $DRIVE"
}

# While a long step runs, copy its stage folder to Drive every 10 min, so a lost VM costs at most
# 10 minutes of checkpoints (Colab can end a runtime without warning).
live_sync() {
  [ -d "$DRIVE" ] || return
  local dir=$1
  ( while sleep 600; do
      [ -d "$dir" ] && mkdir -p "$DRIVE/$dir" && rsync -a --exclude '_tmp_merged/' "$dir/" "$DRIVE/$dir/" \
        && echo "   $(date '+%T') live sync: $dir -> Drive"
    done ) &
  LIVE_SYNC_PID=$!
}
stop_live_sync() { [ -n "${LIVE_SYNC_PID:-}" ] && kill "$LIVE_SYNC_PID" 2>/dev/null; LIVE_SYNC_PID=; }
trap stop_live_sync EXIT

run() {
  local t0=$SECONDS
  echo "== $(date '+%F %T') $STEP START: $*" | tee -a "results/logs/$STEP.log"
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/   GPU memory used: /' | tee -a "results/logs/$STEP.log"
  "$@" 2>&1 | tee -a "results/logs/$STEP.log"
  echo "== $(date '+%F %T') $STEP END after $(( (SECONDS - t0) / 60 ))m$(( (SECONDS - t0) % 60 ))s" | tee -a "results/logs/$STEP.log"
}

BASE=${TAMEEZ_POLICY:-unsloth/Llama-3.2-3B-Instruct}   # policy model; Cohere rerun: TAMEEZ_POLICY=CohereLabs/tiny-aya-global
DATA_SRC=${DATA_SRC:-$DRIVE}                         # where setup copies data/ from (reuse the built sets of another run)
case "$STEP" in
  setup)
    pip install -q uv
    [ -d /content/aligntune ] || git clone -q --depth 1 https://github.com/Lexsi-Labs/aligntune.git /content/aligntune
    uv pip install -q -e /content/aligntune
    uv pip install -q "curatorkit[all]" datasets accelerate peft bitsandbytes vllm datasketch tensorboard
    [ -d "$DATA_SRC/data" ] && rsync -a "$DATA_SRC/data/" data/
    if [ -d "$DRIVE" ]; then
      [ -d "$DRIVE/results" ] && rsync -a "$DRIVE/results/" results/
      [ -d "$DRIVE/stages" ] && rsync -a "$DRIVE/stages/" stages/
    fi
    python -c "import torch, trl, vllm, aligntune; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'trl', trl.__version__, 'vllm', vllm.__version__)"
    exit 0 ;;
  lexicon)   run python -I scripts/build_lexicon.py ;;
  synthetic) run python scripts/make_synthetic.py ;;
  data)      run python scripts/build_data.py --guard
             run python -I scripts/env_check.py --stage stage1_sft ;;
  env)       run python -I scripts/env_check.py --stage stage1_sft ;;
  envs)      rm -f stages/stage0_base/env/rollout*.json stages/stage1_sft/env/rollout*.json
             run python scripts/env_check.py --stage stage0_base --rollout --model $BASE
             run python scripts/env_check.py --stage stage1_sft --rollout ;;
  stage0)    run python scripts/evaluate.py --stage stage0_base --model $BASE
             run python scripts/env_check.py --stage stage0_base --rollout --model $BASE ;;
  sft)       live_sync stages/stage1_sft; run python scripts/train_sft.py; stop_live_sync ;;
  stage1)    run python scripts/evaluate.py --stage stage1_sft --model stages/stage1_sft/model/merged
             run python scripts/env_check.py --stage stage1_sft --rollout ;;
  rewards)   run python -I scripts/test_rewards.py --unit
             run python -I scripts/test_rewards.py --template ;;
  pilot)     run python scripts/train_grpo.py --out stages/stage2_grpo/pilot --max-steps 20 --save-steps 10
             run python -I scripts/show_samples.py --dir stages/stage2_grpo/pilot --per-bucket 30 ;;
  grpo)      live_sync stages/stage2_grpo
             run python scripts/train_grpo.py
             stop_live_sync
             run python -I scripts/show_samples.py --dir stages/stage2_grpo --per-bucket 30 --last-minutes 30 ;;
  ckpts)     live_sync stages/stage2_grpo; run python scripts/score_checkpoints.py; stop_live_sync ;;
  stage2)    MODEL=stages/stage2_grpo/model/best_merged; [ -d "$MODEL" ] || MODEL=stages/stage2_grpo/model/merged
             run python scripts/evaluate.py --stage stage2_grpo --model "$MODEL"
             run python scripts/env_check.py --stage stage2_grpo --rollout --model "$MODEL"
             run python scripts/judge.py --stage stage2_grpo ;;
  ablations) for spec in "no_refusal:refusal_calibration" "no_language:language_match"; do
               name=${spec%%:*}; reward=${spec##*:}; dir=stages/ablations/$name
               run python scripts/train_grpo.py --out $dir --max-steps 60 --save-steps 60 --exclude $reward
               run python scripts/evaluate.py --stage ablations/$name --model $dir/model/merged --out $dir/eval
             done ;;
  plots)     ;;
  *) echo "unknown step: $STEP"; exit 1 ;;
esac
case "$STEP" in   # redraw plots after anything that changes results or data
  data|envs|stage0|stage1|stage2|ablations|plots) run python -I scripts/make_plots.py ;;
esac
sync_up
