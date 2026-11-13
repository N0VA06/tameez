#!/usr/bin/env bash
# Runs on the VM beside the Hinglish job (started over ssh, outside the kernel) and pushes results straight to HF,
# so nothing depends on the VM surviving or on the local watcher:
#   - every finished checkpoint IN FULL (adapter + optimizer/scheduler/RNG state + trainer_state), so a lost VM can
#     resume with train_grpo.py --resume-from <downloaded checkpoint>
#   - the job log, training logs and reward samples every 10 minutes
#   - the scorecards and independent-judge verdicts as soon as they are written
R=/content/tameez_aya; OUT=$R/stages/stage3_hinglish
CK=IINOVAII/Tiny-Aya-Global-Tameez-GRPO-checkpoints; TL=IINOVAII/tameez-training-logs; EV=IINOVAII/tameez-eval-outputs
mkdir -p /content/uploaded /content/livelogs
last_logs=0; final=0
# only lines written after this uploader started count (the log keeps earlier attempts)
start_line=$(wc -l < /content/hinglish_job.log 2>/dev/null || echo 0)
say() { echo "$(date -u +%T) $*"; }
while true; do
  for d in $OUT/model/checkpoint-*; do
    [ -f "$d/trainer_state.json" ] || continue
    n=${d##*-}; [ -f /content/uploaded/ckpt$n ] && continue
    sleep 20   # let the trainer finish writing the folder
    hf upload $CK "$d" hinglish-followup/step-$n --commit-message "Hinglish follow-up GRPO: checkpoint at step $n (with optimizer state for resuming)" >/dev/null 2>&1 \
      && touch /content/uploaded/ckpt$n && say "uploaded checkpoint $n" || say "checkpoint $n upload failed, will retry"
  done
  if [ $(( $(date +%s) - last_logs )) -ge 600 ] || [ $final = 1 ]; then
    cp /content/hinglish_job.log /content/livelogs/ 2>/dev/null; cp $R/results/logs/*.log /content/livelogs/ 2>/dev/null
    [ -d $OUT/train ] && rm -rf /content/livelogs/train && cp -R $OUT/train /content/livelogs/train
    hf upload $TL /content/livelogs tiny_aya/hinglish_followup/live --repo-type dataset --commit-message "Hinglish follow-up GRPO: live logs" >/dev/null 2>&1 \
      && say "uploaded live logs" || say "live log upload failed, will retry"
    last_logs=$(date +%s)
  fi
  for e in stage3_hinglish:stage3_eval stage2_rerun:stage2_rerun_eval; do
    st=${e%%:*}; dst=${e##*:}
    if [ -f $R/stages/$st/eval/metrics.json ] && [ ! -f /content/uploaded/$st ]; then
      hf upload $EV $R/stages/$st/eval tiny_aya/hinglish_followup/$dst/stages/$st/eval --repo-type dataset \
        --commit-message "Hinglish follow-up: scorecard outputs for $st" >/dev/null 2>&1 && touch /content/uploaded/$st && say "uploaded $st scorecard"
    fi
  done
  if [ -f $R/results/independent_judge.jsonl.summary.json ] && [ ! -f /content/uploaded/judge ]; then
    mkdir -p /content/judge_up/results && cp $R/results/independent_judge.jsonl* /content/judge_up/results/
    hf upload $EV /content/judge_up tiny_aya/hinglish_followup/independent_judge --repo-type dataset \
      --commit-message "Hinglish follow-up: Qwen3Guard-Gen-8B verdicts on every Tiny Aya stage" >/dev/null 2>&1 && touch /content/uploaded/judge && say "uploaded judge verdicts"
  fi
  [ $final = 1 ] && { say "done"; break; }
  tail -n +$((start_line + 1)) /content/hinglish_job.log 2>/dev/null | grep -qE "ALL_DONE|FAILED (setup|no HF|smoke|training|eval stage3)" && final=1
  sleep 60
done
