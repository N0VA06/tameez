#!/usr/bin/env bash
# Benchmark the two Stage 1 (Hindi SFT) models with the same settings, to split base -> SFT -> Tameez.
#   nohup bash run_sft.sh > /content/run_sft.log 2>&1 &
cd /content
run() {
  echo "$(date -u +%H:%M:%S) START $2 ($1)"
  if bash /content/run_benchmarks.sh "$1" "$2" 500 > "/content/$2.log" 2>&1; then echo "$(date -u +%H:%M:%S) DONE $2"
  else echo "$(date -u +%H:%M:%S) FAILED $2 (see /content/$2.log)"; fi
}
run IINOVAII/Tiny-Aya-Global-Hindi-SFT tiny_aya_sft
run IINOVAII/Llama-3.2-3B-Hindi-SFT llama_sft
echo "$(date -u +%H:%M:%S) ALL_DONE"
