#!/usr/bin/env bash
# Run every model in sequence on one GPU. The gated base Tiny Aya runs last and waits for a Hugging Face login.
#   nohup bash run_all.sh > /content/run_all.log 2>&1 &
cd /content
run() {  # model short_name
  echo "$(date -u +%H:%M:%S) START $2 ($1)"
  if bash /content/run_benchmarks.sh "$1" "$2" 500 > "/content/$2.log" 2>&1; then
    echo "$(date -u +%H:%M:%S) DONE $2"
  else
    echo "$(date -u +%H:%M:%S) FAILED $2 (see /content/$2.log)"
  fi
}
run IINOVAII/Tiny-Aya-Global-Tameez-GRPO tiny_aya_tameez
run unsloth/Llama-3.2-3B-Instruct llama_base
run IINOVAII/Llama-3.2-3B-Tameez-GRPO llama_tameez
for i in $(seq 1 180); do   # wait up to 3 h for `hf auth login` on the VM
  python -c "from huggingface_hub import HfApi; HfApi().model_info('CohereLabs/tiny-aya-global', files_metadata=False); import huggingface_hub as h; h.hf_hub_download('CohereLabs/tiny-aya-global','config.json')" > /dev/null 2>&1 && break
  [ $i -eq 1 ] && echo "$(date -u +%H:%M:%S) WAITING for Hugging Face login to access CohereLabs/tiny-aya-global"
  sleep 60
done
run CohereLabs/tiny-aya-global tiny_aya_base
echo "$(date -u +%H:%M:%S) ALL_DONE"
