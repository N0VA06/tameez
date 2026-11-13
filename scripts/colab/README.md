# Colab drivers for the extra Hinglish GRPO run

Run on one Colab L4 with the [Colab CLI](https://pypi.org/project/google-colab-cli/). Colab ends sessions whose kernel is
idle, so each job runs inside a single `colab exec` call and keeps the kernel busy until it finishes.

| File | What it does |
|---|---|
| `setup_vm.sh` | evaluation stack: vLLM 0.33.0, transformers 5.20.2, lm-evaluation-harness 0.4.13 |
| `hinglish_run_job.py` | training env (AlignTune 0.1.35, TRL 1.7.1) in `/content/trainenv`; waits for an HF login (gated Llama-Guard-3-8B); 3-step smoke test (falls back to 4 × 4 micro-batches on out-of-memory); 150-step GRPO on the Hinglish prompts; scorecard of the new model; control re-run of Stage 2; Qwen3Guard-Gen-8B verdicts on every stage |
| `hinglish_run_uploader.sh` | started over `colab ssh` next to the job: pushes every checkpoint in full (with optimizer state, for `train_grpo.py --resume-from`), live logs every 10 minutes and each scorecard to Hugging Face as soon as it exists |
| `hinglish_run_bench.py` | lm-evaluation-harness general benchmarks of the new model with the same `run_benchmarks.sh` as the other six models |

```bash
colab new -s tameez-evals --gpu L4
colab upload -s tameez-evals setup_vm.sh /content/setup_vm.sh       # plus a tarball of the repo as /content/tameez_aya
colab exec -s tameez-evals -f hinglish_run_job.py --timeout 0
colab ssh -s tameez-evals -- hf auth login                         # in another terminal, while the job waits
colab ssh -s tameez-evals -- "setsid nohup bash /content/hinglish_run_uploader.sh > /content/uploader.log 2>&1 &"
```
