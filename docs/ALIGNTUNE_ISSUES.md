# AlignTune issues: how to reproduce and fix

Three issues found while using AlignTune for Tameez, re-checked on a fresh install. Each section has the symptom, a minimal reproduction, the cause in AlignTune's code, a workaround you can use today, and a suggested fix.

**Tested with:** AlignTune 0.1.30 (git commit `c3d648f`), TRL 1.7.1, transformers 5.22.0, vLLM 0.22.1, torch 2.11.0+cu130, Python 3.13, NVIDIA L4 (Google Colab). Re-tested 2026-11-04.

| \# | Issue | Reproduced? | Workaround |
| :---- | :---- | :---- | :---- |
| 1 | `backend="unsloth"` crashes rewards that load a Llama model | ✅ directly and via `create_rl_trainer` | use `backend="trl"` |
| 2 | `rollout_backend="vllm"` fails by default on the TRL backend | ✅ | set `ALIGNTUNE_ENABLE_VLLM=1` before importing AlignTune |
| 3 | `aligntune info` always shows ✗ TRL | ✅ | ignore it, or call `aligntune._imports._check_trl_available()` |

## Setup for all reproductions

```sh
git clone https://github.com/Lexsi-Labs/aligntune.git && cd aligntune && git checkout c3d648f
pip install uv && uv pip install -e . && uv pip install vllm
```

Models used (both public, no gated access needed): `unsloth/Llama-3.2-1B-Instruct` (policy) and `alpindale/Llama-Guard-3-1B` (a byte-identical mirror of `meta-llama/Llama-Guard-3-1B`).

A custom reward that scores a completion with Llama Guard, saved as `guard_reward.py`:

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "alpindale/Llama-Guard-3-1B"
_tok = _model = None


def guard_reward(text, prompt=None, **kwargs):
    global _tok, _model
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(NAME)
        _model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.bfloat16).cuda().eval()
    conv = [{"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": text[:500]}]}]
    ids = _tok(_tok.apply_chat_template(conv, tokenize=False), return_tensors="pt",
               add_special_tokens=False).to("cuda")
    with torch.no_grad():
        out = _model.generate(**ids, max_new_tokens=5, do_sample=False, pad_token_id=_tok.eos_token_id)
    verdict = _tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
    return -1.0 if verdict.startswith("unsafe") else 1.0
```

A tiny 1-step GRPO run, saved as `grpo_one_step.py` (argv: `backend` \= `trl|unsloth`, `rollout` \= `hf|vllm`):

```py
import json, sys
from aligntune.core.backend_factory import create_rl_trainer
from guard_reward import guard_reward

backend, rollout = sys.argv[1], sys.argv[2]
with open("tiny.json", "w") as f:
    for q in ["How do I bake bread?", "What is the capital of India?", "Tell me a joke.", "How do plants grow?"] * 2:
        f.write(json.dumps({"prompt": [{"role": "user", "content": q}], "reference": "help"}) + "\n")

trainer = create_rl_trainer(
    model_name="unsloth/Llama-3.2-1B-Instruct", dataset_name="tiny.json", algorithm="grpo",
    backend=backend, rollout_backend=rollout, output_dir=f"out_{backend}_{rollout}",
    max_steps=1, batch_size=4, gradient_accumulation_steps=1, num_generations=2,
    max_completion_length=16, max_prompt_length=64, max_seq_length=256, learning_rate=1e-6,
    use_peft=True, lora_r=8, lora_alpha=16, bf16=True, logging_steps=1, report_to="none",
    rewards=[{"type": "custom", "weight": 1.0, "params": {"function": guard_reward}}],
)
trainer.train()
print("TRAINED OK")
```

---

## 1\. `backend="unsloth"` crashes reward functions that load a Llama model

**Symptom:** every reward call fails during GRPO:

```
RuntimeError: Registry reward 'guard_reward' failed for completion 0:
'LlamaAttention' object has no attribute 'apply_qkv'
```

**Reproduce:**

```sh
python grpo_one_step.py unsloth hf   # FAILS with the error above
python grpo_one_step.py trl hf       # TRAINED OK (same reward)
```

Minimal version without AlignTune, which shows the root cause:

```py
from unsloth import FastLanguageModel
FastLanguageModel.from_pretrained("unsloth/Llama-3.2-1B-Instruct", max_seq_length=512)
from guard_reward import guard_reward
guard_reward("Sure, here is a cake recipe.")   # AttributeError: ... 'apply_qkv'
# Without the two Unsloth lines, guard_reward(...) returns 1.0
```

**Cause:** Unsloth (bundled in `third_party/unsloth`) replaces the attention code for the whole Python process:

- `unsloth/models/llama.py` L2247: `LlamaAttention.forward = LlamaAttention_fast_forward` (class-level)  
- L695: the new forward calls `self.apply_qkv(...)`  
- L2779 / L3669: `apply_qkv` is only attached to layers of models loaded *through Unsloth*

Any second Llama model loaded normally in the same process (judge, guard or classifier in a reward) gets the new forward without `apply_qkv` and crashes. AlignTune's Unsloth backend doesn't isolate reward models or warn.

**Workaround:** use `backend="trl"` when any reward runs a Llama-family model.

**Suggested fix (AlignTune):** document the limitation; warn when a reward loads a Llama model while the Unsloth backend is active; or run model-based rewards in a separate process (e.g. `multiprocessing` with `spawn`).

---

## 2\. `rollout_backend="vllm"` fails by default on the TRL backend

**Symptom:**

```
ImportError: vLLM is not available and `use_vllm` is set to True.
Please install vLLM with `pip install trl[vllm]` to use it.
```

…although vLLM 0.22.1 is installed, and `docs/PARAMETERS.md` L623 lists `'vllm'` as supported on TRL.

**Reproduce:**

```sh
python grpo_one_step.py trl vllm                          # FAILS with the error above
ALIGNTUNE_ENABLE_VLLM=1 python grpo_one_step.py trl vllm  # TRAINED OK
```

With the variable set, the trainer really uses vLLM: `trainer.trainer.args.use_vllm == True`, `vllm_mode == "colocate"`, and `trainer.trainer.vllm_generation.llm` is a `vllm.LLM`.

**Cause:** `src/aligntune/__init__.py` L35–41 replaces TRL's check whenever AlignTune is imported:

```py
# TRL 1.7 GRPOTrainer always imports VLLMGeneration. If vllm is installed but
# its CUDA 13 runtime is missing, that import kills the cell. Notebooks use
# HF rollout; leave vllm off unless ALIGNTUNE_ENABLE_VLLM=1.
if os.environ.get("ALIGNTUNE_ENABLE_VLLM", "0") != "1":
    import trl.import_utils as _trl_iu
    _trl_iu.is_vllm_available = lambda *a, **k: False
```

The workaround is reasonable, but the variable is not mentioned in the docs, README or CHANGELOG, and TRL's error tells the user to reinstall vLLM.

**Workaround:** set `ALIGNTUNE_ENABLE_VLLM=1` **before** `import aligntune` (`os.environ["ALIGNTUNE_ENABLE_VLLM"] = "1"` at the top of the script, or in the shell).

**Suggested fix (AlignTune):** mention the variable next to `rollout_backend` in `docs/PARAMETERS.md`, and in the TRL GRPO backend raise a clear error when `rollout_backend == "vllm"` and the variable is not set, e.g. `ValueError("rollout_backend='vllm' needs ALIGNTUNE_ENABLE_VLLM=1 set before importing aligntune")`.

---

## 3\. `aligntune info` always shows ✗ TRL

**Symptom:** `aligntune info` prints `✗ TRL` (Available Trainers) and `✗ trl` (Dependencies), while the same output shows `Backend Status: ✓ TRL`, and TRL imports and trains fine.

**Reproduce:**

```py
import aligntune
print(aligntune.check_dependencies()["trl"])   # None   (shown as ✗)
from trl import GRPOTrainer                     # works
```

**Cause:** `src/aligntune/_imports.py` L168 defines `TRL_AVAILABLE = None  # Will be set lazily`, and `_check_trl_available()` sets it, but `check_dependencies()` reads the flag without calling the check. The Unsloth entry next to it calls `_check_unsloth_available()`.

**Suggested fix (one line in `check_dependencies()`, `src/aligntune/_imports.py` L539–543; `_check_trl_available()` is defined in the same file at L171):**

```diff
-        "trl": TRL_AVAILABLE,
+        "trl": _check_trl_available(),
```

&nbsp;

---

## Checked and not reported (intended behaviour)

| Observation | Why it is not a bug |
| :---- | :---- |
| Custom rewards logged as `rewards/registry_<name>/mean` | Every reward, built-in or custom, goes through the same wrapper (`wrap_registry_reward`), which names them all `registry_<name>`: a consistent naming scheme |
| `aligntune info` shows ✗ `unified_sft` | Those modules were removed on purpose (`_imports.py`: "These modules don't exist") |
| Unsloth backend ignores `rollout_backend="vllm"` | Documented: vLLM rollouts are TRL-only (`docs/PARAMETERS.md` L623) |
| Warning about a missing OpenAI key for the `rubric` reward, and "import unsloth first" | Cosmetic; the rubric reward is not used, and the Unsloth notice only appears with the Unsloth backend |

- &nbsp;  
- &nbsp;