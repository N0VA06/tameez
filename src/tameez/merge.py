"""Merge a LoRA adapter into its base model so vLLM can serve it."""

import gc
from pathlib import Path


def merge_adapter(base, adapter_dir, out_dir):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(out_dir)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    tok_src = adapter_dir if (Path(adapter_dir) / "tokenizer_config.json").exists() else base
    AutoTokenizer.from_pretrained(tok_src).save_pretrained(out_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out_dir
