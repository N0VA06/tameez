"""vLLM helpers: batched chat generation and Llama Guard classification (GPU only)."""

import gc


def load(model, max_model_len=2048, gpu_memory_utilization=0.88, log=None, **kwargs):
    """Load a vLLM engine. If less GPU memory is free than requested, shrink the request
    (instead of vLLM's hard failure) and say so; a model that truly does not fit still fails."""
    import torch
    from vllm import LLM
    free, total = torch.cuda.mem_get_info()
    msg = f"loading {model} (max_model_len={max_model_len}): {free / 2**30:.1f}/{total / 2**30:.1f} GiB free"
    if free / total < gpu_memory_utilization + 0.01:
        new = round(free / total - 0.03, 2)
        msg += f"; requested {gpu_memory_utilization:.0%} of the GPU, using {new:.0%} (other processes hold memory)"
        gpu_memory_utilization = new
    (log.info if log else print)(msg)
    return LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
               dtype="bfloat16", seed=42, enable_prefix_caching=True, **kwargs)


def unload(llm):
    """Free GPU memory so the next model can load in the same process."""
    import torch
    try:
        llm.llm_engine.engine_core.shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def chat(llm, conversations, temperature=0.0, top_p=1.0, max_tokens=512, system_prompt=None):
    """Generate one reply per conversation (list of message lists). Returns list[str]."""
    from vllm import SamplingParams
    if system_prompt:
        conversations = [[{"role": "system", "content": system_prompt}] + c for c in conversations]
    params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=42)
    outs = llm.chat(conversations, params, use_tqdm=True)
    return [o.outputs[0].text.strip() for o in outs]


def guard_prompts(tokenizer, conversations):
    """Render Llama Guard 3 prompts. The 1B template wants list-of-parts content, the 8B plain strings."""
    rendered = []
    for conv in conversations:
        try:
            p = tokenizer.apply_chat_template(conv, tokenize=False)
            if conv[-1]["content"][:20] not in p:
                raise ValueError("content dropped by template")
        except Exception:
            conv = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in conv]
            p = tokenizer.apply_chat_template(conv, tokenize=False)
        rendered.append(p)
    return rendered


def guard_classify(llm, conversations):
    """Classify conversations with Llama Guard 3. Returns list of (is_unsafe, categories)."""
    from vllm import SamplingParams
    tok = llm.get_tokenizer()
    # the template already starts with <|begin_of_text|>; vLLM would add a second BOS
    prompts = [p[len(tok.bos_token):] if tok.bos_token and p.startswith(tok.bos_token) else p
               for p in guard_prompts(tok, conversations)]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=12), use_tqdm=True)
    results = []
    for o in outs:
        text = o.outputs[0].text.strip()
        unsafe = text.lower().startswith("unsafe")
        cats = text.split("\n", 1)[1].strip() if unsafe and "\n" in text else ""
        results.append((unsafe, cats))
    return results
