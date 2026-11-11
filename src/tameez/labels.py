"""Llama-Guard-3-1B prompt labels, cached in data/processed/cache/guard_prompt_labels.json.

Shared by make_synthetic.py (which Hindi harmful prompts to rewrite into Hinglish) and
build_data.py (which candidates enter the RL pool). Keyed by tameez.sources.text_key.
"""

import json

from .common import CACHE, MODELS, acquire_gpu, user_messages, write_json
from .sources import text_key

PATH = CACHE / "guard_prompt_labels.json"


def load():
    return json.loads(PATH.read_text()) if PATH.exists() else {}


def label(texts, log, run_guard=True):
    """Return {text_key: [is_unsafe, categories]} for all texts, labelling uncached ones on the GPU."""
    labels = load()
    todo = sorted({t for t in texts if t and text_key(t) not in labels})
    if todo and run_guard:
        from . import llm
        log.info(f"Llama Guard 3 1B: labelling {len(todo)} prompts ({len(set(texts)) - len(todo)} cached)")
        acquire_gpu(log)
        model = llm.load(MODELS["guard_train"], max_model_len=4096, gpu_memory_utilization=0.5, log=log)
        for t, (unsafe, cats) in zip(todo, llm.guard_classify(model, [user_messages(t) for t in todo])):
            labels[text_key(t)] = [unsafe, cats]
        llm.unload(model)
        write_json(labels, PATH)
    elif todo:
        log.warning(f"{len(todo)} prompts have no Guard label (needs a GPU); they are kept unfiltered")
    return labels


def is_unsafe(labels, text):
    """True / False, or None if the text was never labelled."""
    return labels.get(text_key(text), [None, ""])[0]
