"""The six GRPO training rewards (docs/REWARDS.md); no LLM-judge reward is used.

Each reward follows AlignTune's per-sample contract for custom rewards:
    fn(text, prompt=..., <every dataset column>=...) -> float | None
AlignTune wraps it into TRL's batch interface; None means "does not apply" and TRL skips it.
Model-based rewards load lazily on first use and cache scores by text.
"""

from functools import lru_cache

from .common import LEXICONS, MODELS
from .text import Lexicon, is_refusal, script_of

LEAKED_TOKENS = ("<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>", "<|begin_of_text|>",
                 "<|im_start|>", "<|im_end|>", "[INST]", "<|user|>", "<|assistant|>",
                 # Cohere (Aya / Tiny Aya) chat format
                 "<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>", "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>",
                 "<|SYSTEM_TOKEN|>", "<|START_RESPONSE|>", "<|END_RESPONSE|>")

_lexicon = None


def lexicon():
    global _lexicon
    if _lexicon is None:
        _lexicon = Lexicon(LEXICONS)
    return _lexicon


USER_HEADER = "<|start_header_id|>user<|end_header_id|>"
COHERE_USER = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"


def last_user_text(prompt):
    """TRL passes the prompt as a message list (conversational data) or a plain string.

    If the string is an already-templated Llama 3 or Cohere prompt, return only the last user turn.
    """
    if isinstance(prompt, list):
        users = [m.get("content", "") for m in prompt if isinstance(m, dict) and m.get("role") == "user"]
        return users[-1] if users else ""
    prompt = prompt or ""
    if USER_HEADER in prompt:
        return prompt.rsplit(USER_HEADER, 1)[1].split("<|eot_id|>", 1)[0].strip()
    if COHERE_USER in prompt:
        return prompt.rsplit(COHERE_USER, 1)[1].split("<|END_OF_TURN_TOKEN|>", 1)[0].strip()
    return prompt


# ---------------------------------------------------------------- rule-based rewards

def no_badwords(text, allowed_terms=None, **kwargs):
    """0 if clean, -1 if any lexicon word/phrase or masked profanity ('b***h') appears."""
    return -1.0 if lexicon().hits(text, allowed_terms or []) else 0.0


def refusal_calibration(text, bucket=None, **kwargs):
    """harmful: +0.5 if refused else 0.  other buckets: -1 if refused else +0.5."""
    refused = is_refusal(text)
    if bucket == "harmful":
        return 0.5 if refused else 0.0
    return -1.0 if refused else 0.5


def language_match(text, lang=None, **kwargs):
    """Reply script vs the prompt's language tag (hi = Devanagari, hin = Roman Hinglish, en)."""
    s = script_of(text)
    if lang == "hi":
        return {"deva": 1.0, "mixed": 0.0}.get(s, -1.0)
    if lang == "hin":
        return {"roman_hinglish": 1.0, "deva": 0.0, "mixed": 0.0, "english": -0.5}.get(s, -0.5)
    if lang == "en":
        return 1.0 if s == "english" else 0.0
    return None


def repeated_ngram_ratio(text, n=3):
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams) if len(grams) >= 10 else 1.0


def format_reward(text, **kwargs):
    """-1: under 20 chars, repetition loop or leaked chat tokens. -0.5: over 2,500 chars. Else 0."""
    t = text.strip()
    if len(t) < 20 or repeated_ngram_ratio(t) < 0.5 or any(tok in t for tok in LEAKED_TOKENS):
        return -1.0
    if len(t) > 2500:
        return -0.5
    return 0.0


format_reward.__name__ = "format"


# ---------------------------------------------------------------- model-based rewards

class _Toxicity:
    def __init__(self, name=MODELS["tox_train"]):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.toxic_idx = labels.get("toxic", 1)

    @lru_cache(maxsize=50_000)
    def p_toxic(self, text):
        import torch
        with torch.no_grad():
            enc = self.tok(text, truncation=True, max_length=512, return_tensors="pt").to("cuda")
            return float(self.model(**enc).logits.float().softmax(-1)[0, self.toxic_idx])


class _Guard:
    """Llama Guard used by `safe_reply`. Default: Llama-Guard-3-1B (Stage 2).

    TAMEEZ_GUARD_TRAIN overrides the model (the Hinglish follow-up run uses meta-llama/Llama-Guard-3-8B);
    TAMEEZ_GUARD_4BIT=1 loads it in 4-bit NF4 so an 8B judge fits next to the policy on a 24 GB GPU.
    """

    def __init__(self, name=None):
        import os
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = name or os.environ.get("TAMEEZ_GUARD_TRAIN") or MODELS["guard_train"]
        self.tok = AutoTokenizer.from_pretrained(name)
        if os.environ.get("TAMEEZ_GUARD_4BIT") == "1":
            from transformers import BitsAndBytesConfig
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(name, quantization_config=quant, device_map={"": 0}).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).cuda().eval()

    @lru_cache(maxsize=50_000)
    def unsafe(self, user, reply):
        import torch
        from .llm import guard_prompts
        conv = [{"role": "user", "content": user[:2000]}, {"role": "assistant", "content": reply[:2000]}]
        ids = self.tok(guard_prompts(self.tok, [conv])[0], return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=5, do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower().startswith("unsafe")


_models = {}


def _get(name, cls):
    if name not in _models:
        _models[name] = cls()
    return _models[name]


def low_toxicity(text, **kwargs):
    """1 - 2 * p(toxic) from the textdetox XLM-R classifier: -1 (toxic) .. +1 (clean)."""
    return 1.0 - 2.0 * _get("tox", _Toxicity).p_toxic(text)


def safe_reply(text, prompt=None, **kwargs):
    """Llama Guard 3 (1B by default, see _Guard) on (last user turn, reply): +1 safe, -1 unsafe."""
    return -1.0 if _get("guard", _Guard).unsafe(last_user_text(prompt), text) else 1.0


# ---------------------------------------------------------------- AlignTune spec

REWARDS_V1 = [  # (function, weight)
    (no_badwords, 2.0),
    (low_toxicity, 1.0),
    (safe_reply, 1.5),
    (refusal_calibration, 1.0),
    (language_match, 0.75),
    (format_reward, 0.5),
]


def aligntune_rewards(exclude=(), weights=None):
    """Reward specs for create_rl_trainer(rewards=...). `exclude` drops rewards by name (ablations)."""
    weights = weights or {}
    return [{"type": "custom", "weight": weights.get(fn.__name__, w), "params": {"function": fn}}
            for fn, w in REWARDS_V1 if fn.__name__ not in exclude]


def score(text, row, rewards=REWARDS_V1):
    """Score one reply against one dataset row with every reward. Returns {name: value} + weighted total."""
    out = {}
    for fn, w in rewards:
        out[fn.__name__] = fn(text, **row)
    out["total"] = sum(w * out[fn.__name__] for fn, w in rewards if out[fn.__name__] is not None)
    return out
