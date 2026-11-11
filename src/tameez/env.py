"""TameezEnv: the GRPO environment = prompt pool (Set B) + the six rewards.

GRPO needs no answers, only prompts tagged with what good behaviour looks like
(bucket, lang, allowed_terms) and reward functions that read those tags. This class keeps
both together so the same object is used to validate the data, score replies, measure
a model's starting reward (rollout) and feed AlignTune's GRPO trainer.

    env = TameezEnv()                       # loads data/processed/B_pool.json
    env.validate()                          # schema / tag checks -> list of problems
    env.stats()                             # rows per bucket x language
    row = env.sample(1, bucket="tricky_benign")[0]
    env.score("reply text", row)            # {reward: value, ..., "total": weighted sum}
    env.rollout("stages/stage1_sft/model/merged", n=200)   # GPU: generate + score + report
    env.reward_specs()                      # rewards=... for create_rl_trainer
"""

import random
from collections import Counter, defaultdict

from . import rewards as R
from .common import PROCESSED, SEED, SYSTEM_PROMPT, read_jsonl
from .text import is_refusal, lang_tag, script_of

BUCKETS = ("harmful", "abusive_legit", "tricky_benign", "benign")
LANGS = ("hi", "hin", "en")
GOOD_BEHAVIOUR = {
    "harmful": "decline politely in the user's language and offer a safe alternative",
    "abusive_legit": "ignore the abuse, solve the real problem, never repeat the insults",
    "tricky_benign": "answer normally: a scary-looking word is not a harmful request",
    "benign": "answer well in the same language and script",
}


class TameezEnv:
    def __init__(self, pool_path=PROCESSED / "B_pool.json", exclude=(), weights=None):
        self.pool_path = pool_path
        self.rows = read_jsonl(pool_path)
        self.exclude = set(exclude)
        weights = weights or {}
        self.rewards = [(fn, weights.get(fn.__name__, w)) for fn, w in R.REWARDS_V1 if fn.__name__ not in self.exclude]
        self.rng = random.Random(SEED)

    def __len__(self):
        return len(self.rows)

    @property
    def reward_names(self):
        return [fn.__name__ for fn, _ in self.rewards]

    # ------------------------------------------------------------ data

    def stats(self):
        """Rows per bucket x language, plus overall shares."""
        by = Counter((r["bucket"], r["lang"]) for r in self.rows)
        n = len(self.rows)
        return {
            "rows": n,
            "by_bucket": {b: sum(v for (bb, _), v in by.items() if bb == b) for b in BUCKETS},
            "by_lang": {l: sum(v for (_, ll), v in by.items() if ll == l) for l in LANGS},
            "bucket_x_lang": {f"{b}/{l}": by[(b, l)] for b in BUCKETS for l in LANGS},
            "share_benign_side": round(sum(r["bucket"] != "harmful" for r in self.rows) / max(1, n), 3),
        }

    def validate(self):
        """Return a list of problems (empty = the pool is consistent with what the rewards expect)."""
        problems = []
        need = {"prompt", "reference", "bucket", "lang", "allowed_terms", "source", "seed_id"}
        for i, r in enumerate(self.rows):
            missing = need - r.keys()
            if missing:
                problems.append(f"row {i}: missing {sorted(missing)}")
                continue
            if r["bucket"] not in BUCKETS:
                problems.append(f"row {i}: unknown bucket {r['bucket']}")
            if r["lang"] not in LANGS:
                problems.append(f"row {i}: unknown lang {r['lang']}")
            if r["reference"] != ("refuse" if r["bucket"] == "harmful" else "help"):
                problems.append(f"row {i}: reference {r['reference']} does not match bucket {r['bucket']}")
            msgs = r["prompt"]
            if not (isinstance(msgs, list) and msgs and msgs[-1].get("role") == "user" and msgs[-1].get("content")):
                problems.append(f"row {i}: prompt must be a chat message list ending with a user turn")
            elif lang_tag(msgs[-1]["content"]) != r["lang"]:
                problems.append(f"row {i}: lang tag {r['lang']} but text looks like {lang_tag(msgs[-1]['content'])}")
            if not isinstance(r["allowed_terms"], list):
                problems.append(f"row {i}: allowed_terms must be a list")
        texts = [r["prompt"][-1]["content"] for r in self.rows if r.get("prompt")]
        dups = len(texts) - len(set(texts))
        if dups:
            problems.append(f"{dups} duplicate prompts")
        return problems

    def sample(self, n, bucket=None, lang=None, stratified=False):
        """Random rows; stratified=True takes an equal share from every bucket."""
        pool = [r for r in self.rows if (bucket is None or r["bucket"] == bucket) and (lang is None or r["lang"] == lang)]
        if not stratified:
            return self.rng.sample(pool, min(n, len(pool)))
        per = max(1, n // len(BUCKETS))
        out = []
        for b in BUCKETS:
            sub = [r for r in pool if r["bucket"] == b]
            out += self.rng.sample(sub, min(per, len(sub)))
        return out

    # ------------------------------------------------------------ rewards

    def score(self, reply, row):
        """Every reward for one reply to one pool row, plus the weighted total GRPO optimises."""
        return R.score(reply, row, self.rewards)

    def score_batch(self, replies, rows):
        return [self.score(rep, row) for rep, row in zip(replies, rows)]

    def report(self, rows, replies, scores=None):
        """Mean of every reward per bucket and per language, plus refusal and script rates."""
        scores = scores or self.score_batch(replies, rows)
        groups = defaultdict(list)
        for row, rep, sc in zip(rows, replies, scores):
            for key in ("all", f"bucket:{row['bucket']}", f"lang:{row['lang']}"):
                groups[key].append((row, rep, sc))
        out = {}
        for key, items in sorted(groups.items()):
            names = self.reward_names + ["total"]
            out[key] = {"n": len(items),
                        **{nm: round(sum((sc[nm] or 0) for _, _, sc in items) / len(items), 3) for nm in names},
                        "refusal_rate": round(sum(is_refusal(rep) for _, rep, _ in items) / len(items), 3),
                        "script_ok_rate": round(sum(script_of(rep) == {"hi": "deva", "hin": "roman_hinglish", "en": "english"}[row["lang"]]
                                                    for row, rep, _ in items) / len(items), 3),
                        "mean_chars": round(sum(len(rep) for _, rep, _ in items) / len(items), 1)}
        return out

    def rollout(self, model, n=200, max_tokens=384, temperature=0.0, log=None):
        """Generate replies for a stratified sample with vLLM and score them (GPU).

        This is the starting reward a model gets in this environment, before any GRPO.
        Returns (rows, replies, scores, report).
        """
        from . import llm
        rows = self.sample(n, stratified=True)
        engine = llm.load(model, max_model_len=4096, gpu_memory_utilization=0.6, log=log)
        replies = llm.chat(engine, [r["prompt"] for r in rows], temperature=temperature,
                           max_tokens=max_tokens, system_prompt=SYSTEM_PROMPT)
        llm.unload(engine)
        scores = self.score_batch(replies, rows)
        return rows, replies, scores, self.report(rows, replies, scores)

    # ------------------------------------------------------------ trainer hook

    def reward_specs(self, wrap=None):
        """`rewards=` for AlignTune's create_rl_trainer. `wrap(fn)` can decorate each reward (e.g. logging)."""
        return [{"type": "custom", "weight": w, "params": {"function": wrap(fn) if wrap else fn}}
                for fn, w in self.rewards]

    def describe(self):
        return {"pool": str(self.pool_path), "rows": len(self), "rewards": {fn.__name__: w for fn, w in self.rewards},
                "good_behaviour": GOOD_BEHAVIOUR}
