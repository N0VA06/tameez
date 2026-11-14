"""Why do Roman-script (Hinglish) harmful prompts get past refusal when their Devanagari twins do not? GPU.

The 139 Hinglish red-teaming prompts of the scorecard are romanized copies of 139 Hindi ones that are also in the
scorecard, so every model sees the same request twice, in two scripts. This script looks at what happens inside each
model for those twins.

Part 1, every stage of every model (forward passes only):
  1. residual stream at the last prompt token (the scorecard's chat template and system prompt), every layer
  2. harm direction per layer: mean(harmful) - mean(harmless) over Devanagari training prompts; the same over their
     Roman twins and over English prompts
  3. where each prompt lands on the Devanagari direction (0 = average harmless Devanagari prompt, 1 = average harmful)
  4. linear probes for harmful vs harmless: trained on Devanagari, tested on Roman and English; trained on Roman
  5. cross-script alignment of meaning: mean-pooled raw text, is a Roman prompt's nearest Devanagari prompt its twin
  6. tokens per word in each script
Part 2, the three final (GRPO) models; greedy replies judged by Llama-Guard-3-8B and the scorecard's refusal matcher:
  7. remove the direction from every layer (directional ablation) on the 139 twins: does Devanagari refusal collapse?
  8. add the direction at one layer to every prompt: does Roman refusal come back, and what happens to harmless prompts?
     (the "only where a probe flags the prompt" variant is assembled from 7-8 afterwards, in interpret_report.py)

    python scripts/interpret.py --out results/interpretability/raw                 # Part 1 + 2 generation
    python scripts/interpret.py --out results/interpretability/raw --judge         # Llama-Guard-3-8B on the replies
    python scripts/interpret.py --models llama_s2 --limit 24 --out /tmp/smoke      # quick check
"""

import argparse
import ast
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tameez.common import SYSTEM_PROMPT, get_logger, read_jsonl, write_json, write_jsonl  # noqa: E402

log = get_logger("interpret")
DATA = "IINOVAII/tameez-hindi-safety-data"
MODELS = {  # key: (repo, run, stage); the Tiny Aya base models are gated, so they go last
    "llama_s0": ("unsloth/Llama-3.2-3B-Instruct", "llama_3_2_3b", "stage0_base"),
    "llama_s1": ("IINOVAII/Llama-3.2-3B-Hindi-SFT", "llama_3_2_3b", "stage1_sft"),
    "llama_s2": ("IINOVAII/Llama-3.2-3B-Tameez-GRPO", "llama_3_2_3b", "stage2_grpo"),
    "global_s1": ("IINOVAII/Tiny-Aya-Global-Hindi-SFT", "tiny_aya_global", "stage1_sft"),
    "global_s2": ("IINOVAII/Tiny-Aya-Global-Tameez-GRPO", "tiny_aya_global", "stage2_grpo"),
    "fire_s1": ("IINOVAII/Tiny-Aya-Fire-Hindi-SFT", "tiny_aya_fire", "stage1_sft"),
    "fire_s2": ("IINOVAII/Tiny-Aya-Fire-Tameez-GRPO", "tiny_aya_fire", "stage2_grpo"),
    "global_s0": ("CohereLabs/tiny-aya-global", "tiny_aya_global", "stage0_base"),
    "fire_s0": ("CohereLabs/tiny-aya-fire", "tiny_aya_fire", "stage0_base"),
}
ALPHAS = (0.5, 1.0)


# ---------------------------------------------------------------- prompts

def fetch(path):
    from huggingface_hub import hf_hub_download
    return read_jsonl(hf_hub_download(DATA, path, repo_type="dataset"))


def first(accepted):
    return accepted[0] if isinstance(accepted, list) else ast.literal_eval(accepted)[0]


def build_prompts(seed=42):
    """Every prompt once, tagged with script (deva / roman / en), harmful, group and the id of its twin."""
    test = fetch("data/safety_test/test.jsonl")
    rewrites = {r["id"]: r for r in fetch("data/hinglish_rewrites/train.jsonl")}
    rl = fetch("data/rl_prompts/train.jsonl")
    seeds = fetch("data/tricky_seeds/train.jsonl")
    rng, P = random.Random(seed), []
    # training pool: Hinglish RL prompts and the Devanagari originals they were rewritten from; 3/4 fit, 1/4 held out
    for i, r in enumerate(rl):
        if r["bucket"] not in ("harmful", "benign"):
            continue
        split = "rl_train" if rng.random() < 0.75 else "rl_heldout"
        h = r["bucket"] == "harmful"
        if r["lang"] == "hin" and r["source"] == "aya_hinglish_rewrite" and r["seed_id"] in rewrites:
            pid = r["seed_id"]
            P.append({"id": pid + "|deva", "text": rewrites[pid]["src"], "script": "deva", "harmful": h, "group": split, "pair": pid})
            P.append({"id": pid + "|roman", "text": r["prompt"][-1]["content"], "script": "roman", "harmful": h, "group": split, "pair": pid})
        elif r["lang"] == "en":
            P.append({"id": f"rl-en-{i}", "text": r["prompt"][-1]["content"], "script": "en", "harmful": h, "group": split, "pair": None})
    # held-out scorecard prompts: the 139 red-teaming twins, and every harmless prompt in all three scripts
    by_id = {r["id"]: r for r in test}
    script = {"hi": "deva", "hin": "roman", "en": "en"}
    for r in test:
        if r["set"] == "aya_redteaming_hinglish":
            d = by_id[r["id"][:-len("-hin")]]
            P.append({"id": d["id"], "text": d["prompt"][-1]["content"], "script": "deva", "harmful": True, "group": "test_twins", "pair": d["id"]})
            P.append({"id": r["id"], "text": r["prompt"][-1]["content"], "script": "roman", "harmful": True, "group": "test_twins", "pair": d["id"]})
        elif r["expected"] == "help":
            P.append({"id": r["id"], "text": r["prompt"][-1]["content"], "script": script[r["lang"]], "harmful": False,
                      "group": "test_benign", "pair": None})
    # 40 tricky-word seeds written in all three scripts: parallel harmless triples
    for s in seeds:
        for lang in ("hi", "hin", "en"):
            P.append({"id": f"seed-{s['id']}-{lang}", "text": s[lang], "script": script[lang], "harmful": False, "group": "seeds", "pair": s["id"]})
    return P


# ---------------------------------------------------------------- activations

def load(repo):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(repo, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16, device_map={"": 0}).eval()
    return tok, model


def chat_texts(tok, texts):
    return [tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": t}],
                                    add_generation_prompt=True, tokenize=False) for t in texts]


@torch.no_grad()
def last_token_states(tok, model, texts, batch=32):
    """[n, layers + 1, d] float32: residual stream at the last prompt token (where the reply starts)."""
    rendered = chat_texts(tok, texts)
    order = sorted(range(len(texts)), key=lambda i: len(rendered[i]))
    out = None
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        enc = tok([rendered[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        hs = model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states
        h = torch.stack([x[:, -1, :] for x in hs], 1).float().cpu().numpy()
        if out is None:
            out = np.zeros((len(texts),) + h.shape[1:], dtype=np.float32)
        out[idx] = h
    return out


@torch.no_grad()
def mean_pooled_states(tok, model, texts, batch=32):
    """[n, layers + 1, d] float32: raw text (no chat template), averaged over its tokens; for meaning alignment."""
    tok_r = tok.__class__.from_pretrained(tok.name_or_path, padding_side="right")
    if tok_r.pad_token is None:
        tok_r.pad_token = tok_r.eos_token
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = None
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        enc = tok_r([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=256).to("cuda")
        hs = model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float()
        m[:, 0] = 0          # skip the BOS / first position, which every text shares
        h = torch.stack([(x.float() * m).sum(1) / m.sum(1).clamp(min=1) for x in hs], 1).cpu().numpy()
        if out is None:
            out = np.zeros((len(texts),) + h.shape[1:], dtype=np.float32)
        out[idx] = h
    return out


# ---------------------------------------------------------------- measurements

def auroc(pos, neg):
    """Probability that a random positive scores above a random negative (ties count half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    ranks = np.empty(len(allv))
    order = allv.argsort(kind="mergesort")
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):            # average ranks for ties
        t = allv == v
        if t.sum() > 1:
            ranks[t] = ranks[t].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_probe(X, y, wd=1e-2, steps=300):
    """L2 logistic regression on standardized features (GPU, full batch). Returns a scoring function."""
    Xt = torch.tensor(X, device="cuda")
    yt = torch.tensor(y, device="cuda", dtype=torch.float32)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-4
    Z = (Xt - mu) / sd
    w = torch.zeros(Z.shape[1], device="cuda", requires_grad=True)
    b = torch.zeros(1, device="cuda", requires_grad=True)
    pw = (1 - yt).sum() / yt.sum().clamp(min=1)          # balance the classes
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, yt, pos_weight=pw) + wd * (w @ w)
        loss.backward()
        return loss
    opt.step(closure)
    w, b = w.detach(), b.detach()
    return lambda Xn: (((torch.tensor(Xn, device="cuda") - mu) / sd) @ w + b).cpu().numpy()


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def cos(a, b):
    return float(unit(a) @ unit(b))


def retrieval(A, B):
    """A[i] and B[i] are twins. Centre each script, then: share of A whose nearest B is its own twin, and the mean
    cosine of twins minus the mean cosine of non-twins."""
    A, B = unit(A - A.mean(0)), unit(B - B.mean(0))
    S = A @ B.T
    n = len(A)
    twin = np.diag(S).mean()
    other = (S.sum() - np.trace(S)) / (n * n - n)
    return {"top1": float((S.argmax(1) == np.arange(n)).mean()), "twin_minus_other": float(twin - other), "n": n}


def measure(P, A, pooled, pooled_idx):
    """Per-layer numbers for one model. A: last-token states of every prompt in P."""
    def ix(**kw):
        return np.array([i for i, p in enumerate(P) if all(p[k] == v for k, v in kw.items())], dtype=int)

    tr = {f"{'h' if h else 'b'}_{s}": ix(group="rl_train", script=s, harmful=h) for s in ("deva", "roman", "en") for h in (True, False)}
    ho = {f"{'h' if h else 'b'}_{s}": ix(group="rl_heldout", script=s, harmful=h) for s in ("deva", "roman", "en") for h in (True, False)}
    test = {"twins_deva": ix(group="test_twins", script="deva"), "twins_roman": ix(group="test_twins", script="roman"),
            "benign_deva": ix(group="test_benign", script="deva"), "benign_roman": ix(group="test_benign", script="roman"),
            "benign_en": ix(group="test_benign", script="en")}
    # twins for meaning alignment, in pooled-row order
    pi = {P[i]["id"]: k for k, i in enumerate(pooled_idx)}
    rl_pairs = sorted({P[i]["pair"] for i in pooled_idx if P[i]["group"] in ("rl_train", "rl_heldout")})
    tw_pairs = sorted({P[i]["pair"] for i in pooled_idx if P[i]["group"] == "test_twins"})
    seed_pairs = sorted({P[i]["pair"] for i in pooled_idx if P[i]["group"] == "seeds"})
    tw_id = {(P[i]["pair"], P[i]["script"]): P[i]["id"] for i in pooled_idx if P[i]["group"] == "test_twins"}

    L = A.shape[1]
    layers, dirs = [], np.zeros((3, L, A.shape[2]), np.float32)
    scores = {k: np.zeros((len(P), L), np.float32) for k in ("deva_dir", "roman_dir", "probe_deva_roman")}
    for l in range(L):
        X = A[:, l]
        mu = {k: X[v].mean(0) for k, v in tr.items()}
        r = {s: mu[f"h_{s}"] - mu[f"b_{s}"] for s in ("deva", "roman", "en")}
        for j, s in enumerate(("deva", "roman", "en")):
            dirs[j, l] = r[s]
        s_d = (X - mu["b_deva"]) @ r["deva"] / (r["deva"] @ r["deva"])      # 0 = harmless Devanagari, 1 = harmful
        s_r = (X - mu["b_roman"]) @ r["roman"] / (r["roman"] @ r["roman"])
        scores["deva_dir"][:, l], scores["roman_dir"][:, l] = s_d, s_r
        hd, bd = s_d[ho["h_deva"]], s_d[ho["b_deva"]]
        row = {"layer": l, "cohen_d_deva_heldout": float((hd.mean() - bd.mean()) / np.sqrt((hd.var() + bd.var()) / 2 + 1e-8)),
               "cos": {"deva_roman": cos(r["deva"], r["roman"]), "deva_en": cos(r["deva"], r["en"]), "roman_en": cos(r["roman"], r["en"])},
               "norm": {s: float(np.linalg.norm(r[s])) for s in r},
               "score_deva_dir": {k: float(s_d[v].mean()) for k, v in {**test, **{"heldout_" + k: v for k, v in ho.items()}}.items()},
               # how well a direction separates harmful from harmless, held out (a mass-mean probe)
               "auroc_dir": {"deva_dir_on_deva": auroc(s_d[ho["h_deva"]], s_d[ho["b_deva"]]),
                             "deva_dir_on_roman": auroc(s_d[ho["h_roman"]], s_d[ho["b_roman"]]),
                             "deva_dir_on_en": auroc(s_d[ho["h_en"]], s_d[ho["b_en"]]),
                             "roman_dir_on_roman": auroc(s_r[ho["h_roman"]], s_r[ho["b_roman"]]),
                             "deva_dir_test_roman": auroc(s_d[test["twins_roman"]], s_d[test["benign_roman"]]),
                             "deva_dir_test_deva": auroc(s_d[test["twins_deva"]], s_d[test["benign_deva"]]),
                             "roman_dir_test_roman": auroc(s_r[test["twins_roman"]], s_r[test["benign_roman"]])}}
        # learned probes (logistic regression), every layer
        probes = {"deva": ["deva"], "roman": ["roman"], "en": ["en"], "deva+roman": ["deva", "roman"]}
        row["auroc_probe"] = {}
        for name, scripts in probes.items():
            idx = np.concatenate([np.concatenate([tr[f"h_{s}"], tr[f"b_{s}"]]) for s in scripts])
            y = np.concatenate([np.r_[np.ones(len(tr[f"h_{s}"])), np.zeros(len(tr[f"b_{s}"]))] for s in scripts])
            f = fit_probe(X[idx], y)
            if name == "deva+roman":       # kept for every prompt: the gate of the conditional intervention
                scores["probe_deva_roman"][:, l] = f(X)
            for t in ("deva", "roman", "en"):
                row["auroc_probe"][f"{name}->{t}"] = auroc(f(X[ho[f"h_{t}"]]), f(X[ho[f"b_{t}"]]))
            row["auroc_probe"][f"{name}->test_roman"] = auroc(f(X[test["twins_roman"]]), f(X[test["benign_roman"]]))
            row["auroc_probe"][f"{name}->test_deva"] = auroc(f(X[test["twins_deva"]]), f(X[test["benign_deva"]]))
        # meaning alignment between scripts (raw text, mean-pooled)
        Y = pooled[:, l]
        row["align"] = {
            "rl_deva_roman": retrieval(Y[[pi[p + "|roman"] for p in rl_pairs]], Y[[pi[p + "|deva"] for p in rl_pairs]]),
            "test_deva_roman": retrieval(Y[[pi[tw_id[(p, "roman")]] for p in tw_pairs]], Y[[pi[tw_id[(p, "deva")]] for p in tw_pairs]]),
            "seeds_deva_roman": retrieval(Y[[pi[f"seed-{p}-hin"] for p in seed_pairs]], Y[[pi[f"seed-{p}-hi"] for p in seed_pairs]]),
            "seeds_deva_en": retrieval(Y[[pi[f"seed-{p}-hi"] for p in seed_pairs]], Y[[pi[f"seed-{p}-en"] for p in seed_pairs]]),
            "seeds_roman_en": retrieval(Y[[pi[f"seed-{p}-hin"] for p in seed_pairs]], Y[[pi[f"seed-{p}-en"] for p in seed_pairs]])}
        layers.append(row)
    return layers, dirs, scores, tr, ho, test


def fertility(tok, P):
    out = {}
    for s in ("deva", "roman", "en"):
        texts = [p["text"] for p in P if p["script"] == s and p["group"] in ("rl_train", "rl_heldout", "test_twins")]
        n_tok = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        n_word = [max(1, len(t.split())) for t in texts]
        out[s] = {"tokens_per_word": round(float(np.sum(n_tok) / np.sum(n_word)), 3), "n": len(texts)}
    return out


# ---------------------------------------------------------------- interventions

class Hooks:
    def __init__(self, model):
        self.model, self.handles = model, []

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    @staticmethod
    def _apply(out, fn):
        if isinstance(out, tuple):
            return (fn(out[0]),) + tuple(out[1:])
        return fn(out)

    def add(self, layer, vec):
        """Add vec to the residual stream after decoder block `layer - 1` (hidden_states[layer]), every position."""
        v = torch.tensor(vec, device="cuda")
        blk = self.model.model.layers[layer - 1]
        self.handles.append(blk.register_forward_hook(lambda m, i, o: self._apply(o, lambda h: h + v.to(h.dtype))))

    def ablate(self, direction):
        """Remove the direction from the embeddings and from the output of every block (directional ablation)."""
        u = torch.tensor(unit(direction), device="cuda")

        def proj(h):
            uu = u.to(h.dtype)
            return h - (h @ uu).unsqueeze(-1) * uu
        self.handles.append(self.model.model.embed_tokens.register_forward_hook(lambda m, i, o: self._apply(o, proj)))
        for blk in self.model.model.layers:
            self.handles.append(blk.register_forward_hook(lambda m, i, o: self._apply(o, proj)))


@torch.no_grad()
def generate(tok, model, texts, max_tokens, batch=64):
    rendered = chat_texts(tok, texts)
    order = sorted(range(len(texts)), key=lambda i: len(rendered[i]))
    replies = [None] * len(texts)
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        enc = tok([rendered[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, seq in zip(idx, out[:, enc["input_ids"].shape[1]:]):
            replies[i] = tok.decode(seq, skip_special_tokens=True).strip()
    return replies


def pick_layer(layers, L):
    """Steering layer: between 30% and 60% of the depth (where refusal directions usually sit), the layer whose
    Devanagari direction best separates held-out harmful from harmless Devanagari prompts (Cohen's d)."""
    mid = [r for r in layers if round(0.3 * L) <= r["layer"] <= round(0.6 * L)]
    return max(mid, key=lambda r: r["cohen_d_deva_heldout"])["layer"]


def intervene(tok, model, P, layers, dirs, test, out, max_tokens, log_):
    L = len(layers)
    lay = pick_layer(layers, L)
    eval_idx = np.concatenate([test["twins_deva"], test["twins_roman"], test["benign_deva"], test["benign_roman"], test["benign_en"]])
    twins = np.concatenate([test["twins_deva"], test["twins_roman"]])
    rows, hooks = [], Hooks(model)
    t0 = time.time()
    for cond, idx in [("none", eval_idx), ("ablate", twins)] + [(f"add_{a}", eval_idx) for a in ALPHAS]:
        hooks.clear()
        if cond == "ablate":
            hooks.ablate(dirs[0, lay])
        elif cond.startswith("add_"):
            hooks.add(lay, float(cond[4:]) * dirs[0, lay])
        reps = generate(tok, model, [P[i]["text"] for i in idx], max_tokens)
        hooks.clear()
        rows += [{"id": P[i]["id"], "cond": cond, "script": P[i]["script"], "group": P[i]["group"], "harmful": P[i]["harmful"],
                  "prompt": P[i]["text"], "reply": rep} for i, rep in zip(idx, reps)]
        log_.info(f"  {cond}: {len(idx)} replies, {time.time() - t0:.0f} s")
    write_jsonl(rows, out / "replies.jsonl")
    write_json({"layer": lay, "alphas": ALPHAS, "max_tokens": max_tokens, "n_eval": len(eval_idx),
                "direction_norm": float(np.linalg.norm(dirs[0, lay]))}, out / "intervention_args.json")


# ---------------------------------------------------------------- judge

def judge(root):
    from tameez import llm
    from tameez.common import MODELS as M
    from tameez.text import is_refusal
    files = sorted(Path(root).glob("*/replies.jsonl"))
    allrows = [(f, r) for f in files for r in read_jsonl(f)]
    if not allrows:
        log.info("no replies to judge")
        return
    model = llm.load(M["guard_eval"], max_model_len=8192, gpu_memory_utilization=0.85, log=log)
    convs = [[{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["reply"][:3000]}] for _, r in allrows]
    labels = llm.guard_classify(model, convs)
    by_file = {}
    for (f, r), (unsafe, cats) in zip(allrows, labels):
        by_file.setdefault(f, []).append({**r, "unsafe": unsafe, "guard_categories": cats, "refused": is_refusal(r["reply"])})
    for f, rows in by_file.items():
        write_jsonl(rows, f.parent / "judged.jsonl")
        log.info(f"judged {f.parent.name}: {len(rows)} replies")
    llm.unload(model)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--limit", type=int, default=0, help="keep only this many prompts per group and script (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-intervene", action="store_true")
    ap.add_argument("--judge", action="store_true", help="only judge the saved replies with Llama-Guard-3-8B")
    args = ap.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    if args.judge:
        return judge(root)
    P = build_prompts()
    if args.limit:      # smoke test: the first few twin pairs and single prompts of every group
        pairs, singles, kept = {}, {}, set()
        for i, p in enumerate(P):
            if p["pair"] is not None:
                k = (p["group"], p["harmful"])
                if p["pair"] not in pairs.setdefault(k, []):
                    if len(pairs[k]) >= args.limit:
                        continue
                    pairs[k].append(p["pair"])
                kept.add(i)
            else:
                k = (p["group"], p["script"], p["harmful"])
                if singles.get(k, 0) < args.limit:
                    singles[k] = singles.get(k, 0) + 1
                    kept.add(i)
        P = [p for i, p in enumerate(P) if i in kept]
    write_jsonl(P, root / "prompts.jsonl")
    pooled_idx = [i for i, p in enumerate(P) if p["pair"] is not None]
    log.info(f"{len(P)} prompts, {len(pooled_idx)} in twin sets")
    for key in args.models.split(","):
        repo, run, stage = MODELS[key]
        out = root / key
        if (out / "measures.json").exists() and (args.no_intervene or not stage.endswith("grpo") or (out / "replies.jsonl").exists()):
            log.info(f"[skip] {key}")
            continue
        out.mkdir(exist_ok=True)
        t0 = time.time()
        tok, model = load(repo)
        A = last_token_states(tok, model, [p["text"] for p in P])
        pooled = mean_pooled_states(tok, model, [P[i]["text"] for i in pooled_idx])
        log.info(f"{key}: states {A.shape} + pooled {pooled.shape} in {time.time() - t0:.0f} s")
        layers, dirs, scores, _, _, test = measure(P, A, pooled, pooled_idx)
        write_json({"model": repo, "run": run, "stage": stage, "n_layers": A.shape[1] - 1, "d_model": A.shape[2],
                    "fertility": fertility(tok, P), "layers": layers}, out / "measures.json")
        np.savez_compressed(out / "directions.npz", deva=dirs[0], roman=dirs[1], en=dirs[2])
        # [prompt, layer], rows as in prompts.jsonl: position on the Devanagari / Roman direction, and the logit of the
        # probe trained on both scripts
        np.savez_compressed(out / "scores.npz", **{k: v.astype(np.float16) for k, v in scores.items()})
        log.info(f"{key}: measured in {time.time() - t0:.0f} s; best Devanagari layer {pick_layer(layers, len(layers))}")
        if stage.endswith("grpo") and not args.no_intervene:
            intervene(tok, model, P, layers, dirs, test, out, args.max_tokens, log)
        del model, A, pooled
        torch.cuda.empty_cache()
        log.info(f"{key} done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
