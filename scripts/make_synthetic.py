"""Generate natural Hinglish rewrites and tricky-seed paraphrases with Aya Expanse 8B (GPU).

IndicAlign's `hin_Latn` is machine transliteration nobody types ("haye", "kii"), so the
Roman-Hinglish slices of sets A, B and C are rewritten from Devanagari here instead.

Tasks (rows chosen by the stable roles in tameez.sources):
    sft_hin      Dolly-T pairs (role A_hin_rewrite): prompt + answer -> Hinglish      -> Set A
    benign_hin   Anudesh Devanagari prompts (role B_hin_rewrite) -> Hinglish          -> Set B
    harmful_hin  harmful Hindi prompts that Llama-Guard-3-1B confirms unsafe IN DEVANAGARI
                 (role B_harmful_hin_rewrite) -> Hinglish                            -> Set B
                 (Guard-1B misses much of the harm once text is Roman Hinglish: 17-24% of
                 Hinglish rewrites were flagged vs 46-53% of Hindi originals, so harmfulness
                 is decided on the Hindi original and carries over to its rewrite)
    redteam_hin  Aya Red-teaming Hindi (role C_hin_rewrite) -> Hinglish               -> Set C only
    seed_para    4 paraphrases of every tricky seed, per language                     -> Sets B/C by seed

Output: data/processed/cache/synthetic.json (JSON Lines). Re-running skips finished items.

Usage (GPU):
    python scripts/make_synthetic.py                     # all tasks
    python scripts/make_synthetic.py --tasks seed_para --limit 20
"""

import argparse
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tameez import labels as guard, llm, sources  # noqa: E402
from tameez.common import CACHE, MODELS, acquire_gpu, get_logger, read_jsonl, timed, write_jsonl  # noqa: E402

log = get_logger("make_synthetic")
from tameez.text import is_refusal, script_of  # noqa: E402

OUT = CACHE / "synthetic.json"
PROMPT_VERSION = 2   # cached rows from an older prompt version are regenerated

# Rewrite = transliterate. The first prompt listed English words to keep and Aya copied them into
# outputs ("phone, office, file") and added greetings/translations; ~40% of accepted rewrites were
# contaminated. Fixed prompt: system rule + two worked examples, greedy decoding, cleanup + stricter checks.
HINGLISH_SYSTEM = ("You convert Hindi written in Devanagari into Hinglish written in Roman (Latin) letters, "
                   "exactly the way Indians type Hindi on WhatsApp, with everyday spellings like hai, nahi, kya, "
                   "mujhe, kaise, bahut. Only change the script: keep the same words, meaning, tone, length, line "
                   "breaks and lists. Never translate into English, never answer or comment on the text, never add "
                   "greetings, notes, translations or explanations. Output only the converted text.")
HINGLISH_SHOTS = [
    ("मुझे कल सुबह जयपुर जाना है, सबसे सस्ता तरीका क्या है?", "Mujhe kal subah Jaipur jaana hai, sabse sasta tareeka kya hai?"),
    ("मेरा लैपटॉप बहुत धीमा चल रहा है। इसे तेज़ करने के लिए क्या करूँ?\n1. पुरानी फ़ाइलें हटाऊँ?\n2. रैम बढ़ाऊँ?",
     "Mera laptop bahut dheema chal raha hai. Ise tez karne ke liye kya karun?\n1. Purani files hataun?\n2. RAM badhaun?"),
]
SHOT_WORDS = {"jaipur", "sasta", "laptop", "dheema", "ram", "purani"}
GREETING = re.compile(r"^\s*(hai|hi|hello|hey|namaste|namaskar)\b[,!.]?", re.I)
NOTE = re.compile(r"\((?:translation|note|english|meaning)[^)]*\)|^\s*(?:translation|note|explanation)\s*:.*$", re.I | re.M)


def hinglish_messages(text):
    msgs = [{"role": "system", "content": HINGLISH_SYSTEM}]
    for src, tgt in HINGLISH_SHOTS:
        msgs += [{"role": "user", "content": src}, {"role": "assistant", "content": tgt}]
    return msgs + [{"role": "user", "content": text}]


PARAPHRASE = """Write 4 different paraphrases of the question below. Each paraphrase must:
- be in {lang} (same language and script as the question)
- ask for exactly the same thing{keep}
Vary the wording and length. Output exactly 4 lines, one paraphrase per line, with no numbering or quotes.

Question: {text}"""

LANG_NAME = {"hi": "Hindi in Devanagari script", "hin": "Hinglish in Roman script (casual WhatsApp style)",
             "en": "English"}
LANG_SCRIPT = {"hi": "deva", "hin": "roman_hinglish", "en": "english"}


HARMFUL_HIN_PER_SOURCE = 350      # Guard-confirmed Hindi originals rewritten per source (HH-RLHF, Toxic-Matrix)


def harmful_hin_rows(labels):
    """Rows with the Hinglish-rewrite role whose Hindi original Guard-1B calls unsafe, balanced by source."""
    harm = sources.harmful_prompts()
    harm = harm[harm.id.map(lambda i: sources.role("harmful", i) == "B_harmful_hin_rewrite")]
    harm = harm[harm.prompt_hi.map(lambda t: guard.is_unsafe(labels, t) is True)]
    return harm.sort_values(["source", "id"]).groupby("source").head(HARMFUL_HIN_PER_SOURCE)


def build_jobs(limit, labels=None):
    """Return list of jobs: dict(task, id, field, src, prompt, check)."""
    jobs = []

    def add(task, row_id, field, src, prompt, check):
        """prompt: a user string (paraphrase) or a full message list (Hinglish rewrite)."""
        msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        jobs.append({"task": task, "id": row_id, "field": field, "src": src, "messages": msgs, "check": check})

    dolly = sources.dolly_t()
    dolly = dolly[dolly.id.map(lambda i: sources.role("dolly", i) == "A_hin_rewrite")]
    dolly = dolly[dolly.answer.str.len() <= 1200]
    for r in dolly.head(limit).itertuples():
        add("sft_hin", r.id, "prompt", r.prompt, hinglish_messages(r.prompt), "hin")
        add("sft_hin", r.id, "answer", r.answer, hinglish_messages(r.answer), "hin")

    anu = sources.anudesh()
    anu = anu[[sources.role("anudesh", r.id, r.script) == "B_hin_rewrite" for r in anu.itertuples()]]
    for r in anu.head(min(limit, 600)).itertuples():
        add("benign_hin", r.id, "prompt", r.prompt, hinglish_messages(r.prompt), "hin")

    harm = harmful_hin_rows(labels or {})
    for r in harm.head(limit).itertuples():
        add("harmful_hin", r.id, "prompt", r.prompt_hi, hinglish_messages(r.prompt_hi), "hin")

    rt = sources.aya_redteaming_hindi()
    for r in rt[rt.id.map(lambda i: sources.role("redteam", i) == "C_hin_rewrite")].head(limit).itertuples():
        add("redteam_hin", r.id, "prompt", r.prompt, hinglish_messages(r.prompt), "hin")

    for seed in sources.tricky_seeds()[:limit]:
        for lang in ("hi", "hin", "en"):
            # only ask to keep a scary word that actually appears in this language version
            term = next((t for t in seed["allowed_terms"] if t.lower() in seed[lang].lower()), None)
            keep = f'\n- keep the word "{term}"' if term else ""
            add("seed_para", seed["id"], lang, seed[lang],
                PARAPHRASE.format(lang=LANG_NAME[lang], keep=keep, text=seed[lang]), f"para_{lang}")
    return jobs


def clean(out):
    return NOTE.sub("", out).strip().strip('"').strip()


def validate(job, out):
    """Return (accepted texts, reason). Empty list = rejected, reason says why."""
    if job["check"] == "hin":
        out = clean(out)
        words = set(re.findall(r"[a-z]+", out.lower()))
        src_has_greeting = bool(re.match(r"\s*(नमस्ते|नमस्कार|हाय|हैलो|hi|hello)", job["src"], re.I))
        ratio = len(out) / max(1, len(job["src"]))
        checks = [("script", script_of(out) == "roman_hinglish"), ("refusal", not is_refusal(out)),
                  ("length", 0.5 < ratio < 2.0), ("greeting", src_has_greeting or not GREETING.match(out)),
                  ("example_leak", not (words & SHOT_WORDS - set(job["src"].lower().split())))]
        failed = [name for name, ok in checks if not ok]
        return ([out], "") if not failed else ([], ",".join(failed))
    lang = job["check"].split("_")[1]
    lines = [re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", l).strip(" \"'") for l in out.splitlines()]
    good = [l for l in lines if 8 <= len(l) <= 300 and script_of(l) == LANG_SCRIPT[lang] and not is_refusal(l)]
    good = list(dict.fromkeys(good))[:4]
    return good, "" if good else "no valid paraphrase"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=["sft_hin", "benign_hin", "harmful_hin", "redteam_hin", "seed_para"])
    ap.add_argument("--limit", type=int, default=100_000, help="max items per task (for quick tests)")
    ap.add_argument("--model", default=MODELS["generator"])
    args = ap.parse_args()

    log.info(f"args: {vars(args)}")
    done = read_jsonl(OUT) if OUT.exists() else []
    stale = [d for d in done if d.get("prompt_version", 1) < PROMPT_VERSION and d["task"] != "seed_para"]
    if stale:
        log.info(f"dropping {len(stale)} cached rewrites made with an older prompt (regenerated)")
        done = [d for d in done if d not in stale]
    done_keys = {(d["task"], d["id"], d["field"]) for d in done}
    labels = {}
    if "harmful_hin" in args.tasks:
        with timed(log, "Guard-1B labels for Hindi originals of harmful Hinglish candidates"):
            harm = sources.harmful_prompts()
            harm = harm[harm.id.map(lambda i: sources.role("harmful", i) == "B_harmful_hin_rewrite")]
            labels = guard.label(harm.prompt_hi.tolist(), log)
    with timed(log, "select rows to rewrite"):
        jobs = [j for j in build_jobs(args.limit, labels) if j["task"] in args.tasks and (j["task"], j["id"], j["field"]) not in done_keys]
    from collections import Counter
    log.info(f"{len(done)} cached, {len(jobs)} to generate: {dict(Counter(j['task'] for j in jobs))}")
    if not jobs:
        return

    acquire_gpu(log)
    with timed(log, "load Aya Expanse"):
        model = llm.load(args.model, max_model_len=4096, gpu_memory_utilization=0.9, log=log)
    with timed(log, f"generate {len(jobs)} rewrites (vLLM progress bar below)"):
        outs = llm.chat(model, [j["messages"] for j in jobs], temperature=0.0, max_tokens=1024)
    for job, out in tqdm(list(zip(jobs, outs)), desc="validate", mininterval=5):
        accepted, reason = validate(job, out)
        done.append({"task": job["task"], "id": job["id"], "field": job["field"], "src": job["src"],
                     "raw": out, "accepted": accepted, "valid": bool(accepted), "reject_reason": reason,
                     "prompt_version": PROMPT_VERSION})
    write_jsonl(done, OUT)
    log.info(f"saved {OUT}")

    total, valid = Counter(d["task"] for d in done), Counter(d["task"] for d in done if d["valid"])
    log.info(f"valid generations per task: { {t: f'{valid[t]}/{total[t]}' for t in total} }")
    reasons = Counter((d["task"], r) for d in done if not d["valid"] for r in d.get("reject_reason", "?").split(","))
    log.info(f"reject reasons: {dict(reasons.most_common(15))}")


if __name__ == "__main__":
    main()
