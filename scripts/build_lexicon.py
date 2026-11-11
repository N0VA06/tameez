"""Merge the bad-word lists into data/lexicons/ and propose a whitelist.

Outputs (data/lexicons/, git-ignored except whitelist.txt):
    lexicon.txt                 merged, normalised entries (Setu hi/en + textdetox hi/hin/en)
    whitelist_auto.txt          single words seen >= --auto-min times in benign text (homonyms)
    whitelist_candidates.tsv    every entry seen in benign text, with counts -> native-speaker review
    whitelist.txt               reviewed whitelist (you edit this; never overwritten)
    lexicon_stats.json          counts only

Usage:
    python -I scripts/build_lexicon.py
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tameez.common import LEXICONS, RAW, get_logger, write_json  # noqa: E402

log = get_logger("build_lexicon")
from tameez.text import tokens  # noqa: E402


def load_sources():
    setu = RAW / "setu_nsfw/setu/data/filter_data/nsfw"
    lists = {
        "setu_hindi": (setu / "hindi.txt").read_text(encoding="utf8").splitlines(),
        "setu_english": (setu / "english.txt").read_text(encoding="utf8").splitlines(),
    }
    for lang in ["hi", "hin", "en"]:
        lists[f"textdetox_{lang}"] = pd.read_parquet(
            RAW / f"textdetox_lexicon/data/{lang}-00000-of-00001.parquet").text.tolist()
    return {k: {" ".join(tokens(x)[0]) for x in v if isinstance(x, str) and x.strip()} - {""} for k, v in lists.items()}


def benign_counts():
    """Word and phrase frequencies over benign Hindi/Hinglish/English text."""
    texts = []
    aya = pd.read_parquet(RAW / "aya_dataset/data/train-00000-of-00001.parquet")
    aya = aya[aya.language == "Hindi"]
    texts += aya.inputs.tolist() + aya.targets.tolist()
    anu = pd.read_parquet(RAW / "indic_align_instruct/indicalign-instruct/anudesh/anudesh1.parquet")
    texts += [s for conv in anu.interactions for turn in conv for s in turn if s]
    dolly = pd.read_parquet(RAW / "indic_align_instruct/indicalign-instruct/dolly/Dolly.parquet", columns=["hin_Deva"])
    texts += [s for conv in dolly.hin_Deva for turn in conv for s in turn if s]
    texts += pd.read_json(RAW / "dolly_15k/databricks-dolly-15k.jsonl", lines=True).response.tolist()
    words = Counter()
    joined = []
    for t in tqdm(texts, desc="tokenise benign text", mininterval=5):
        toks = tokens(t)[0]
        words.update(toks)
        joined.append(f" {' '.join(toks)} ")
    return words, joined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto-min", type=int, default=10, help="benign occurrences needed to auto-whitelist a word")
    args = ap.parse_args()

    sources = load_sources()
    lexicon = set().union(*sources.values())
    words, joined = benign_counts()

    candidates = []
    for entry in tqdm(sorted(lexicon), desc="count lexicon entries in benign text", mininterval=5):
        if " " in entry:
            n = sum(1 for j in joined if f" {entry} " in j)
        else:
            n = words[entry]
        if n:
            candidates.append((entry, n))

    auto = sorted(e for e, n in candidates if " " not in e and n >= args.auto_min)

    LEXICONS.mkdir(parents=True, exist_ok=True)
    (LEXICONS / "lexicon.txt").write_text("\n".join(sorted(lexicon)) + "\n", encoding="utf8")
    (LEXICONS / "whitelist_auto.txt").write_text(
        f"# auto: single words seen >= {args.auto_min} times in benign text\n" + "\n".join(auto) + "\n", encoding="utf8")
    pd.DataFrame(sorted(candidates, key=lambda x: -x[1]), columns=["entry", "benign_count"]).assign(
        keep_as_bad_word="").to_csv(LEXICONS / "whitelist_candidates.tsv", sep="\t", index=False)
    wl = LEXICONS / "whitelist.txt"
    if not wl.exists() or not wl.read_text(encoding="utf8").strip():
        wl.write_text("# Native-speaker reviewed whitelist: one word or phrase per line.\n"
                      "# Fill it from whitelist_candidates.tsv (entries that are NOT bad words in context).\n",
                      encoding="utf8")

    stats = {
        "per_source": {k: len(v) for k, v in sources.items()},
        "lexicon_entries": len(lexicon),
        "single_words": sum(" " not in e for e in lexicon),
        "phrases": sum(" " in e for e in lexicon),
        "seen_in_benign_text": len(candidates),
        "auto_whitelisted": len(auto),
        "auto_min": args.auto_min,
    }
    write_json(stats, LEXICONS / "lexicon_stats.json")
    log.info(f"lexicon stats: {stats}")


if __name__ == "__main__":
    main()
