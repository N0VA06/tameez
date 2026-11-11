"""Download the raw datasets used by Tameez into data/raw/.

Only the files the plan needs are fetched (ai4bharat/indic-align alone is 28 GB).
Gated repos are attempted and skipped with a warning if access is not granted.

Usage:
    python scripts/download_data.py            # everything
    python scripts/download_data.py --only aya_dataset indicsafe
"""

import argparse
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# name -> (hf repo id, [files]); files are paths inside the dataset repo
HF_SOURCES = {
    # Set A: Hindi SFT
    "aya_dataset": ("CohereLabs/aya_dataset", [
        "data/train-00000-of-00001.parquet",
        "data/test-00000-of-00001.parquet",
    ]),
    "indic_align_instruct": ("ai4bharat/indic-align", [
        "indicalign-instruct/dolly/Dolly.parquet",
        "indicalign-instruct/oasst/oasst.parquet",
        "indicalign-instruct/anudesh/anudesh1.parquet",
    ]),
    # Set B: alignment pool
    "indic_align_toxic": ("ai4bharat/indic-align", [
        "indicalign-toxic/hhrlhf/hh-rlhf.parquet",
        "indicalign-toxic/toxicmatrix/toxic_prompts_sarvam.parquet",
    ]),
    "textdetox_toxicity": ("textdetox/multilingual_toxicity_dataset", [
        "data/hi-00000-of-00001.parquet",
        "data/hin-00000-of-00001.parquet",
        "data/en-00000-of-00001.parquet",
    ]),
    "textdetox_lexicon": ("textdetox/multilingual_toxic_lexicon", [
        "data/hi-00000-of-00001.parquet",
        "data/hin-00000-of-00001.parquet",
        "data/en-00000-of-00001.parquet",
    ]),
    # Set C: safety test
    "aya_redteaming": ("CohereLabs/aya_redteaming", [
        "aya_hin.jsonl",
        "aya_eng.jsonl",
    ]),
    "suraksha_eval": ("SurakshaAI/SurakshaEval", None),  # gated; whole repo is small
    # Set D: skill test
    "milu": ("ai4bharat/MILU", None),  # gated; whole repo is small
    "aya_evaluation_suite": ("CohereLabs/aya_evaluation_suite", [
        "dolly_human_edited/test-00000-of-00001.parquet",
    ]),
    # Leakage check only: maps aya_evaluation_suite source_id -> English Dolly prompt
    "dolly_15k": ("databricks/databricks-dolly-15k", [
        "databricks-dolly-15k.jsonl",
    ]),
}

# name -> (git url, sparse paths or None for full shallow clone)
GIT_SOURCES = {
    "setu_nsfw": ("https://github.com/AI4Bharat/setu.git",
                  ["setu/data/filter_data/nsfw"]),
    "indicsafe": ("https://github.com/ppattnayak/indicsafe.git", None),
}


def fetch_hf(name, repo, files):
    out = RAW / name
    out.mkdir(parents=True, exist_ok=True)
    try:
        if files is None:
            from huggingface_hub import snapshot_download
            snapshot_download(repo, repo_type="dataset", local_dir=out)
        else:
            for f in files:
                hf_hub_download(repo, f, repo_type="dataset", local_dir=out)
        print(f"[ok]   {name} <- {repo}")
    except (GatedRepoError, HfHubHTTPError) as e:
        print(f"[skip] {name} <- {repo}: {type(e).__name__} (request access on the Hub)")


def fetch_git(name, url, sparse):
    out = RAW / name
    if out.exists():
        print(f"[ok]   {name} already present")
        return
    if sparse:
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                        "--sparse", url, str(out)], check=True)
        subprocess.run(["git", "-C", str(out), "sparse-checkout", "set", *sparse], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", url, str(out)], check=True)
    print(f"[ok]   {name} <- {url}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="subset of source names")
    args = ap.parse_args()
    wanted = set(args.only) if args.only else None

    for name, (repo, files) in HF_SOURCES.items():
        if wanted is None or name in wanted:
            fetch_hf(name, repo, files)
    for name, (url, sparse) in GIT_SOURCES.items():
        if wanted is None or name in wanted:
            fetch_git(name, url, sparse)


if __name__ == "__main__":
    main()
