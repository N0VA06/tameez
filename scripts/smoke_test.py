"""Quick check that a Colab session is ready: data on Drive, GPU, and a reward model.

Prints counts and scores only, never toxic text.

Usage (on the Colab VM, after `colab drivemount`):
    python scripts/smoke_test.py
    python scripts/smoke_test.py --data-dir /content/tameez/data/raw
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

DRIVE_RAW = "/content/drive/MyDrive/tameez/data/raw"

# name -> (relative path, reader)
FILES = {
    "aya_dataset (train)": ("aya_dataset/data/train-00000-of-00001.parquet", pd.read_parquet),
    "indicalign dolly": ("indic_align_instruct/indicalign-instruct/dolly/Dolly.parquet", pd.read_parquet),
    "indicalign oasst": ("indic_align_instruct/indicalign-instruct/oasst/oasst.parquet", pd.read_parquet),
    "indicalign anudesh": ("indic_align_instruct/indicalign-instruct/anudesh/anudesh1.parquet", pd.read_parquet),
    "toxic hh-rlhf": ("indic_align_toxic/indicalign-toxic/hhrlhf/hh-rlhf.parquet", pd.read_parquet),
    "toxic toxicmatrix": ("indic_align_toxic/indicalign-toxic/toxicmatrix/toxic_prompts_sarvam.parquet", pd.read_parquet),
    "textdetox hi": ("textdetox_toxicity/data/hi-00000-of-00001.parquet", pd.read_parquet),
    "textdetox hin": ("textdetox_toxicity/data/hin-00000-of-00001.parquet", pd.read_parquet),
    "lexicon hi": ("textdetox_lexicon/data/hi-00000-of-00001.parquet", pd.read_parquet),
    "aya redteaming hi": ("aya_redteaming/aya_hin.jsonl", lambda p: pd.read_json(p, lines=True)),
    "indicsafe": ("indicsafe/data/Full_translated_prompts.csv", pd.read_csv),
    "aya eval dolly-he": ("aya_evaluation_suite/dolly_human_edited/test-00000-of-00001.parquet", pd.read_parquet),
    "dolly 15k (ref)": ("dolly_15k/databricks-dolly-15k.jsonl", lambda p: pd.read_json(p, lines=True)),
}


def check_data(raw):
    print(f"\n[1/3] Data in {raw}")
    ok = True
    for name, (rel, reader) in FILES.items():
        path = raw / rel
        try:
            t = time.time()
            n = len(reader(path))
            print(f"  ✓ {name:22s} {n:>7,} rows  ({time.time() - t:.1f}s)")
        except Exception as e:
            ok = False
            print(f"  ✗ {name:22s} {type(e).__name__}: {e}")
    setu = raw / "setu_nsfw/setu/data/filter_data/nsfw/hindi.txt"
    print(f"  {'✓' if setu.exists() else '✗'} {'setu nsfw hindi':22s} {'present' if setu.exists() else 'missing'}")
    return ok and setu.exists()


def check_gpu():
    print("\n[2/3] GPU")
    if not torch.cuda.is_available():
        print("  ✗ CUDA not available")
        return False
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    x @ x  # warm-up: first call includes cuBLAS init
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        x @ x
    torch.cuda.synchronize()
    tflops = 10 * 2 * 4096**3 / (time.time() - t) / 1e12
    free, total = torch.cuda.mem_get_info()
    print(f"  ✓ {torch.cuda.get_device_name(0)}  {total / 1e9:.0f} GB total, {free / 1e9:.0f} GB free  ~{tflops:.0f} TFLOPS bf16")
    return True


def check_reward_model(raw):
    """Score held-out textdetox `hin` texts with the training toxicity classifier."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("\n[3/3] Reward model: textdetox/xlmr-large-toxicity-classifier-v2")
    name = "textdetox/xlmr-large-toxicity-classifier-v2"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()
    toxic_idx = 1  # per the model card: idx 1 = toxic
    print(f"  id2label = {model.config.id2label}")

    df = pd.read_parquet(raw / "textdetox_toxicity/data/hin-00000-of-00001.parquet")
    sample = pd.concat([df[df.toxic == 1].sample(50, random_state=0), df[df.toxic == 0].sample(50, random_state=0)])
    with torch.no_grad():
        enc = tok(sample.text.tolist(), padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
        p_toxic = model(**enc).logits.float().softmax(-1)[:, toxic_idx].cpu()
    sample = sample.assign(p=p_toxic.numpy())
    mean_toxic = sample[sample.toxic == 1].p.mean()
    mean_clean = sample[sample.toxic == 0].p.mean()
    acc = ((sample.p > 0.5).astype(int) == sample.toxic).mean()
    print(f"  mean p(toxic): toxic texts {mean_toxic:.2f} | clean texts {mean_clean:.2f} | accuracy@0.5 {acc:.0%} (n=100, Hinglish)")

    benign = ["मुझे दिल्ली से आगरा जाने का सबसे सस्ता तरीका बताइए।", "bhai linux mein process kaise kill karte hain?"]
    with torch.no_grad():
        p = model(**tok(benign, padding=True, return_tensors="pt").to("cuda")).logits.float().softmax(-1)[:, toxic_idx]
    for text, score in zip(benign, p.tolist()):
        print(f"  benign prompt -> p(toxic) {score:.2f}  | {text}")
    return bool(mean_toxic > mean_clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DRIVE_RAW)
    raw = Path(ap.parse_args().data_dir)

    results = {"data": check_data(raw), "gpu": check_gpu()}
    results["reward_model"] = check_reward_model(raw) if results["gpu"] else False
    print("\nRESULT", json.dumps(results), "->", "ALL OK" if all(results.values()) else "SOMETHING FAILED")


if __name__ == "__main__":
    main()
