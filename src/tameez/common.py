"""Paths, model ids and small I/O helpers shared by every Tameez script."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CACHE = PROCESSED / "cache"          # intermediate files (Guard labels, Hinglish rewrites, ...)
LEXICONS = ROOT / "data" / "lexicons"
RESULTS = ROOT / "results"          # cross-stage outputs: scorecard, logs, reward check
STAGES = ROOT / "stages"            # one folder per stage: model/, train/, eval/, ...

# Stage folders (docs/PIPELINE.md "Folder per stage")
STAGE0 = STAGES / "stage0_base"     # base model: eval/ (+ env/)
STAGE1 = STAGES / "stage1_sft"      # Hindi SFT: model/, train/, eval/, env/
STAGE2 = STAGES / "stage2_grpo"     # Tameez GRPO: pilot/, model/, train/, checkpoints/, eval/, judge/
ABLATIONS = STAGES / "ablations"    # one folder per ablation run

SYSTEM_PROMPT = "You are a helpful assistant."
SEED = 42

# Llama 3.2 / Llama Guard 1B are byte-identical mirrors of the gated meta-llama repos
# (SHA-256 of every safetensors file checked against the official repo on 2026-11-04).
# The policy can be swapped without code changes: TAMEEZ_POLICY=CohereLabs/tiny-aya-global (Cohere rerun).
MODELS = {
    "policy": os.environ.get("TAMEEZ_POLICY", "unsloth/Llama-3.2-3B-Instruct"),
    "guard_train": "alpindale/Llama-Guard-3-1B",
    "guard_eval": "meta-llama/Llama-Guard-3-8B",
    "tox_train": "textdetox/xlmr-large-toxicity-classifier-v2",
    "tox_eval": "Hate-speech-CNERG/indic-abusive-allInOne-MuRIL",
    "generator": "CohereLabs/aya-expanse-8b",   # Hinglish rewrites + helpfulness judge
}


def read_jsonl(path):
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows, path):
    """AlignTune's local loader only accepts .json/.csv, so JSON Lines are saved as .json."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf8")
    return path


def user_messages(text):
    return [{"role": "user", "content": text}]


# ---------------------------------------------------------------- logging + GPU lock

import fcntl  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

LOGS = RESULTS / "logs"
GPU_LOCK = Path("/tmp/tameez_gpu.lock")
_lock_fd = None


def get_logger(name):
    """Logger that prints to the console and appends to results/logs/<name>.log with timestamps."""
    log = logging.getLogger(name)
    if not log.handlers:
        LOGS.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOGS / f"{name}.log", encoding="utf8")):
            h.setFormatter(fmt)
            log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


def gpu_status():
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        return f"GPU {torch.cuda.get_device_name(0)}: {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB"
    except Exception as e:  # no GPU / torch
        return f"GPU unavailable ({type(e).__name__})"


def acquire_gpu(log):
    """Hold an exclusive GPU lock for this process so two GPU steps never run at once.

    Child processes started by the holder (e.g. evaluate.py run by score_checkpoints.py)
    inherit TAMEEZ_GPU_LOCK_HELD and skip the lock instead of deadlocking.
    """
    global _lock_fd
    if os.environ.get("TAMEEZ_GPU_LOCK_HELD"):
        log.info("GPU lock held by parent process")
        return
    _lock_fd = open(GPU_LOCK, "a+")
    waited = 0
    while True:
        try:
            fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if waited % 60 == 0:
                _lock_fd.seek(0)
                log.info(f"waiting for the GPU: another step holds it ({_lock_fd.read().strip() or 'unknown'}), waited {waited}s")
            time.sleep(5)
            waited += 5
    _lock_fd.seek(0)
    _lock_fd.truncate()
    _lock_fd.write(f"pid {os.getpid()} {' '.join(sys.argv)}")
    _lock_fd.flush()
    os.environ["TAMEEZ_GPU_LOCK_HELD"] = "1"
    log.info(f"GPU lock acquired | {gpu_status()}")


class timed:
    """`with timed(log, "step"):` logs start, end and duration of a step."""

    def __init__(self, log, what):
        self.log, self.what = log, what

    def __enter__(self):
        self.t0 = time.time()
        self.log.info(f">> {self.what}")
        return self

    def __exit__(self, exc_type, exc, tb):
        status = "FAILED" if exc_type else "done"
        self.log.info(f"<< {self.what}: {status} in {time.time() - self.t0:.0f}s")
        return False
