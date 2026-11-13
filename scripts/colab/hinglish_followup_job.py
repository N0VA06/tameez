# Hinglish follow-up run, driven from ONE kernel cell so the kernel stays busy (Colab prunes idle sessions).
# Prints "PUSHABLE <name> <tgz>" whenever a result is ready; the local watcher downloads and pushes it at once.
import glob, json, os, subprocess, time

R = "/content/tameez_aya"
TRAIN_PY = "/content/trainenv/bin/python"
START = "IINOVAII/Tiny-Aya-Global-Tameez-GRPO"
GUARD8B = "meta-llama/Llama-Guard-3-8B"
os.makedirs(f"{R}/results/logs", exist_ok=True)
peak = {"mib": 0}


def stamp():
    return time.strftime("%H:%M:%S", time.gmtime())


def note(msg):
    line = f"{stamp()} {msg}"
    print(line, flush=True)
    with open("/content/hinglish_job.log", "a") as f:
        f.write(line + "\n")


def gpu_mib():
    try:
        out = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                             shell=True, capture_output=True, text=True).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return 0


def run(cmd, log, env=None, every=60, tick=None):
    full_env = {**os.environ, **(env or {})}
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=R, env=full_env)
        t0, last = time.time(), 0
        while p.poll() is None:
            time.sleep(10)
            peak["mib"] = max(peak["mib"], gpu_mib())
            if tick:
                tick()
            if time.time() - last >= every:
                last = time.time()
                print(f"{stamp()} heartbeat {os.path.basename(log)} {int(time.time() - t0) // 60} min, "
                      f"GPU peak {peak['mib']} MiB", flush=True)
    return p.returncode


def tail(path, n=15):
    try:
        return "".join(open(path, errors="replace").readlines()[-n:])
    except Exception:
        return ""


def pushable(name, paths, exclude=()):
    tgz = f"/content/push_{name}.tgz"
    ex = " ".join(f"--exclude='{e}'" for e in exclude)
    subprocess.run(f"cd {R} && tar czf {tgz} {ex} {' '.join(paths)}", shell=True)
    print(f"{stamp()} PUSHABLE {name} {tgz}", flush=True)


def main():
    # ---- 0. environments: system (vLLM eval stack) + /content/trainenv (AlignTune 0.1.35, TRL 1.7.1)
    ready = subprocess.run(f"{TRAIN_PY} -c 'import aligntune, trl, bitsandbytes' && grep -q 'VERSIONS 0.33.0 5.20.2' /content/setup.log",
                           shell=True, capture_output=True).returncode == 0
    if ready:
        note("SETUP ok (already installed on this VM) | " + [l for l in tail("/content/trainenv.log", 5).splitlines() if "TRAINENV" in l][0])
    elif not setup():
        return
    after_setup()


def setup():
    note("SETUP start")
    if run("bash /content/setup_vm.sh", "/content/setup.log") != 0 or "VERSIONS 0.33.0 5.20.2" not in tail("/content/setup.log", 40):
        note("FAILED setup (system env): " + tail("/content/setup.log", 5).replace("\n", " | "))
        return False
    rc = run("pip install -q uv && uv venv /content/trainenv --python 3.13 --seed -q && "
             f"uv pip install -q --python {TRAIN_PY} 'aligntune==0.1.35' peft bitsandbytes accelerate tensorboard && "
             f"{TRAIN_PY} -c \"import torch, trl, transformers, peft, bitsandbytes, aligntune; "
             "print('TRAINENV', torch.__version__, 'trl', trl.__version__, 'transformers', transformers.__version__, "
             "'peft', peft.__version__, 'bnb', bitsandbytes.__version__, 'cuda', torch.cuda.is_available())\"",
             "/content/trainenv.log")
    if rc != 0 or "TRAINENV" not in tail("/content/trainenv.log", 5):
        note("FAILED setup (train env): " + tail("/content/trainenv.log", 6).replace("\n", " | "))
        return False
    note("SETUP ok | " + [l for l in tail("/content/trainenv.log", 5).splitlines() if "TRAINENV" in l][0])
    return True


def after_setup():
    # ---- 1. wait for the HF login (gated Llama-Guard-3-8B); the user logs in over colab ssh
    note("WAITING_FOR_HF_LOGIN")
    t0 = time.time()
    check = (f"python -c \"from huggingface_hub import hf_hub_download, whoami; "
             f"hf_hub_download('{GUARD8B}', 'config.json'); print('LOGIN_OK', whoami()['name'])\"")
    while True:
        out = subprocess.run(check, shell=True, capture_output=True, text=True)
        if "LOGIN_OK" in out.stdout:
            note("HF_LOGIN ok as " + out.stdout.split("LOGIN_OK", 1)[1].strip())
            break
        if time.time() - t0 > 60 * 60:
            note("FAILED no HF login with access to Llama-Guard-3-8B after 60 min")
            return
        print(f"{stamp()} heartbeat waiting for HF login {int(time.time() - t0) // 60} min", flush=True)
        time.sleep(30)

    env8b = {"TAMEEZ_GUARD_TRAIN": GUARD8B, "TAMEEZ_GUARD_4BIT": "1", "TOKENIZERS_PARALLELISM": "false"}
    base = (f"{TRAIN_PY} scripts/train_grpo.py --model {START} --data data/processed/B_pool_hin.json")

    # ---- 2. smoke test: 3 steps with the 4-bit 8B judge; fall back to smaller micro-batches on OOM
    batch = "--batch-size 8 --grad-accum 2"
    for attempt, b in enumerate(["--batch-size 8 --grad-accum 2", "--batch-size 4 --grad-accum 4"]):
        peak["mib"] = 0
        note(f"SMOKE start ({b})")
        subprocess.run(f"rm -rf {R}/stages/stage3_smoke", shell=True)
        rc = run(f"{base} --out stages/stage3_smoke --max-steps 3 --save-steps 3 --no-merge --sample-rate 1.0 {b}",
                 f"{R}/results/logs/smoke.log", env=env8b, every=30)
        log = tail(f"{R}/results/logs/smoke.log", 60)
        if rc == 0:
            samples = [json.loads(l) for l in open(f"{R}/stages/stage3_smoke/train/reward_samples.json")]
            sr = [s["value"] for s in samples if s["reward"] == "safe_reply"]
            tl = json.load(open(f"{R}/stages/stage3_smoke/train/train_log.json"))
            note(f"SMOKE ok ({b}): GPU peak {peak['mib']} MiB, {tl['minutes']} min for 3 steps, "
                 f"safe_reply values {sorted(set(sr))} over {len(sr)} completions")
            batch = b
            break
        oom = "out of memory" in log.lower() or "OutOfMemory" in log
        note(f"SMOKE failed rc={rc} oom={oom}: " + log[-600:].replace("\n", " | "))
        if not oom or attempt == 1:
            note("FAILED smoke test")
            return

    # ---- 3. the run: 150 steps on the 1,357 Hinglish prompts, checkpoint every 25 steps (pushed as they land)
    out = "stages/stage3_hinglish"
    done = set()

    def push_new_checkpoints():
        for d in sorted(glob.glob(f"{R}/{out}/model/checkpoint-*")):
            n = d.rsplit("-", 1)[1]
            if n in done or not os.path.exists(f"{d}/trainer_state.json"):
                continue
            time.sleep(15)  # let the trainer finish writing the folder
            done.add(n)
            pushable(f"ckpt{n}", [f"{out}/model/checkpoint-{n}"], exclude=["optimizer.pt", "rng_state.pth", "scheduler.pt"])

    peak["mib"] = 0
    note(f"TRAIN start (150 steps, {batch}, start model {START}, safe_reply = {GUARD8B} 4-bit)")
    rc = run(f"{base} --out {out} --max-steps 150 --save-steps 25 {batch}", f"{R}/results/logs/grpo_hinglish.log",
             env=env8b, tick=push_new_checkpoints)
    push_new_checkpoints()
    if rc != 0 or not os.path.exists(f"{R}/{out}/model/merged/config.json"):
        note(f"FAILED training rc={rc}: " + tail(f"{R}/results/logs/grpo_hinglish.log", 8).replace("\n", " | "))
        pushable("train_logs_partial", [f"{out}/train", "results/logs"])
        return
    note(f"TRAIN done, GPU peak {peak['mib']} MiB")
    pushable("train_final", [f"{out}/model/adapter", f"{out}/train", "results/logs/grpo_hinglish.log", "results/logs/smoke.log",
                             "stages/stage3_smoke/train"])

    # ---- 4. scorecard of the new model (same script and settings as every earlier stage)
    note("EVAL stage3_hinglish start")
    rc = run(f"python scripts/evaluate.py --stage stage3_hinglish --model {out}/model/merged", f"{R}/results/logs/eval_stage3.log")
    if rc != 0:
        note("FAILED eval stage3: " + tail(f"{R}/results/logs/eval_stage3.log", 8).replace("\n", " | "))
        return
    note("EVAL stage3_hinglish done")
    pushable("eval_stage3", [f"{out}/eval", "results/logs/eval_stage3.log"])

    # ---- 5. control: the published Stage 2 model through the same pipeline today (no MILU)
    note("CONTROL stage2_rerun start")
    rc = run(f"python scripts/evaluate.py --stage stage2_rerun --model {START} --skip-milu", f"{R}/results/logs/eval_stage2_rerun.log")
    if rc != 0:
        note("FAILED control: " + tail(f"{R}/results/logs/eval_stage2_rerun.log", 8).replace("\n", " | "))
    else:
        note("CONTROL done")
        pushable("eval_stage2_rerun", ["stages/stage2_rerun/eval", "results/logs/eval_stage2_rerun.log"])

    # ---- 6. independent judge (never trained against) on every Tiny Aya stage
    note("JUDGE start (Qwen3Guard-Gen-8B)")
    inputs = " ".join([f"stage0_base=prior/stage0_base.jsonl", f"stage1_sft=prior/stage1_sft.jsonl",
                       f"stage2_grpo=prior/stage2_grpo.jsonl", f"stage3_hinglish={out}/eval/generations.json"]
                      + (["stage2_rerun=stages/stage2_rerun/eval/generations.json"]
                         if os.path.exists(f"{R}/stages/stage2_rerun/eval/generations.json") else []))
    rc = run(f"python scripts/judge_independent.py {inputs} --out results/independent_judge.jsonl", f"{R}/results/logs/judge_independent.log")
    if rc != 0:
        note("FAILED judge: " + tail(f"{R}/results/logs/judge_independent.log", 8).replace("\n", " | "))
    else:
        note("JUDGE done")
        pushable("judge", ["results/independent_judge.jsonl", "results/independent_judge.jsonl.summary.json",
                           "results/logs/judge_independent.log"])
    note("ALL_DONE")


main()
