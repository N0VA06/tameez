# General benchmarks for the model after the extra Hinglish GRPO run: same run_benchmarks.sh and settings as the other six models.
# Runs as one kernel cell (queued right behind the Hinglish job) so the kernel never idles.
import json, os, subprocess, time
M = "/content/tameez_aya/stages/stage3_hinglish/model/merged"; N = "tiny_aya_hinglish"
def stamp(): return time.strftime("%H:%M:%S", time.gmtime())
def note(msg):
    line = f"{stamp()} {msg}"; print(line, flush=True)
    with open("/content/bench_hinglish.log", "a") as f: f.write(line + "\n")
note("BENCH generation_config eos " + str(json.load(open(f"{M}/generation_config.json")).get("eos_token_id")))
note(f"BENCH START {N} ({M})")
with open(f"/content/{N}.log", "w") as f:
    p = subprocess.Popen(f"bash /content/run_benchmarks.sh {M} {N} 500", shell=True, stdout=f, stderr=subprocess.STDOUT, cwd="/content")
    t0 = time.time()
    while p.poll() is None:
        time.sleep(60); print(f"{stamp()} heartbeat {N} {int(time.time() - t0) // 60} min", flush=True)
note(("BENCH DONE " if p.returncode == 0 else f"BENCH FAILED rc={p.returncode} ") + N)
subprocess.run(f"cd /content && tar czf /content/{N}.tgz results/{N} {N}.log bench_hinglish.log", shell=True)
print(f"{stamp()} PUSHABLE {N} /content/{N}.tgz", flush=True)
r = subprocess.run(f"hf upload IINOVAII/tameez-eval-outputs /content/results/{N} general_benchmarks/results/{N} --repo-type dataset "
                   f"--commit-message 'Add raw general-benchmark results for the model after the extra Hinglish run'", shell=True, capture_output=True, text=True)
note("BENCH VM-PUSH " + ("ok" if r.returncode == 0 else "failed: " + r.stderr[-200:]))
