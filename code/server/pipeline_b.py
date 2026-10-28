"""Paper B, VLM-B cross-fitting on server1 (the analysis plan), one uploaded batch, resumable stage by stage.

    python pipeline_b.py --gpus 0 [--folds 0,1,2,3,4]

For each fold k: train_k (QLoRA on the phase-2 points outside fold k) -> infer_k (that adapter on the fold-k phase-2
points and on every phase-1 point of the RQ3 population that is not in phase 2). Folds run one after another on each
GPU in --gpus (several GPUs: several folds at once). A GPU is free when it has < 2000 MiB in use and no stage of this
pipeline holds it; a stage that fails with an out-of-memory error is retried up to 3 times after waiting. A marker in
state/ records each finished stage, so a rerun continues where it stopped.
"""
import argparse
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent  # run/code
RUN = HERE.parent
PY = "/home/user/venv/bin/python"  # Python environment on the GPU server
MODEL = "/home/user/llm4deforest/models/Qwen3.5-9B"
ENV = "source ~/llm4deforest/env.sh >/dev/null 2>&1; deactivate 2>/dev/null; "
OOM = ("out of memory", "available kv cache memory: -", "no available memory for the cache")


def gpu_used():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return {int(i): int(m) for i, m in (l.split(",") for l in out.strip().splitlines())}


class Pool:
    def __init__(self, gpus):
        self.gpus, self.held, self.lock = gpus, set(), threading.Lock()

    def acquire(self, stage, hours=24):
        t0, said = time.time(), 0
        while True:
            with self.lock:
                used = gpu_used()
                for g in self.gpus:
                    if g not in self.held and used.get(g, 10 ** 9) < 2000:
                        self.held.add(g)
                        return g
            if time.time() - t0 > hours * 3600:
                raise SystemExit(f"[{stage}] no free GPU for {hours} h")
            if time.time() - said > 1800:
                print(f"[{stage}] waiting for a free GPU (in use: {used})", flush=True)
                said = time.time()
            time.sleep(120)

    def release(self, g):
        with self.lock:
            self.held.discard(g)


def run(pool, stage, cmd):
    state = RUN / "state" / stage
    if state.exists():
        print(f"[{stage}] already done", flush=True)
        return
    log = RUN / "logs" / f"{stage}.txt"
    for attempt in range(3):
        g = pool.acquire(stage)
        try:
            print(f"[{stage}] attempt {attempt + 1} on GPU {g}", flush=True)
            t0 = time.time()
            code = subprocess.run(["bash", "-c", f"{ENV}export CUDA_VISIBLE_DEVICES={g}; cd {RUN} && {cmd} > {log} 2>&1"]
                                  ).returncode
        finally:
            pool.release(g)
        if code == 0:
            state.touch()
            print(f"[{stage}] done in {time.time() - t0:.0f} s", flush=True)
            return
        text = log.read_text(errors="replace").lower()
        print(f"[{stage}] exit {code}: {text[-600:]}", flush=True)
        if not any(s in text for s in OOM):
            raise SystemExit(f"stage {stage} failed")
        time.sleep(120)
    raise SystemExit(f"stage {stage} failed 3 times with OOM")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    for d in ("state", "logs", "out", "adapters"):
        (RUN / d).mkdir(exist_ok=True)
    pool = Pool([int(g) for g in a.gpus.split(",")])
    folds = [int(k) for k in a.folds.split(",")]
    queue, lock, errors = list(folds), threading.Lock(), []

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                k = queue.pop(0)
            try:
                run(pool, f"train_{k}", f"{PY} code/qlora_b.py --model {MODEL} --train train_{k}.jsonl --images cards "
                                       f"--out adapters/fold{k} --epochs 2")
                run(pool, f"infer_{k}", f"{PY} code/infer_b.py --model {MODEL} --input infer_{k}.jsonl --images cards "
                                       f"--output out/infer_{k}.jsonl --lora adapters/fold{k}/epoch2")
            except SystemExit as e:
                errors.append(str(e))
                return

    threads = [threading.Thread(target=worker) for _ in pool.gpus]
    for t in threads:
        t.start()
        time.sleep(30)
    for t in threads:
        t.join()
    if errors:
        raise SystemExit("; ".join(errors))
    print("pipeline finished", flush=True)


if __name__ == "__main__":
    main()
