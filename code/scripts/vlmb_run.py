"""Paper B, VLM-B driver (the analysis plan): prompts, one batch on server1 (upload -> run -> pull -> delete), conversion.

    LAB_PW=... .venv_rs/Scripts/python paper/land_audit/scripts/vlmb_run.py run [--smoke] [--gpus 0]
    LAB_PW=... .venv_rs/Scripts/python paper/land_audit/scripts/vlmb_run.py resume --run-id vlmb_... [--gpus 1]
    .venv_rs/Scripts/python paper/land_audit/scripts/vlmb_run.py convert

Prompt: the card (vlmb_build.py render) plus the index table and the terrain; 6 options in cyclic order. Training uses
all 6 orders (one at random per point and epoch, qlora_b.py); inference uses the orders shifted by 0, 2 and 4 for every
point, and the class probabilities are the softmax over A-F averaged over those orders. Fold k's adapter predicts the
fold-k phase-2 points (out of fold) and the phase-1 points of the RQ3 population that are not in phase 2; those are
averaged over the 5 adapters. Writes data/interim/land_audit/vlmb/preds.csv.
"""
import argparse
import json
import math
import shutil
import sys
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/interim/land_audit"
V = D / "vlmb"
RAW = ROOT / "runs/land_audit/vlmb_raw"
REMOTE = "/data/tmp/llm4deforest_paperB"
PY = "/home/user/venv/bin/python"  # Python environment on the GPU server
QUEUE = "/data/tmp/llm4deforest_gpuq/gpu_queue.sh"  # tools/gpu_queue.sh: our GPU jobs one at a time (24/10/2026)
ALIVE = 'p=$(cat {R}/{run_id}.exit.pid 2>/dev/null); [ -n "$p" ] && kill -0 $p 2>/dev/null && echo yes || echo no'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lab_b import Lab  # noqa: E402

GROUPS = ["natural forest", "planted forest", "forest land without trees", "perennial crops", "annual crops", "other"]
OPTION = {"natural forest": "natural forest (including secondary forest and bamboo)",
          "planted forest": "planted timber forest (acacia, eucalyptus, pine, including newly planted)",
          "forest land without trees": "forest land temporarily without trees (cleared or burnt, regrowing)",
          "perennial crops": "perennial crops (coffee, rubber, cashew, pepper, tea, fruit trees)",
          "annual crops": "annual crops (rice, maize, cassava, sugarcane, upland fields)",
          "other": "other land (grassland, shrubland, settlements, water, bare land)"}
ORDERS = [GROUPS[i:] + GROUPS[:i] for i in range(6)]
INFER_ORDERS = (0, 2, 4)
YEARS = (2019, 2020, 2021, 2025)


def facts(r):
    f = lambda v: "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"  # noqa: E731
    lines = [f"{y}: NDVI {f(r[f'ndvi_{y}_cell'])} / {f(r[f'ndvi_{y}_ring'])}, NBR {f(r[f'nbr_{y}_cell'])} / "
             f"{f(r[f'nbr_{y}_ring'])}, NDMI {f(r[f'ndmi_{y}_cell'])} / {f(r[f'ndmi_{y}_ring'])}" for y in YEARS]
    return ("Sentinel-2 annual median composites around a sample point in Vietnam, for 2019, 2020, 2021 and 2025 "
            "(columns). Top row: true colour. Bottom row: false colour (SWIR1, NIR, red), where green vegetation is "
            "bright green and bare or dry land is pink. Each tile covers 640 m x 640 m; the yellow square is the 1-ha "
            "cell (100 m x 100 m) at the point. Black means no cloud-free observation that year.\n"
            "Spectral indices, annual median, in the cell / in the surroundings (100-300 m from the point):\n"
            + "\n".join(lines) + f"\nTerrain: elevation {r['elev']:.0f} m, slope {r['slope']:.0f} degrees.\n"
            "Question: what was the land use of the 1-ha cell on 31 December 2020? Judge land use, not only tree "
            "cover: forest is land with trees taller than 5 m and canopy cover above 10% on at least 0.5 ha that is "
            "not under agricultural use.")


def prompt(base, order):
    opts = "\n".join(f"{chr(65 + i)}. {OPTION[g]}" for i, g in enumerate(ORDERS[order]))
    return f"{base}\n{opts}\nAnswer with the letter only."


def tables(smoke=False):
    pts = pd.read_csv(V / "points.csv")
    if smoke:  # the smoke test only needs points whose evidence is already fetched
        pts = pts[pts.pid.isin(pd.read_csv(V / "indices.csv").pid) & pts.pid.map(lambda p: (V / "cards" / f"{p}.jpg").exists())]
    ind = pd.read_csv(V / "indices.csv").drop_duplicates("pid")
    ter = pd.read_csv(D / "phase1_terrain.csv")
    folds = pd.read_csv(V / "folds.csv")
    lab = pd.read_csv(D / "labels_final.csv", usecols=["ma_diem", "landuse_2020"])
    p = pts.merge(ind, on="pid", how="left").merge(ter, on="pid", how="left").merge(
        folds[["pid", "ma_diem", "fold"]], on="pid", how="left").merge(lab, on="ma_diem", how="left")
    missing = p[p.ndvi_2020_cell.isna() & p.clear_2020.isna()]
    if len(missing):
        raise SystemExit(f"{len(missing)} points have no fetched evidence yet")
    return p


def build(pkg, smoke):
    p = tables(smoke)
    ph2 = p[p.phase2 == 1]
    rest = p[(p.phase2 == 0) & (p.rq3 == 1)]
    folds = [0] if smoke else range(5)
    if smoke:
        ph2 = pd.concat([g.head(8) for _, g in ph2.groupby("fold")])
        rest = rest.head(16)
    used = set()
    for k in folds:
        with open(pkg / f"train_{k}.jsonl", "w", encoding="utf-8") as f:
            for r in ph2[ph2.fold != k].itertuples():
                base = facts(r._asdict())
                f.write(json.dumps({"id": str(r.pid), "image": f"{r.pid}.jpg", "texts": [prompt(base, o) for o in range(6)],
                                    "answers": [chr(65 + ORDERS[o].index(r.landuse_2020)) for o in range(6)]}) + "\n")
                used.add(r.pid)
        with open(pkg / f"infer_{k}.jsonl", "w", encoding="utf-8") as f:
            for r in pd.concat([ph2[ph2.fold == k], rest]).itertuples():
                base = facts(r._asdict())
                for o in INFER_ORDERS:
                    f.write(json.dumps({"id": f"{r.pid}|{o}", "image": f"{r.pid}.jpg", "text": prompt(base, o)}) + "\n")
                used.add(r.pid)
    (pkg / "cards").mkdir()
    for pid in used:
        shutil.copy(V / "cards" / f"{pid}.jpg", pkg / "cards" / f"{pid}.jpg")
    shutil.copytree(ROOT / "paper/land_audit/server", pkg / "code", ignore=shutil.ignore_patterns("__pycache__"))
    print(f"package: folds {list(folds)}, phase-2 {len(ph2)}, other RQ3 points {len(rest)}, cards {len(used)}", flush=True)
    return list(folds)


def queued(lab, gpu, workdir, cmd):
    """cmd wrapped in the GPU queue (installed from tools/gpu_queue.sh if missing)."""
    if lab.ssh(f"test -x {QUEUE} && echo ok || echo no").strip() != "ok":
        lab.ssh(f"mkdir -p {Path(QUEUE).parent.as_posix()}")
        lab.put(ROOT / "tools/gpu_queue.sh", QUEUE)
        lab.ssh(f"chmod 700 {QUEUE}")
    return f"{QUEUE} {gpu} {workdir} {cmd}"


def collect(lab, run_id, raw):
    """Pack the results on the server, pull them into raw/ and unpack."""
    # only the epoch-2 adapters are used (qlora_b.py), so only those come back
    lab.retry(lambda: lab.ssh(f"cd {REMOTE}/{run_id} && tar cf ../{run_id}_out.tar out logs adapters/fold*/epoch2 "
                              f"pipeline_log.txt state"), "pack")
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir(parents=True)
    lab.get_chunked(f"{REMOTE}/{run_id}_out.tar", raw / "out.tar")
    with tarfile.open(raw / "out.tar") as t:
        t.extractall(raw, filter="data")
    (raw / "out.tar").unlink()
    (raw / "RUN_ID").write_text(run_id)


def cleanup(lab, run_id):
    left = lab.retry(lambda: lab.ssh(f"rm -rf {REMOTE}/{run_id} {REMOTE}/{run_id}.tar* {REMOTE}/{run_id}_out.tar "
                                     f"{REMOTE}/{run_id}.exit* {REMOTE}/{run_id}.unpack*; ls {REMOTE} | wc -l"), "clean")
    print("server data left on server1:", left.strip(), flush=True)


def resume(a):
    """Continue a batch whose pipeline stopped, in its own directory on server1 (stages are resumable), through the GPU
    queue; then pull and delete. 24/10/2026: train_4 died when another job took GPU 0 at the same time
    (folds 0-3 were done and stayed on the server)."""
    run_id, lab, keep = a.run_id, Lab(), True
    try:
        alive = lab.ssh(ALIVE.format(R=REMOTE, run_id=run_id)).strip() == "yes"
        if not alive:  # a job still running (queued or on the GPU) keeps its pid file, so it is polled, not restarted
            lab.ssh(f"rm -f {REMOTE}/{run_id}.exit {REMOTE}/{run_id}.exit.pid {REMOTE}/{run_id}_out.tar")
        cmd = queued(lab, a.gpus, f"{REMOTE}/{run_id}", f"{PY} code/pipeline_b.py --gpus {a.gpus} --folds 0,1,2,3,4")
        code = lab.run_detached(REMOTE, f"{cmd} >> {REMOTE}/{run_id}/pipeline_log.txt 2>&1", f"{REMOTE}/{run_id}.exit",
                                poll_s=120, timeout_h=30)
        print("pipeline exit", code, "|", lab.ssh(f"tail -3 {REMOTE}/{run_id}/pipeline_log.txt").strip()[-300:], flush=True)
        collect(lab, run_id, RAW / "full")
        keep = code != 0  # a failed pipeline keeps its inputs on the server for another resume
    finally:
        if keep:
            print(f"server data kept on server1 in {REMOTE}/{run_id}", flush=True)
        else:
            cleanup(lab, run_id)
        lab.close()


def run(a):
    run_id = f"vlmb{'smoke' if a.smoke else ''}_{time.strftime('%Y%m%d%H%M%S')}"
    raw = RAW / ("smoke" if a.smoke else "full")
    lab = Lab()
    keep = False
    try:
        with tempfile.TemporaryDirectory(prefix="vlmb_") as td:
            pkg = Path(td) / run_id
            pkg.mkdir()
            folds = build(pkg, a.smoke)
            # gzip: the prompts repeat (356 MB of jsonl -> 14 MB); the JPEG cards do not shrink (24/10/2026, slow bore)
            tar = Path(td) / f"{run_id}.tar.gz"
            with tarfile.open(tar, "w:gz", compresslevel=1) as t:
                t.add(pkg, arcname=run_id)
            print(f"payload {tar.stat().st_size / 1e6:.0f} MB", flush=True)
            lab.ssh(f"mkdir -p {REMOTE}")
            lab.put_chunked(tar, f"{REMOTE}/{run_id}.tar.gz", workers=6)
        if lab.run_detached(REMOTE, f"tar xzf {run_id}.tar.gz && rm {run_id}.tar.gz", f"{REMOTE}/{run_id}.unpack", poll_s=15) != 0:
            raise SystemExit("unpack failed")
        code = lab.run_detached(REMOTE, f"cd {run_id} && {PY} code/pipeline_b.py "
                                        f"--gpus {a.gpus} --folds {','.join(map(str, folds))} > pipeline_log.txt 2>&1",
                                f"{REMOTE}/{run_id}.exit", poll_s=120, timeout_h=30)
        print("pipeline exit", code, "|", lab.ssh(f"tail -3 {REMOTE}/{run_id}/pipeline_log.txt").strip()[-300:], flush=True)
        collect(lab, run_id, raw)
        keep = code != 0  # 24/10/2026: a failed pipeline keeps its inputs on the server, for `resume`
    finally:
        if keep:
            print(f"server data kept on server1 in {REMOTE}/{run_id}", flush=True)
        else:
            cleanup(lab, run_id)
        lab.close()


def convert():
    out = RAW / "full" / "out"
    rows = {}
    for k in range(5):
        for line in open(out / f"infer_{k}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            pid, o = map(int, r["id"].split("|"))
            lp = np.array([r["logprobs"][chr(65 + i)] for i in range(6)], float)
            pr = np.exp(lp - np.max(lp))
            pr /= pr.sum()
            p = np.zeros(6)
            for i, g in enumerate(ORDERS[o]):
                p[GROUPS.index(g)] = pr[i]
            rows.setdefault(pid, {}).setdefault(k, []).append(p)
    pts = pd.read_csv(V / "points.csv").merge(pd.read_csv(V / "folds.csv")[["pid", "fold"]], on="pid", how="left")
    rec = []
    for r in pts.itertuples():
        by = rows.get(r.pid, {})
        if r.phase2:  # out of fold: only the adapter that did not see the point
            ps = [np.mean(by[int(r.fold)], 0)] if int(r.fold) in by else []
            kind = "out_of_fold"
        else:
            ps = [np.mean(v, 0) for v in by.values()]
            kind = f"mean_of_{len(ps)}"
        if not ps:
            continue
        p = np.mean(ps, 0)
        rec.append({"pid": r.pid, "kind": kind, **{f"p_{g}": round(float(v), 5) for g, v in zip(GROUPS, p)}})
    t = pd.DataFrame(rec)
    t.to_csv(V / "preds.csv", index=False)
    print(t.kind.value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "resume", "convert"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--run-id")
    a = ap.parse_args()
    {"run": lambda: run(a), "resume": lambda: resume(a), "convert": convert}[a.what]()


if __name__ == "__main__":
    main()
