"""Paper B, phase 1: stratified random points over the national 2020 forest-status lots, uniform by area.

Runs on the lab server with the standard library and numpy only (the lots are paged GeoJSON from the public WFS):

    python draw_phase1.py <lots_dir> <out_dir> [--n1 20000] [--floor 1000] [--seed 20261021] [--workers 12]

Strata (12) = record class from `nggocr` (1 natural forest, 2 planted forest, 3 not forest) x forest type from `malr3`
(1 protection, 2 special use, 3 production, other = 0 or missing). Lots without a valid `nggocr` are outside the
population and counted in the summary. Allocation: proportional to stratum area with a floor per stratum, rescaled to
n1. Lots are drawn with probability proportional to their polygon area (with replacement) and a point is drawn
uniformly inside the drawn lot, so points are uniform by area within each stratum. Areas use a local equal-area
approximation (lon scaled by cos(lat)), which is exact enough for lots of a few hectares and for stratum totals.

Output: phase1_points.csv (one row per point, lot attributes included) and phase1_strata.json (stratum areas and lot
counts, needed for the design weights). Pass 1 reads every page once; pass 2 rereads only the pages with drawn lots.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

R_EARTH = 6371008.8
RC = {1: "natural", 2: "planted", 3: "nonforest"}
FT = {1: "protection", 2: "special", 3: "production"}
KEEP = ["matinh", "tinh", "mahuyen", "huyen", "maxa", "xa", "tk", "khoanh", "lo", "dtich", "nggocr", "ldlr", "malr3",
        "mdsd", "churung", "sldlr", "namtr"]


def stratum(p):
    rc = RC.get(p.get("nggocr"))
    if rc is None:
        return None
    return f"{rc}|{FT.get(p.get('malr3'), 'other')}"


def rings(geom):
    if geom is None:
        return []
    if geom["type"] == "Polygon":
        return [geom["coordinates"]]
    if geom["type"] == "MultiPolygon":
        return geom["coordinates"]
    return []


def poly_area_m2(geom):
    """Area of a (Multi)Polygon in lon/lat degrees, in m2 (outer rings minus holes, local equal-area scaling)."""
    total = 0.0
    for poly in rings(geom):
        for k, ring in enumerate(poly):
            a = np.asarray(ring, dtype=float)
            if len(a) < 4:
                continue
            lat0 = math.radians(a[:, 1].mean())
            x = np.radians(a[:, 0]) * math.cos(lat0) * R_EARTH
            y = np.radians(a[:, 1]) * R_EARTH
            s = abs(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
            total += s if k == 0 else -s
    return max(total, 0.0)


def inside(px, py, geom):
    """Even-odd test of points (arrays) against all rings of a (Multi)Polygon."""
    res = np.zeros(len(px), dtype=bool)
    for poly in rings(geom):
        for ring in poly:
            a = np.asarray(ring, dtype=float)
            x1, y1, x2, y2 = a[:-1, 0], a[:-1, 1], a[1:, 0], a[1:, 1]
            cond = (y1[None, :] > py[:, None]) != (y2[None, :] > py[:, None])
            with np.errstate(divide="ignore", invalid="ignore"):
                xint = x1[None, :] + (py[:, None] - y1[None, :]) * (x2 - x1)[None, :] / (y2 - y1)[None, :]
            res ^= (np.sum(cond & (px[:, None] < xint), axis=1) % 2).astype(bool)
    return res


def pages(lots_dir):
    out = []
    for prov in sorted(os.listdir(lots_dir)):
        d = os.path.join(lots_dir, prov)
        if os.path.isdir(d):
            out += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".geojson.gz")]
    return out


def read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)["features"]


def pass1(path):
    rows, skipped = [], 0
    for i, ft in enumerate(read(path)):
        s = stratum(ft.get("properties") or {})
        a = poly_area_m2(ft.get("geometry"))
        if s is None or a <= 0:
            skipped += 1
            continue
        rows.append((s, a, i, float((ft["properties"] or {}).get("dtich") or 0)))
    return path, rows, skipped


def pass2(job):
    path, wanted, seed = job  # wanted: {feature index: [point ids]}
    rng = np.random.default_rng(seed)
    feats = read(path)
    out = []
    for i, pids in wanted.items():
        ft = feats[i]
        g = ft["geometry"]
        allc = np.concatenate([np.asarray(r, dtype=float) for poly in rings(g) for r in poly])
        (x0, y0), (x1, y1) = allc.min(0), allc.max(0)
        got = []
        for _ in range(200):
            if len(got) >= len(pids):
                break
            m = 64
            px, py = rng.uniform(x0, x1, m), rng.uniform(y0, y1, m)
            ok = inside(px, py, g)
            got += list(zip(px[ok], py[ok]))
        if len(got) < len(pids):
            raise RuntimeError(f"could not place points in {path}#{i}")
        p = ft["properties"]
        for pid, (x, y) in zip(pids, got):
            out.append({"pid": pid, "lon": round(float(x), 7), "lat": round(float(y), 7), "page": os.path.relpath(path),
                        "feature": i, **{k: p.get(k) for k in KEEP}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lots_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--n1", type=int, default=20000)
    ap.add_argument("--floor", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    plist = pages(a.lots_dir)
    print(f"{len(plist)} pages", flush=True)

    keys, area, page_of, feat_of, dt = [], [], [], [], []
    skipped = 0
    with Pool(a.workers) as pool:
        for n, (path, rows, sk) in enumerate(pool.imap_unordered(pass1, plist, chunksize=4)):
            skipped += sk
            for s, ar, i, d in rows:
                keys.append(s); area.append(ar); page_of.append(path); feat_of.append(i); dt.append(d)
            if n % 100 == 0:
                print(f"pass 1: {n}/{len(plist)} pages, {len(keys)} lots", flush=True)
    keys = np.asarray(keys); area = np.asarray(area); dt = np.asarray(dt)
    strata = sorted(set(keys.tolist()))
    A = {s: float(area[keys == s].sum()) for s in strata}
    L = {s: int((keys == s).sum()) for s in strata}
    D = {s: float(dt[keys == s].sum()) for s in strata}  # recorded area (ha), as a check on the polygon areas
    tot = sum(A.values())

    # allocation: proportional with a floor, the proportional part rescaled so the total is n1
    fixed = set()  # strata held at the floor; repeat until no proportional share falls below it
    while True:
        free = [s for s in strata if s not in fixed]
        rest = a.n1 - a.floor * len(fixed)
        Afree = sum(A[s] for s in free)
        low = {s for s in free if rest * A[s] / Afree < a.floor}
        if not low:
            break
        fixed |= low
    alloc = {s: a.floor for s in fixed}
    alloc.update({s: int(round(rest * A[s] / Afree)) for s in free})
    rng = np.random.default_rng(a.seed)
    wanted = defaultdict(lambda: defaultdict(list))
    pid = 0
    pid_stratum = {}
    for s in strata:
        idx = np.flatnonzero(keys == s)
        pick = rng.choice(idx, size=alloc[s], replace=True, p=area[idx] / area[idx].sum())
        for j in pick:
            wanted[page_of[j]][feat_of[j]].append(pid)
            pid_stratum[pid] = s
            pid += 1
    jobs = [(p, dict(w), int(rng.integers(2**31))) for p, w in wanted.items()]
    rows = []
    with Pool(a.workers) as pool:
        for n, out in enumerate(pool.imap_unordered(pass2, jobs, chunksize=4)):
            rows += out
            if n % 100 == 0:
                print(f"pass 2: {n}/{len(jobs)} pages, {len(rows)} points", flush=True)
    for r in rows:
        r["stratum"] = pid_stratum[r["pid"]]
    rows.sort(key=lambda r: r["pid"])
    cols = ["pid", "stratum", "lon", "lat"] + KEEP + ["page", "feature"]
    with open(os.path.join(a.out_dir, "phase1_points.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    summary = {"pages": len(plist), "lots_in_population": int(len(keys)), "lots_skipped": skipped,
               "area_ha_total": tot / 1e4, "seed": a.seed, "n1": len(rows),
               "strata": {s: {"area_ha": A[s] / 1e4, "dtich_ha": D[s], "lots": L[s], "n1": alloc[s]} for s in strata}}
    with open(os.path.join(a.out_dir, "phase1_strata.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
