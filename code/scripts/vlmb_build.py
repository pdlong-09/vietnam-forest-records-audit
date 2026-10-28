"""Paper B, VLM-B (the analysis plan): folds, point list, Sentinel-2 evidence from Earth Engine, evidence cards.

    .venv_rs/Scripts/python paper/land_audit/scripts/vlmb_build.py folds    # K = 5 over the phase-2 points, by cell
    .venv_rs/Scripts/python paper/land_audit/scripts/vlmb_build.py points   # points that need a card
    .venv_rs/Scripts/python paper/land_audit/scripts/vlmb_build.py fetch [--threads 24] [--n N]
    .venv_rs/Scripts/python paper/land_audit/scripts/vlmb_build.py render

Card content (§6): Sentinel-2 L2A annual median composites for 2019, 2020, 2021 and 2025 (the latest full year), cloud-
masked with Cloud Score+ (cs_cdf >= 0.6) after dropping scenes over 70% cloudy, in a 64 x 64 px window (10 m, 640 m) in UTM around the point; true colour
(B4 B3 B2) and false colour (B11 B8 B4); NDVI / NBR / NDMI medians of the same years in the 1 ha cell (central 10 x 10 px)
and in the surroundings (pixels 100-300 m from the point); SRTM elevation and slope. Nothing from the record, the owner
or GFC is fetched or shown. Reads no label; the folds read only the point ids and cells.
Outputs in data/interim/land_audit/vlmb/: folds.csv, points.csv, tiles/<pid>.png (2 x 4 tiles of 64 px; black where
no clear pixel), indices.csv, cards/<pid>.jpg.
"""
import argparse
import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/interim/land_audit"
V = D / "vlmb"
YEARS = (2019, 2020, 2021, 2025)
PX, RES = 64, 10


# ---------- folds and points ----------
def folds():
    key = pd.read_csv(D / "phase2_key.csv", usecols=["ma_diem", "pid", "cell"])
    key = key[key.ma_diem.str.startswith("B")].sort_values("ma_diem")
    rng = np.random.default_rng(20261024)
    key["fold"] = -1
    for _, g in key.groupby("cell"):  # random order within the cell, then 0, 1, 2, 3, 4, 0, ... (stratified by cell)
        idx = g.index.to_numpy()[rng.permutation(len(g))]
        start = rng.integers(5)
        key.loc[idx, "fold"] = (start + np.arange(len(idx))) % 5
    V.mkdir(parents=True, exist_ok=True)
    key[["ma_diem", "pid", "cell", "fold"]].to_csv(V / "folds.csv", index=False)
    print(key.fold.value_counts().sort_index().to_dict())


def points():
    p1 = pd.read_csv(D / "phase1_points.csv", usecols=["pid", "stratum", "lon", "lat"])
    f = pd.read_csv(V / "folds.csv")
    rq3 = p1[~p1.stratum.str.startswith("nonforest")]  # RQ3 population: forest on the record (nggocr 1, 2)
    pts = p1[p1.pid.isin(set(rq3.pid) | set(f.pid))].copy()
    pts["phase2"] = pts.pid.isin(f.pid).astype(int)
    pts["rq3"] = pts.pid.isin(rq3.pid).astype(int)
    pts[["pid", "lon", "lat", "phase2", "rq3"]].to_csv(V / "points.csv", index=False)
    print(len(pts), "points;", int(pts.phase2.sum()), "phase 2;", int(pts.rq3.sum()), "in the RQ3 population")


# ---------- Earth Engine ----------
def init_ee():
    import ee
    project = os.environ.get("GEE_PROJECT")
    env = ROOT / ".env"
    if not project and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEE_PROJECT="):
                project = line.split("=", 1)[1].strip().strip('"')
    ee.Initialize(project=project)
    return ee


def request(ee, lon, lat):
    from pyproj import Transformer
    epsg = 32648 if lon < 108 else 32649  # Vietnam spans UTM zones 48 and 49
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    geom = ee.Geometry.Point([lon, lat]).buffer(500)
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    bands = []
    for yr in YEARS:
        c = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(f"{yr}-01-01", f"{yr + 1}-01-01").filterBounds(geom)
        # scenes over 70% cloudy add few clear pixels but most of the cost (24/10/2026: 2.3x faster, 24 points)
        c = c.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
        c = c.linkCollection(cs, ["cs_cdf"]).map(lambda i: i.updateMask(i.select("cs_cdf").gte(0.6)))
        m = c.select(["B2", "B3", "B4", "B8", "B11", "B12"]).median()
        bands += [m.visualize(bands=["B4", "B3", "B2"], min=0, max=2500, gamma=1.2).toInt16(),
                  m.visualize(bands=["B11", "B8", "B4"], min=0, max=[3500, 4500, 2000]).toInt16(),
                  m.normalizedDifference(["B8", "B4"]).multiply(10000).toInt16(),
                  m.normalizedDifference(["B8", "B12"]).multiply(10000).toInt16(),
                  m.normalizedDifference(["B8", "B11"]).multiply(10000).toInt16(),
                  m.select("B4").mask().gt(0).toInt16()]
    return ee.data.computePixels({
        "expression": ee.Image.cat(bands), "fileFormat": "NUMPY_NDARRAY",
        "grid": {"dimensions": {"width": PX, "height": PX}, "crsCode": f"EPSG:{epsg}",
                 "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": x - PX * RES / 2, "shearY": 0,
                                     "scaleY": -RES, "translateY": y + PX * RES / 2}}})


def summarise(arr):
    """Structured array (10 bands per year in request order: true RGB, false RGB, NDVI, NBR, NDMI, clear) -> tile
    mosaic (128 x 256 x 3 uint8) and index stats."""
    a = np.stack([arr[n] for n in arr.dtype.names], -1).astype(np.int32)  # (64, 64, 10 * 4)
    assert a.shape[-1] == 10 * len(YEARS), a.shape
    yy, xx = np.mgrid[:PX, :PX]
    dist = np.hypot(yy - (PX - 1) / 2, xx - (PX - 1) / 2) * RES
    cell = (np.abs(yy - (PX - 1) / 2) < 5) & (np.abs(xx - (PX - 1) / 2) < 5)
    ring = (dist >= 100) & (dist <= 300)
    mosaic = np.zeros((2 * PX, len(YEARS) * PX, 3), np.uint8)
    stats = {}
    for j, yr in enumerate(YEARS):
        b = a[..., 10 * j:10 * (j + 1)]
        valid = b[..., 9] > 0
        mosaic[:PX, j * PX:(j + 1) * PX] = np.where(valid[..., None], b[..., 0:3], 0)
        mosaic[PX:, j * PX:(j + 1) * PX] = np.where(valid[..., None], b[..., 3:6], 0)
        for k, name in enumerate(("ndvi", "nbr", "ndmi")):
            v = b[..., 6 + k] / 10000
            for zone, m in (("cell", cell), ("ring", ring)):
                ok = m & valid
                stats[f"{name}_{yr}_{zone}"] = round(float(np.median(v[ok])), 3) if ok.sum() >= 5 else None
        stats[f"clear_{yr}"] = round(float(valid.mean()), 3)
    return mosaic, stats


def fetch(threads, n):
    from PIL import Image
    ee = init_ee()
    pts = pd.read_csv(V / "points.csv")
    tiles = V / "tiles"
    tiles.mkdir(exist_ok=True)
    out = V / "indices.csv"
    done = set(pd.read_csv(out).pid) if out.exists() else set()
    todo = pts[~pts.pid.isin(done)]
    todo = todo.head(n) if n else todo
    lock, t0, count = threading.Lock(), time.time(), [0]
    f = open(out, "a", newline="", encoding="utf-8")
    writer = [None]

    def one(r):
        for k in range(8):
            try:
                arr = request(ee, r.lon, r.lat)
                break
            except Exception as e:  # 429 / transient compute errors: back off
                print(f"pid {r.pid} attempt {k + 1}: {str(e)[:120]}", flush=True)
                if k == 7:
                    print(f"pid {r.pid} failed", flush=True)
                    return
                time.sleep(10 * (k + 1))
        mosaic, stats = summarise(arr)
        Image.fromarray(mosaic).save(tiles / f"{r.pid}.png")
        with lock:
            if writer[0] is None:
                writer[0] = csv.DictWriter(f, ["pid"] + list(stats))
                if not done:
                    writer[0].writeheader()
            writer[0].writerow({"pid": r.pid, **stats})
            count[0] += 1
            if count[0] % 200 == 0:
                f.flush()
                rate = count[0] / (time.time() - t0)
                print(f"{count[0]}/{len(todo)}  {rate:.2f}/s  ~{(len(todo) - count[0]) / rate / 3600:.1f} h left", flush=True)

    with ThreadPoolExecutor(threads) as pool:
        list(pool.map(one, todo.itertuples()))
    f.close()
    print("done", count[0], "of", len(todo))


# ---------- cards ----------
def render():
    from PIL import Image, ImageDraw
    tiles, cards = V / "tiles", V / "cards"
    cards.mkdir(exist_ok=True)
    S, M, TOP, LEFT = 2, 6, 22, 70  # upscale, gap, header and row-label sizes (px)
    w = LEFT + len(YEARS) * (PX * S + M)
    h = TOP + 2 * (PX * S + M)
    n = 0
    for p in sorted(tiles.glob("*.png")):
        dst = cards / f"{p.stem}.jpg"
        if dst.exists():
            continue
        with Image.open(p) as im:
            mos = im.convert("RGB")
        card = Image.new("RGB", (w, h), (255, 255, 255))
        dr = ImageDraw.Draw(card)
        for j, yr in enumerate(YEARS):
            x0 = LEFT + j * (PX * S + M)
            dr.text((x0 + PX * S // 2 - 14, 5), str(yr), fill=(0, 0, 0))
            for i in range(2):
                y0 = TOP + i * (PX * S + M)
                tile = mos.crop((j * PX, i * PX, (j + 1) * PX, (i + 1) * PX)).resize((PX * S, PX * S), Image.NEAREST)
                card.paste(tile, (x0, y0))
                c0, c1 = (PX // 2 - 5) * S, (PX // 2 + 5) * S  # the 1 ha cell (10 x 10 px at 10 m)
                dr.rectangle((x0 + c0, y0 + c0, x0 + c1 - 1, y0 + c1 - 1), outline=(255, 255, 0), width=1)
        dr.text((4, TOP + PX * S // 2 - 6), "true\ncolour", fill=(0, 0, 0))
        dr.text((4, TOP + PX * S + M + PX * S // 2 - 12), "false\ncolour\nSWIR-NIR\n-red", fill=(0, 0, 0))
        card.save(dst, quality=92)
        n += 1
    print("rendered", n, "cards; size", (w, h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["folds", "points", "fetch", "render"])
    ap.add_argument("--threads", type=int, default=16)  # 32 threads were throttled to ~0.4 points/s (24/10/2026)
    ap.add_argument("--n", type=int)
    a = ap.parse_args()
    {"folds": folds, "points": points, "render": render}.get(a.what, lambda: fetch(a.threads, a.n))()


if __name__ == "__main__":
    main()
