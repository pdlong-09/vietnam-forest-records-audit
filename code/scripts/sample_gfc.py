"""Read GFC2020 (JRC open data COG, 1 = forest, 0 = non-forest) at the phase-1 points, in parallel over HTTP ranges.

    python paper/land_audit/scripts/sample_gfc.py data/interim/land_audit/phase1_points.csv V4 [--threads 16]

Writes <points>_gfc<version>.csv with pid and the map value. Already sampled pids are skipped, so the run resumes.
"""
import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pandas as pd
import rasterio

os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                  GDAL_HTTP_MULTIRANGE="YES", VSI_CACHE="TRUE", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3")
URL = "/vsicurl/https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/FOREST/GFC2020/LATEST/single-cog/JRC_GFC2020_{v}_COG.tif"
local = threading.local()


def value(row, url):
    if not hasattr(local, "ds"):
        local.ds = rasterio.open(url)
    return row.pid, int(next(local.ds.sample([(row.lon, row.lat)]))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("version")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    out = Path(a.points).with_name(Path(a.points).stem + f"_gfc{a.version}.csv")
    done = set(pd.read_csv(out).pid) if out.exists() else set()
    pts = pd.read_csv(a.points, usecols=["pid", "lon", "lat"])
    pts = pts[~pts.pid.isin(done)]
    url = URL.format(v=a.version)
    new = not out.exists()
    with open(out, "a", newline="") as f, ThreadPoolExecutor(a.threads) as pool:
        w = csv.writer(f)
        if new:
            w.writerow(["pid", f"gfc{a.version}"])
        for n, (pid, v) in enumerate(pool.map(lambda r: value(r, url), pts.itertuples())):
            w.writerow([pid, v])
            if n % 500 == 0:
                f.flush()
                print(f"{n}/{len(pts)}", flush=True)
    print("done", out)


if __name__ == "__main__":
    main()
