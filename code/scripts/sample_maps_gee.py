"""Paper B: global-map values and terrain at the sample points, from Earth Engine (the analysis plan, RQ2 and RQ3 controls).

    .venv_rs/Scripts/python paper/land_audit/scripts/sample_maps_gee.py maps     # phase-2 points: GFC2020 V1-V4, TMF, WorldCover
    .venv_rs/Scripts/python paper/land_audit/scripts/sample_maps_gee.py terrain  # phase-1 points: SRTM elevation and slope

Each value is the pixel that contains the point, read in the map's own projection. Reads only the maps, never a label.
GFC2020 V4 is read here too and checked against the JRC COG values used for the phase-2 cells (sample_gfc.py).
Output: data/interim/land_audit/phase2_maps.csv, phase1_terrain.csv (one row per pid).
"""
import os
import sys
import time
from pathlib import Path

import ee
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/interim/land_audit"
TMF = "projects/JRC/TMF/v1_2025/AnnualChanges"  # latest TMF release on Earth Engine (2026-02-09)


def init():
    project = os.environ.get("GEE_PROJECT")
    env = ROOT / ".env"
    if not project and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEE_PROJECT="):
                project = line.split("=", 1)[1].strip().strip('"')
    ee.Initialize(project=project)


def layers():
    """name -> (image, projection) at native resolution."""
    out = {}
    for v in ("V1", "V2"):
        col = ee.ImageCollection(f"JRC/GFC2020/{v}")
        out[f"gfc{v}"] = (col.mosaic().select([0]).rename("v"), col.first().select([0]).projection())
    for v in ("V3", "V4"):
        img = ee.Image(f"JRC/GFC2020/{v}").select("Map")
        out[f"gfc{v}"] = (img.rename("v"), img.projection())
    tmf = ee.ImageCollection(TMF)
    out["tmf2020"] = (tmf.select("Dec2020").mosaic().rename("v"), tmf.first().select("Dec2020").projection())
    wc = ee.ImageCollection("ESA/WorldCover/v100").first().select("Map")
    out["wc2020"] = (wc.rename("v"), wc.projection())
    return out


def terrain():
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    img = dem.rename("elev").addBands(ee.Terrain.slope(dem).rename("slope"))
    return {"terrain": (img, dem.projection())}


def sample(pts: pd.DataFrame, lays: dict, chunk=1000) -> pd.DataFrame:
    rows = []
    for s in range(0, len(pts), chunk):
        part = pts.iloc[s:s + chunk]
        fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"pid": int(r.pid)})
                                   for r in part.itertuples()])
        got = {int(r.pid): {"pid": int(r.pid)} for r in part.itertuples()}
        for name, (img, proj) in lays.items():
            for k in range(6):
                try:
                    res = img.reduceRegions(collection=fc, reducer=ee.Reducer.first(), crs=proj).getInfo()
                    break
                except ee.EEException as e:
                    print(f"{name} chunk {s}: {str(e)[:100]}; retry", flush=True)
                    time.sleep(20 * (k + 1))
            else:
                raise RuntimeError(f"{name} chunk {s} failed")
            for f in res["features"]:
                p = f["properties"]
                if name == "terrain":
                    got[p["pid"]].update(elev=p.get("elev"), slope=p.get("slope"))
                else:
                    got[p["pid"]][name] = p.get("first", p.get("v"))
        rows += got.values()
        print(f"{min(s + chunk, len(pts))}/{len(pts)}", flush=True)
    return pd.DataFrame(rows)


def main():
    what = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else None  # optional: first n points only (test)
    init()
    if what == "maps":
        key = pd.read_csv(D / "phase2_key.csv", usecols=["ma_diem", "pid", "lon", "lat", "gfcV4"])
        key = key[key.ma_diem.str.startswith("B")]
        pts = key.head(n) if n else key
        out = sample(pts, layers()).merge(key[["pid", "gfcV4"]].rename(columns={"gfcV4": "gfcV4_cog"}), on="pid")
        for v in ("V1", "V2", "V3", "V4"):  # GFC2020 on Earth Engine masks non-forest instead of writing 0
            out[f"gfc{v}"] = out[f"gfc{v}"].fillna(0).astype(int)
        bad = (out.gfcV4 != out.gfcV4_cog).sum()
        print(f"GFC2020 V4, Earth Engine vs JRC COG: {bad} of {len(out)} points differ")
        print({c: out[c].value_counts(dropna=False).to_dict() for c in out.columns if c not in ("pid",)})
        if not n:
            out.to_csv(D / "phase2_maps.csv", index=False)
    elif what == "terrain":
        pts = pd.read_csv(D / "phase1_points.csv", usecols=["pid", "lon", "lat"])
        pts = pts.head(n) if n else pts
        out = sample(pts, terrain(), chunk=2000)
        print(out.describe().round(1).to_string())
        if not n:
            out.to_csv(D / "phase1_terrain.csv", index=False)


if __name__ == "__main__":
    main()
