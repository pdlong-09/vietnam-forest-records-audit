"""Paper B: independent check of the two-phase estimates, from the public reference sample only (no code shared with
analysis_b.py).

    python verify_b.py path/to/reference_sample_folder [path/to/results_folder]

1. Design weights recomputed from strata.csv and the phase-one cell counts, against weight_ha; they must sum to the
   population area.
2. Point estimates as weighted ratios sum(w y) / sum(w x), which equal the two-phase stratified estimator.
3. Analytic standard errors with the variance of two-phase sampling for stratification written in Cochran's form,
       V_h = sum_c W_hc^2 s_c^2 / n2_c + sum_c W_hc (ybar_c - ybar_h)^2 / n1_h,   V = sum_h P_h^2 V_h,
   applied to the linearised residual y - R x of each ratio.
4. A two-phase bootstrap (2000 replicates): phase-one points resampled within strata (new cell shares), phase-two
   points resampled within cells; the standard deviation of the replicate estimates is compared with 3.
If a results folder (results/ of the code release) is given, the published estimates are compared with 2-3.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

Z = 1.959964
B = 2000


def load(folder):
    f = Path(folder)
    st = pd.read_csv(f / "strata.csv").set_index("stratum")
    p1 = pd.read_csv(f / "phase1_points.csv", usecols=["stratum", "cell"])
    d = pd.read_csv(f / "phase2_points_labels.csv")
    n1h = p1.groupby("stratum").size()
    n1c = p1.groupby("cell").size()
    n2c = d.groupby("cell").size()
    w = d.stratum.map(st.area_ha) * d.cell.map(n1c) / d.stratum.map(n1h) / d.cell.map(n2c)
    checks = {"max relative difference to weight_ha": float((w / d.weight_ha - 1).abs().max()),
              "weights sum (Mha)": float(w.sum() / 1e6), "population (Mha)": float(st.area_ha.sum() / 1e6)}
    d["w"] = w
    return d, st, p1, n1h, n1c, n2c, checks


def cochran_se(d, st, n1h, n1c, n2c, z):
    """SE of the two-phase stratified mean (per ha of the population) of z."""
    P = st.area_ha / st.area_ha.sum()
    t = pd.DataFrame({"h": d.stratum, "c": d.cell, "z": z})
    v = 0.0
    for h, g in t.groupby("h"):
        cells = g.groupby("c").z.agg(["mean", "var"])
        W = cells.index.map(n1c) / n1h[h]
        yh = (W * cells["mean"]).sum()
        vh = (W ** 2 * cells["var"] / cells.index.map(n2c)).sum() + (W * (cells["mean"] - yh) ** 2).sum() / n1h[h]
        v += P[h] ** 2 * vh
    return np.sqrt(v)


def ratio(d, st, n1h, n1c, n2c, y, x):
    w = d.w.to_numpy()
    r = (w * y).sum() / (w * x).sum()
    xbar = (w * x).sum() / w.sum()
    return r, cochran_se(d, st, n1h, n1c, n2c, y - r * x) / xbar


def bootstrap(d, p1, st, items, seed=20261028):
    rng = np.random.default_rng(seed)
    P = (st.area_ha / st.area_ha.sum()).to_dict()
    cells_of = {h: g.cell.to_numpy() for h, g in p1.groupby("stratum")}
    idx_of = {c: np.flatnonzero((d.cell == c).to_numpy()) for c in d.cell.unique()}
    out = np.zeros((B, len(items)))
    Y = np.column_stack([np.column_stack([y, x]) for y, x in items.values()])  # pairs of columns
    for b in range(B):
        num = np.zeros(len(items))
        den = np.zeros(len(items))
        for h, cells in cells_of.items():
            draw = rng.choice(cells, size=len(cells), replace=True)
            u, cnt = np.unique(draw, return_counts=True)
            for c, k in zip(u, cnt):
                ii = idx_of[c]
                m = Y[rng.choice(ii, size=len(ii), replace=True)].mean(0)
                wgt = P[h] * k / len(cells)
                num += wgt * m[0::2]
                den += wgt * m[1::2]
        out[b] = num / den
    return out.std(0, ddof=1)


def main(folder, results=None):
    d, st, p1, n1h, n1c, n2c, checks = load(folder)
    print("weights:", checks)
    known = (d.f1_forest_2020 != "unknown").to_numpy(float)
    forest = (d.f1_forest_2020 == "forest").to_numpy(float)
    agri = d.agriculture_2020.to_numpy(float)
    recf = (d.record_origin != "without forest").to_numpy(float)
    recn = 1 - recf
    v4 = d.gfc2020_v4.to_numpy(float)
    comm = recf * (d.owner_group == "commune people's committee").to_numpy(float)
    items = {"EUDR forest share, record forest": (recf * forest * known, recf * known),
             "agriculture share, record forest": (recf * agri, recf),
             "EUDR forest share, record without forest": (recn * forest * known, recn * known),
             "agriculture share, record forest, commune People's Committees": (comm * agri, comm),
             "OA national records": (known * (recf == forest), known),
             "OA GFC2020 V4": (known * (v4 == forest), known),
             "paired OA difference V4 - records": (known * ((v4 == forest).astype(float) - (recf == forest)), known)}
    rows = []
    for name, (y, x) in items.items():
        r, se = ratio(d, st, n1h, n1c, n2c, np.asarray(y, float), np.asarray(x, float))
        rows.append({"estimate": name, "est": r, "se_analytic": se, "lo": r - Z * se, "hi": r + Z * se})
    t = pd.DataFrame(rows)
    t["se_bootstrap"] = bootstrap(d, p1, st, {k: (np.asarray(y, float), np.asarray(x, float)) for k, (y, x) in items.items()})
    t["bootstrap / analytic"] = t.se_bootstrap / t.se_analytic
    if results:
        R = Path(results)
        l1 = pd.read_csv(R / "rq1_landuse.csv")
        acc = pd.read_csv(R / "rq2_accuracy.csv")
        pdif = pd.read_csv(R / "rq2_paired_differences.csv")
        own = pd.read_csv(R / "rq1_owner.csv")
        pick = lambda df, **kw: df.loc[np.logical_and.reduce([df[k] == v for k, v in kw.items()])].iloc[0]  # noqa: E731
        pub = [pick(l1, domain="record=natural+planted", item="forest 2020, EUDR/FAO (F1)"),
               pick(l1, domain="record=natural+planted", item="agriculture 2020 (F3 in N1-N11)"),
               pick(l1, domain="record=nonforest", item="forest 2020, EUDR/FAO (F1)"),
               pick(own, record="record forest (natural + planted)", owner="ubnd_xa", item="agriculture 2020"),
               pick(acc, reference="EUDR/FAO", map="record (natural or planted)", item="OA"),
               pick(acc, reference="EUDR/FAO", map="GFC2020 V4", item="OA")]
        t["published est"] = [p.est for p in pub] + [np.nan]
        t["published lo"] = [p.lo for p in pub] + [np.nan]
        t["published hi"] = [p.hi for p in pub] + [np.nan]
        print("published paired differences:\n", pdif.to_string(index=False))
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print(t.round(4).to_string(index=False))
    return t, checks


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
