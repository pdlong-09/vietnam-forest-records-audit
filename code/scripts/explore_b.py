"""Paper B, exploratory analyses written 25/10/2026 AFTER the planned results (the results notes); post hoc throughout.

    .venv_rs/Scripts/python paper/land_audit/scripts/explore_b.py gft       # local (GEE): JRC GFT2020 at the phase-2 points
    .venv/Scripts/python paper/land_audit/scripts/explore_b.py bundle       # local: anonymised inputs for server2
    venv/bin/python code/explore_b.py run --bundle bundle --out out         # server2 workspace (numpy, pandas only)

1. PPI++ for the RQ3 logit (Angelopoulos, Duchi & Zrnic 2023): the two-phase equations of ppi_b.py with weight lambda on
   the model terms,
       sum_P2 w2 x (y - lambda f - (1 - lambda) p) + lambda sum_P1 w1 x (f - p) = 0,
   lambda = 0 is the human-only fit, lambda = 1 the planned PPI. Every lambda on a grid of 0.05 is fitted when the
   equations have a finite root (each 0/1 column's target share inside its bounds); lambda-hat minimises the summed
   variance of the non-intercept coefficients (power tuning, on the grid). The same with the owner groups that have fewer
   than 30 phase-2 points merged into one group.
2. Natural/planted split against JRC GFT2020 (JRC/GFC2020_subtypes/V1, band Map: 1 naturally regenerating, 10 primary,
   20 planted; unmapped = not forest in GFC2020 V3): by record class and human label, area-weighted (two-phase ratio
   estimator of analysis_b.py). Tests the caveat that interpreters call 47.8% of record-planted area natural forest.
The bundle holds no owner name, place name, interpreter name, lot number or coordinate: owners are already mapped to
their groups (owner_groups.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_b as A  # noqa: E402
import ppi_b as P  # noqa: E402

KEEP = ["pid", "cell", "stratum", "n1_h", "n1_c", "n2_c", "P_h", "w", "record", "record_forest", "malr3_cat", "owner",
        "region", "lu20", "known", "agri_2020", "forest_eudr", "elev", "slope"]
GFT = {1: "naturally regenerating", 10: "primary", 20: "planted", 0: "not forest (GFC2020)"}


# ---------- local steps ----------
def gft():
    import ee
    from sample_maps_gee import init, sample
    init()
    img = ee.Image("JRC/GFC2020_subtypes/V1").select("Map")
    pts = pd.read_csv(A.D / "phase2_key.csv", usecols=["pid", "lon", "lat"])
    t = sample(pts, {"gft": (img.unmask(0).rename("v"), img.projection())}, chunk=600)
    t["gft"] = t.gft.fillna(0).astype(int)
    t[["pid", "gft"]].to_csv(A.D / "phase2_gft.csv", index=False)
    print(t.gft.value_counts().to_dict())


def bundle():
    out = A.ROOT / "runs/land_audit/explore_bundle"
    out.mkdir(parents=True, exist_ok=True)
    d, _ = A.load()
    pr = pd.read_csv(P.V / "preds.csv")
    d = d[KEEP].merge(pr, on="pid", how="left").merge(pd.read_csv(A.D / "phase2_gft.csv"), on="pid", how="left")
    d.to_csv(out / "phase2.csv", index=False)
    p1 = P.phase1()[["pid", "stratum", "w1", "owner", "malr3_cat", "region", "elev", "slope", "f", "kind"]]
    p1.to_csv(out / "phase1.csv", index=False)
    print(f"bundle: phase 2 {d.shape}, phase 1 {p1.shape} -> {out}")


# ---------- PPI++ ----------
def fit(p1, p2, ref, lam, cols=None):
    """Returns (beta, cov, cols) or None when the equations have no finite root for this lambda."""
    X1, cols = P.design(p1, ref, cols)
    X2, _ = P.design(p2, ref, cols)
    w1, f1 = p1.w1.to_numpy(), p1.f.to_numpy()
    w2, y2, f2 = p2.w.to_numpy(), p2.agri_2020.to_numpy(float), p2.f.to_numpy()
    for j in range(len(cols) - 2):  # 0/1 columns: the target share must lie inside what probabilities can reach
        m1, m2 = X1[:, j] == 1, X2[:, j] == 1
        target = (w2 * m2 * (y2 - lam * f2)).sum() + lam * (w1 * m1 * f1).sum()
        upper = (1 - lam) * (w2 * m2).sum() + lam * (w1 * m1).sum()
        if not (0 < target < upper):
            return None
    beta = np.zeros(X1.shape[1])
    for _ in range(200):
        p_1, p_2 = 1 / (1 + np.exp(-X1 @ beta)), 1 / (1 + np.exp(-X2 @ beta))
        g = X2.T @ (w2 * (y2 - lam * f2 - (1 - lam) * p_2)) + lam * X1.T @ (w1 * (f1 - p_1))
        H = (X2 * (w2 * (1 - lam) * p_2 * (1 - p_2))[:, None]).T @ X2 + lam * (X1 * (w1 * p_1 * (1 - p_1))[:, None]).T @ X1
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    else:
        return None
    p_1, p_2 = 1 / (1 + np.exp(-X1 @ beta)), 1 / (1 + np.exp(-X2 @ beta))
    H = (X2 * (w2 * (1 - lam) * p_2 * (1 - p_2))[:, None]).T @ X2 + lam * (X1 * (w1 * p_1 * (1 - p_1))[:, None]).T @ X1
    V2 = P.strat_var(X2 * (w2 * (y2 - lam * f2 - (1 - lam) * p_2))[:, None], p2.cell.to_numpy())
    V1 = P.strat_var(X1 * (lam * w1 * (f1 - p_1))[:, None], p1.stratum.to_numpy()) if lam > 0 else 0
    Hi = np.linalg.inv(H)
    return beta, Hi @ (V1 + V2) @ Hi, cols


def ppi_plus(p1, p2, tag, out):
    ref = {"owner": "ubnd_xa", "malr3_cat": "production", "region": p2.groupby("region").size().idxmax()}
    p1 = p1[p1.owner.isin(set(p2.owner))]  # levels without phase-2 points cannot be rectified (hop_tac_xa: 3 points)
    grid, best = [], None
    for lam in np.round(np.arange(0, 1.0001, 0.05), 2):
        r = fit(p1, p2, ref, lam)
        tr = float(np.trace(r[1][1:-2, 1:-2]) + np.trace(r[1][-2:, -2:])) if r else np.nan
        grid.append({"lambda": lam, "feasible": r is not None, "sum_var_non_intercept": tr})
        if r and (best is None or tr < best[0]):
            best = (tr, lam, r)
    g = pd.DataFrame(grid)
    g.to_csv(out / f"ppipp_{tag}_lambda_grid.csv", index=False)
    _, lam, (beta, cov, cols) = best
    b0, c0, _ = fit(p1, p2, ref, 0.0, cols)  # human-only on the same specification
    se, se0 = np.sqrt(np.diag(cov)), np.sqrt(np.diag(c0))
    t = pd.DataFrame({"term": cols, "coef": beta, "se": se, "human_coef": b0, "human_se": se0})
    i, j = cols.index("owner=vuon_quoc_gia_khu_bao_ton"), cols.index("malr3_cat=special")
    t.loc[len(t)] = ["owner=vuon_quoc_gia_khu_bao_ton + malr3_cat=special", beta[i] + beta[j],
                     np.sqrt(cov[i, i] + cov[j, j] + 2 * cov[i, j]), b0[i] + b0[j], np.sqrt(c0[i, i] + c0[j, j] + 2 * c0[i, j])]
    for k, (c, s) in {"": ("coef", "se"), "human_": ("human_coef", "human_se")}.items():
        t[f"{k}OR"], t[f"{k}lo"], t[f"{k}hi"] = np.exp(t[c]), np.exp(t[c] - A.Z * t[s]), np.exp(t[c] + A.Z * t[s])
    t["se_ratio"] = t.se / t.human_se
    t["n_phase2"] = [len(p2)] + [int((p2[c.split("=")[0]] == c.split("=")[1]).sum()) if "=" in c and "+" not in c else len(p2)
                                 for c in t.term[1:]]
    t.to_csv(out / f"ppipp_{tag}_logit.csv", index=False)
    return {"lambda_hat": float(lam), "feasible_lambdas": g.loc[g.feasible, "lambda"].tolist(), "n_phase1": len(p1),
            "n_phase2": len(p2)}


# ---------- GFT2020 ----------
def gft_check(p2, out):
    d = p2[p2.known == 1].copy() if "known" in p2 else p2.copy()
    d["gft_class"] = d.gft.map(GFT)
    rows = []
    for rec in ("natural", "planted"):
        for lab in ("natural forest", "planted forest"):
            x = ((d.record == rec) & (d.lu20 == lab)).astype(float).to_numpy()
            for g in GFT.values():
                y = x * (d.gft_class == g).to_numpy()
                r = A.ratio(d, y, x)
                rows.append({"record": rec, "human label": lab, "GFT2020": g, "n": int(y.sum()), "share": r["est"],
                             "lo": r["est"] - A.Z * r["se"], "hi": r["est"] + A.Z * r["se"]})
    t = pd.DataFrame(rows)
    t.to_csv(out / "gft2020_by_record_and_label.csv", index=False)
    # agreement on natural vs planted among points that the interpreters call forest and GFT2020 maps as forest
    f = d[d.lu20.isin(["natural forest", "planted forest"]) & d.gft.isin([1, 10, 20])]
    agree = ((f.lu20 == "planted forest") == (f.gft == 20)).astype(float).to_numpy()
    r = A.ratio(f, agree, np.ones(len(f)))
    return {"points both forest": int(len(f)), "natural/planted agreement (area-weighted)": r["est"],
            "ci": [r["est"] - A.Z * r["se"], r["est"] + A.Z * r["se"]]}


# ---------- 3. AlphaEarth 2020 as the predictor (the analysis plan, 25/10/2026, post hoc) ----------
AEF_BANDS = [f"A{i:02d}" for i in range(64)]


def aef():
    """Local (GEE): mean AlphaEarth 2020 embedding in the 1-ha cell and the 100-300 m ring of every VLM-B point."""
    import csv
    from concurrent.futures import ThreadPoolExecutor
    import ee
    from sample_maps_gee import init
    init()
    img = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate("2020-01-01", "2021-01-01").mosaic() \
        .select(AEF_BANDS)
    pts = pd.read_csv(P.V / "points.csv", usecols=["pid", "lon", "lat"])
    out = A.D / "aef2020.csv"
    done = set(pd.read_csv(out, usecols=["pid"]).pid) if out.exists() else set()
    todo = pts[~pts.pid.isin(done)]
    parts = [todo.iloc[i:i + 150] for i in range(0, len(todo), 150)]

    def req(part):
        feats = []
        for r in part.itertuples():
            pt = ee.Geometry.Point([float(r.lon), float(r.lat)])
            feats += [ee.Feature(pt.buffer(50).bounds(), {"pid": int(r.pid), "zone": "cell"}),
                      ee.Feature(pt.buffer(300).difference(pt.buffer(100)), {"pid": int(r.pid), "zone": "ring"})]
        for k in range(8):
            try:
                got = img.reduceRegions(ee.FeatureCollection(feats), ee.Reducer.mean(), 10, tileScale=4).getInfo()
                return [{"pid": f["properties"]["pid"], "zone": f["properties"]["zone"],
                         **{b: f["properties"].get(b) for b in AEF_BANDS}} for f in got["features"]]
            except Exception as e:  # 429 in Restricted Mode, transient errors
                print(f"retry {k + 1}: {str(e)[:100]}", flush=True)
                import time
                time.sleep(20 * (k + 1))
        raise RuntimeError("AlphaEarth request failed")
    new = not out.exists()
    with open(out, "a", newline="") as f, ThreadPoolExecutor(4) as pool:
        w = csv.DictWriter(f, ["pid", "zone"] + AEF_BANDS)
        if new:
            w.writeheader()
        for n, rows in enumerate(pool.map(req, parts), 1):
            w.writerows(rows)
            f.flush()
            if n % 10 == 0 or n == len(parts):
                print(f"{n}/{len(parts)}", flush=True)


def aef_wide():
    t = pd.read_csv(A.D / "aef2020.csv").drop_duplicates(["pid", "zone"], keep="last")
    w = t.pivot(index="pid", columns="zone", values=AEF_BANDS)
    w.columns = [f"{z}_{b}" for b, z in w.columns]
    ring = [c for c in w.columns if c.startswith("ring_")]
    for c in ring:  # a ring without pixels takes the cell value (none expected)
        w[c] = w[c].fillna(w[c.replace("ring_", "cell_")])
    return w.reset_index()


def bundle2():
    out = A.ROOT / "runs/land_audit/explore_bundle2"
    out.mkdir(parents=True, exist_ok=True)
    b = A.ROOT / "runs/land_audit/explore_bundle"
    a = aef_wide()
    folds = pd.read_csv(P.V / "folds.csv", usecols=["pid", "fold"])
    p2 = pd.read_csv(b / "phase2.csv").merge(folds, on="pid", how="left", validate="1:1").merge(a, on="pid", how="left")
    p1 = pd.read_csv(b / "phase1.csv").merge(pd.read_csv(P.V / "preds.csv"), on="pid", how="left").merge(a, on="pid", how="left")
    assert p2.fold.notna().all() and p2.cell_A00.notna().all() and p1.cell_A00.notna().all()
    p2.to_csv(out / "phase2.csv", index=False)
    p1.to_csv(out / "phase1.csv", index=False)
    print(f"bundle2: phase 2 {p2.shape}, phase 1 {p1.shape} -> {out}")


def crossfit(p2, p1, feats):
    """Agriculture probability: out of fold on phase 2, mean of the 5 fold models on the other phase-1 points."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    y = p2.agri_2020.to_numpy()
    oof, rest, Cs = np.full(len(p2), np.nan), np.zeros(len(p1)), []
    for k in range(5):
        tr = (p2.fold != k).to_numpy()
        sc = StandardScaler().fit(p2.loc[tr, feats])
        m = LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=StratifiedKFold(5, shuffle=True, random_state=k),
                                 scoring="neg_log_loss", max_iter=5000).fit(sc.transform(p2.loc[tr, feats]), y[tr])
        oof[~tr] = m.predict_proba(sc.transform(p2.loc[~tr, feats]))[:, 1]
        rest += m.predict_proba(sc.transform(p1[feats]))[:, 1] / 5
        Cs.append(float(m.C_[0]))
    f1 = pd.Series(rest, index=p1.pid).copy()
    f1.loc[p2.pid[p2.pid.isin(f1.index)]] = oof[p2.pid.isin(f1.index).to_numpy()]  # phase-2 points keep their OOF value
    return oof, f1.to_numpy(), Cs


def run2(a):
    from sklearn.metrics import roc_auc_score
    b, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p2 = pd.read_csv(b / "phase2.csv")
    p1 = pd.read_csv(b / "phase1.csv")
    aefc = [c for c in p2.columns if c.startswith(("cell_A", "ring_A"))]
    vlm = [f"p_{g}" for g in A.GROUPS]
    p2["f_vlm"] = p2["p_perennial crops"] + p2["p_annual crops"]
    p1["f_vlm"] = p1.f
    res = {}
    for tag, name, feats in (("A", "A (AlphaEarth only)", aefc), ("AV", "A + V (AlphaEarth + VLM-B, main)", aefc + vlm),
                             ("V", "V (VLM-B, as registered)", None)):
        if feats is None:
            oof, f1, Cs = p2.f_vlm.to_numpy(), p1.f_vlm.to_numpy(), None
        else:
            oof, f1, Cs = crossfit(p2, p1, feats)
        q2, q1 = p2.assign(f=oof), p1.assign(f=f1)
        rf = q2[q2.record_forest == 1].copy()
        diag = {"C per fold": Cs}
        for nm, s in (("all phase 2", q2), ("record forest", rf)):
            diag[nm] = {"AUC": roc_auc_score(s.agri_2020, s.f), "AUC (area-weighted)": roc_auc_score(s.agri_2020, s.f, sample_weight=s.w),
                        "corr": float(np.corrcoef(s.agri_2020, s.f)[0, 1]),
                        "mean f (area)": float(np.average(s.f, weights=s.w)), "mean y (area)": float(np.average(s.agri_2020, weights=s.w))}
        o = out / tag
        o.mkdir(exist_ok=True)
        ref = {"owner": "ubnd_xa", "malr3_cat": "production", "region": rf.groupby("region").size().idxmax()}
        P.feasibility(q1[q1.owner.isin(set(rf.owner))], rf, ref).to_csv(o / "feasibility_lambda1.csv", index=False)
        diag["ppi_plus"] = ppi_plus(q1, rf, "rq3", o)
        P.owner_shares(q1, rf, o)
        res[name] = diag
    (out / "explore_b2.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


# ---------- 4. boosting and per-group PPI++ shares (the analysis plan, 25/10/2026, post hoc) ----------
def crossfit_gbm(p2, p1, feats):
    """As crossfit(), with fixed HistGradientBoostingClassifier settings (no tuning)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    y = p2.agri_2020.to_numpy()
    oof, rest = np.full(len(p2), np.nan), np.zeros(len(p1))
    for k in range(5):
        tr = (p2.fold != k).to_numpy()
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=20,
                                           l2_regularization=1.0, random_state=0).fit(p2.loc[tr, feats], y[tr])
        oof[~tr] = m.predict_proba(p2.loc[~tr, feats])[:, 1]
        rest += m.predict_proba(p1[feats])[:, 1] / 5
    f1 = pd.Series(rest, index=p1.pid).copy()
    f1.loc[p2.pid[p2.pid.isin(f1.index)]] = oof[p2.pid.isin(f1.index).to_numpy()]
    return oof, f1.to_numpy()


def share_ppipp(p1, p2, out):
    """Agriculture share by owner group, lambda per group on a 0.01 grid minimising the variance; plus human only."""
    rows = []
    w1, f1, h1 = p1.w1.to_numpy(), p1.f.to_numpy(), p1.stratum.to_numpy()
    w2, y2, f2, c2 = p2.w.to_numpy(), p2.agri_2020.to_numpy(float), p2.f.to_numpy(), p2.cell.to_numpy()
    for o in sorted(set(p2.owner)):
        m1, m2 = (p1.owner == o).to_numpy(), (p2.owner == o).to_numpy()
        den = (w1 * m1).sum()
        best = None
        for lam in np.round(np.arange(0, 1.0001, 0.01), 2):
            r = (lam * (w1 * m1 * f1).sum() + (w2 * m2 * (y2 - lam * f2)).sum()) / den
            v1 = P.strat_var(((lam * w1 * m1 * f1 - w1 * m1 * r) / den)[:, None], h1)[0, 0]
            v2 = P.strat_var(((w2 * m2 * (y2 - lam * f2)) / den)[:, None], c2)[0, 0]
            if best is None or v1 + v2 < best[2]:
                best = (lam, r, v1 + v2)
        lam, r, v = best
        hum = A.ratio(p2, m2 * y2, m2.astype(float))
        rows.append({"owner": o, "n_phase2": int(m2.sum()), "lambda": lam, "ppipp": r, "ppipp_lo": r - A.Z * np.sqrt(v),
                     "ppipp_hi": r + A.Z * np.sqrt(v), "human": hum["est"], "human_lo": hum["est"] - A.Z * hum["se"],
                     "human_hi": hum["est"] + A.Z * hum["se"], "width_ratio": 2 * A.Z * np.sqrt(v) / (2 * A.Z * hum["se"])})
    t = pd.DataFrame(rows)
    t.to_csv(out / "owner_share_ppipp.csv", index=False)
    return t


def run3(a):
    from sklearn.metrics import roc_auc_score
    b, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p2 = pd.read_csv(b / "phase2.csv")
    p1 = pd.read_csv(b / "phase1.csv")
    aefc = [c for c in p2.columns if c.startswith(("cell_A", "ring_A"))]
    vlm = [f"p_{g}" for g in A.GROUPS]
    preds = {"V": (p2["p_perennial crops"] + p2["p_annual crops"]).to_numpy(), "V1": p1.f.to_numpy()}
    oA, fA, _ = crossfit(p2, p1, aefc)
    oAV, fAV, _ = crossfit(p2, p1, aefc + vlm)
    oG, fG = crossfit_gbm(p2, p1, aefc + vlm)
    res = {}
    for tag, o2, o1 in (("V", preds["V"], preds["V1"]), ("A", oA, fA), ("AV", oAV, fAV), ("G", oG, fG)):
        q2, q1 = p2.assign(f=o2), p1.assign(f=o1)
        rf = q2[q2.record_forest == 1].copy()
        q1 = q1[q1.owner.isin(set(rf.owner))]
        o = out / tag
        o.mkdir(exist_ok=True)
        sh = share_ppipp(q1, rf, o)
        res[tag] = {"AUC record forest": roc_auc_score(rf.agri_2020, rf.f), "corr record forest": float(np.corrcoef(rf.agri_2020, rf.f)[0, 1]),
                    "owner shares": sh[["owner", "n_phase2", "lambda", "width_ratio"]].round(3).to_dict("records")}
        if tag == "G":
            res[tag]["ppi_plus logit"] = ppi_plus(q1, rf, "rq3", o)
    (out / "explore_b3.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


# ---------- 5. FDP commodity probabilities and 2020 seasonality (the analysis plan, 25/10/2026, post hoc) ----------
FDP = ("coffee", "rubber", "palm", "cocoa")
MONTHS = [f"ndvi_m{m:02d}" for m in range(1, 13)]
SEAS = MONTHS + ["ndvi_p10", "ndvi_p50", "ndvi_p90", "vv_mean", "vv_sd", "vh_mean", "vh_sd"]


def extra():
    """Local (GEE): FDP 2020 probabilities (cell and ring) and Sentinel-2/-1 seasonality 2020 (cell) at every point."""
    import csv
    import time
    from concurrent.futures import ThreadPoolExecutor
    import ee
    from sample_maps_gee import init
    init()
    fdp = ee.Image.cat([ee.ImageCollection(f"projects/forestdatapartnership/assets/{c}/model_2026a")
                        .filterDate("2020-01-01", "2021-01-01").mosaic().select("probability").rename(f"fdp_{c}") for c in FDP])
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")

    def ndvi_col(d0, d1, geom):
        c = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(d0, d1).filterBounds(geom) \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70)).linkCollection(cs, ["cs_cdf"])
        c = c.map(lambda i: i.updateMask(i.select("cs_cdf").gte(0.6)).normalizedDifference(["B8", "B4"]).rename("ndvi").toFloat())
        empty = ee.ImageCollection([ee.Image.constant(0).rename("ndvi").toFloat().updateMask(0)])  # a month without scenes: no data
        return c.merge(empty)

    def seasonal(geom):
        months = [ndvi_col(f"2020-{m:02d}-01", f"2020-{m + 1:02d}-01" if m < 12 else "2021-01-01", geom).median()
                  .rename(f"ndvi_m{m:02d}") for m in range(1, 13)]
        year = ndvi_col("2020-01-01", "2021-01-01", geom)
        pct = year.reduce(ee.Reducer.percentile([10, 50, 90])).select(["ndvi_p10", "ndvi_p50", "ndvi_p90"])
        s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterDate("2020-01-01", "2021-01-01").filterBounds(geom) \
            .filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")) \
            .select(["VV", "VH"])
        s1s = s1.reduce(ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True)) \
            .select(["VV_mean", "VV_stdDev", "VH_mean", "VH_stdDev"], ["vv_mean", "vv_sd", "vh_mean", "vh_sd"])
        return ee.Image.cat(months + [pct, s1s])
    pts = pd.read_csv(P.V / "points.csv", usecols=["pid", "lon", "lat"])
    out = A.D / "extra2020.csv"
    done = set(pd.read_csv(out, usecols=["pid"]).pid) if out.exists() else set()
    todo = pts[~pts.pid.isin(done)]
    parts = [todo.iloc[i:i + 100] for i in range(0, len(todo), 100)]
    cols = ["pid"] + [f"cell_fdp_{c}" for c in FDP] + [f"ring_fdp_{c}" for c in FDP] + SEAS

    def req(part):
        cells, rings = [], []
        for r in part.itertuples():
            pt = ee.Geometry.Point([float(r.lon), float(r.lat)])
            cells.append(ee.Feature(pt.buffer(50).bounds(), {"pid": int(r.pid)}))
            rings.append(ee.Feature(pt.buffer(300).difference(pt.buffer(100)), {"pid": int(r.pid)}))
        fc_c, fc_r = ee.FeatureCollection(cells), ee.FeatureCollection(rings)
        for k in range(8):
            try:
                gc = fdp.addBands(seasonal(fc_c.geometry())).reduceRegions(fc_c, ee.Reducer.mean(), 10, tileScale=8).getInfo()
                gr = fdp.reduceRegions(fc_r, ee.Reducer.mean(), 10, tileScale=4).getInfo()
                rows = {f["properties"]["pid"]: {"pid": f["properties"]["pid"], **{f"cell_fdp_{c}": f["properties"].get(f"fdp_{c}") for c in FDP},
                                                 **{s: f["properties"].get(s) for s in SEAS}} for f in gc["features"]}
                for f in gr["features"]:
                    rows[f["properties"]["pid"]].update({f"ring_fdp_{c}": f["properties"].get(f"fdp_{c}") for c in FDP})
                return list(rows.values())
            except Exception as e:
                print(f"retry {k + 1}: {str(e)[:100]}", flush=True)
                time.sleep(20 * (k + 1))
        raise RuntimeError("request failed")
    new = not out.exists()
    with open(out, "a", newline="") as f, ThreadPoolExecutor(4) as pool:
        w = csv.DictWriter(f, cols)
        if new:
            w.writeheader()
        for n, rows in enumerate(pool.map(req, parts), 1):
            w.writerows(rows)
            f.flush()
            if n % 10 == 0 or n == len(parts):
                print(f"{n}/{len(parts)}", flush=True)


def bundle3():
    out = A.ROOT / "runs/land_audit/explore_bundle3"
    out.mkdir(parents=True, exist_ok=True)
    b = A.ROOT / "runs/land_audit/explore_bundle2"
    x = pd.read_csv(A.D / "extra2020.csv").drop_duplicates("pid", keep="last")
    for m in MONTHS:  # a month without a clear observation takes the 2020 median
        x[m] = x[m].fillna(x.ndvi_p50)
    x = x.fillna(x.median(numeric_only=True))  # rare gaps (no S1 or no clear S2 at all): column median
    for name in ("phase2", "phase1"):
        t = pd.read_csv(b / f"{name}.csv").merge(x, on="pid", how="left")
        assert t[x.columns[1:]].notna().all().all()
        t.to_csv(out / f"{name}.csv", index=False)
    print(f"bundle3 -> {out}; gaps filled: months {int(pd.read_csv(A.D / 'extra2020.csv')[MONTHS].isna().sum().sum())} values")


def run4(a):
    from sklearn.metrics import roc_auc_score
    b, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p2 = pd.read_csv(b / "phase2.csv")
    p1 = pd.read_csv(b / "phase1.csv")
    aefc = [c for c in p2.columns if c.startswith(("cell_A", "ring_A"))]
    vlm = [f"p_{g}" for g in A.GROUPS]
    fdp = [c for c in p2.columns if "_fdp_" in c]
    res = {}
    for tag, feats in (("F", fdp), ("S", SEAS), ("AV", aefc + vlm), ("ALL", aefc + vlm + fdp + SEAS)):
        o2, o1, Cs = crossfit(p2, p1, feats)
        q2, q1 = p2.assign(f=o2), p1.assign(f=o1)
        rf = q2[q2.record_forest == 1].copy()
        q1 = q1[q1.owner.isin(set(rf.owner))]
        o = out / tag
        o.mkdir(exist_ok=True)
        sh = share_ppipp(q1, rf, o)
        res[tag] = {"C per fold": Cs, "AUC all": roc_auc_score(q2.agri_2020, q2.f), "AUC record forest": roc_auc_score(rf.agri_2020, rf.f),
                    "AUC record forest (area)": roc_auc_score(rf.agri_2020, rf.f, sample_weight=rf.w),
                    "corr record forest": float(np.corrcoef(rf.agri_2020, rf.f)[0, 1]),
                    "ppi_plus logit": ppi_plus(q1, rf, "rq3", o),
                    "owner shares": sh[["owner", "n_phase2", "lambda", "width_ratio"]].round(3).to_dict("records")}
    (out / "explore_b4.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {kk: v[kk] for kk in ("AUC record forest", "corr record forest")} for k, v in res.items()}, indent=1))


# ---------- 6. stratified PPI and high-confidence training (the analysis plan, 25/10/2026, post hoc) ----------
def bundle4():
    """bundle3 plus conf_high: every interpretation of the point says F4 = high and the label did not need consensus."""
    out = A.ROOT / "runs/land_audit/explore_bundle4"
    out.mkdir(parents=True, exist_ok=True)
    b = A.ROOT / "runs/land_audit/explore_bundle3"
    L = pd.concat([pd.read_csv(A.D / f, dtype=str) for f in ("labels_long.csv", "labels_giaolai_long.csv")], ignore_index=True)
    allhigh = L.groupby("ma_diem").f4.agg(lambda s: bool((s.str.lower() == "high").all()))
    fin = pd.read_csv(A.D / "labels_final.csv", dtype=str).set_index("ma_diem")
    d, _ = A.load()
    conf = pd.DataFrame({"pid": d.pid, "conf_high": [int(allhigh.get(m, False) and "consensus" not in str(fin.source.get(m, "")))
                                                     for m in d.ma_diem]})
    p2 = pd.read_csv(b / "phase2.csv").merge(conf, on="pid", how="left", validate="1:1")
    p2.to_csv(out / "phase2.csv", index=False)
    pd.read_csv(b / "phase1.csv").to_csv(out / "phase1.csv", index=False)
    print(f"bundle4 -> {out}; high-confidence phase-2 points {int(p2.conf_high.sum())} of {len(p2)}; "
          f"sources {fin.source.value_counts().to_dict()}")


def crossfit_masked(p2, p1, feats, train_ok):
    """crossfit() with the training rows of every fold restricted to train_ok (predictions still for every point)."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    y = p2.agri_2020.to_numpy()
    oof, rest = np.full(len(p2), np.nan), np.zeros(len(p1))
    for k in range(5):
        te = (p2.fold == k).to_numpy()
        tr = ~te & np.asarray(train_ok, bool)
        sc = StandardScaler().fit(p2.loc[tr, feats])
        m = LogisticRegressionCV(Cs=[0.01, 0.1, 1, 10], cv=StratifiedKFold(5, shuffle=True, random_state=k),
                                 scoring="neg_log_loss", max_iter=5000).fit(sc.transform(p2.loc[tr, feats]), y[tr])
        oof[te] = m.predict_proba(sc.transform(p2.loc[te, feats]))[:, 1]
        rest += m.predict_proba(sc.transform(p1[feats]))[:, 1] / 5
    f1 = pd.Series(rest, index=p1.pid).copy()
    f1.loc[p2.pid[p2.pid.isin(f1.index)]] = oof[p2.pid.isin(f1.index).to_numpy()]
    return oof, f1.to_numpy()


def strat_share(p1, p2, m1, m2):
    """Share of agriculture in a domain: human only, lambda = 0, one lambda (PPI++), lambda per phase-1 stratum (StratPPI)."""
    w1, f1, h1 = p1.w1.to_numpy(), p1.f.to_numpy(), p1.stratum.to_numpy()
    w2, y2, f2, c2, h2 = p2.w.to_numpy(), p2.agri_2020.to_numpy(float), p2.f.to_numpy(), p2.cell.to_numpy(), p2.stratum.to_numpy()
    den = (w1 * m1).sum()
    grid = np.round(np.arange(0, 1.0001, 0.01), 2)

    def parts(lam_of, r):  # per-stratum variance contributions for a lambda per stratum
        out = {}
        for h in np.unique(h1):
            i1, i2 = h1 == h, h2 == h
            lam = lam_of(h)
            v1 = P.strat_var(((lam * w1[i1] * m1[i1] * f1[i1] - w1[i1] * m1[i1] * r) / den)[:, None], h1[i1])[0, 0]
            v2 = P.strat_var(((w2[i2] * m2[i2] * (y2[i2] - lam * f2[i2])) / den)[:, None], c2[i2])[0, 0] if i2.any() else 0.0
            out[h] = v1 + v2
        return out

    def est(lam_of):
        return sum(lam_of(h) * (w1 * m1 * f1)[h1 == h].sum() + (w2 * m2 * (y2 - lam_of(h) * f2))[h2 == h].sum()
                   for h in np.unique(h1)) / den
    res = {}
    r0 = est(lambda h: 0.0)
    res["lambda 0"] = (r0, sum(parts(lambda h: 0.0, r0).values()))
    best = None
    for lam in grid:  # one lambda for the domain (PPI++)
        r = est(lambda h, lam=lam: lam)
        v = sum(parts(lambda h, lam=lam: lam, r).values())
        if best is None or v < best[2]:
            best = (lam, r, v)
    res["PPI++"] = (best[1], best[2], best[0])
    lam_h, r = {h: 0.0 for h in np.unique(h1)}, r0
    for _ in range(3):  # StratPPI: each stratum's lambda minimises its own contribution
        for h in lam_h:
            vals = [parts(lambda hh, lam=lam: lam if hh == h else lam_h[hh], r)[h] for lam in grid]
            lam_h[h] = float(grid[int(np.argmin(vals))])
        r = est(lambda h: lam_h[h])
    res["StratPPI"] = (r, sum(parts(lambda h: lam_h[h], r).values()), lam_h)
    hum = A.ratio(p2, m2 * y2, m2.astype(float))
    res["human"] = (hum["est"], hum["se"] ** 2)
    return res


def run5(a):
    from sklearn.metrics import roc_auc_score
    b, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p2 = pd.read_csv(b / "phase2.csv")
    p1 = pd.read_csv(b / "phase1.csv")
    feats = [c for c in p2.columns if c.startswith(("cell_A", "ring_A"))] + [f"p_{g}" for g in A.GROUPS]
    preds = {"AV": crossfit(p2, p1, feats)[:2], "H (high-confidence training)": crossfit_masked(p2, p1, feats, p2.conf_high == 1)}
    rows, diag = [], {"high-confidence points": int(p2.conf_high.sum())}
    for tag, (o2, o1) in preds.items():
        q2, q1 = p2.assign(f=o2), p1.assign(f=o1)
        rf = q2[q2.record_forest == 1].copy()
        q1 = q1[q1.owner.isin(set(rf.owner))]
        diag[tag] = {"AUC record forest": roc_auc_score(rf.agri_2020, rf.f), "corr record forest": float(np.corrcoef(rf.agri_2020, rf.f)[0, 1])}
        domains = {"all record forest": (np.ones(len(q1), bool), np.ones(len(rf), bool))}
        domains.update({o: ((q1.owner == o).to_numpy(), (rf.owner == o).to_numpy()) for o in sorted(set(rf.owner))})
        for dname, (m1, m2) in domains.items():
            res = strat_share(q1, rf, m1.astype(float), m2.astype(float))
            se_h = np.sqrt(res["human"][1])
            for meth, v in res.items():
                rows.append({"predictor": tag, "domain": dname, "n_phase2": int(m2.sum()), "method": meth, "est": v[0],
                             "lo": v[0] - A.Z * np.sqrt(v[1]), "hi": v[0] + A.Z * np.sqrt(v[1]), "width_vs_human": np.sqrt(v[1]) / se_h,
                             "lambda": v[2] if meth == "PPI++" else (json.dumps(v[2]) if meth == "StratPPI" else None)})
    t = pd.DataFrame(rows)
    t.to_csv(out / "strat_ppi_shares.csv", index=False)
    (out / "explore_b5.json").write_text(json.dumps(diag, indent=1), encoding="utf-8")
    print(json.dumps(diag, indent=1))
    print(t[t.method != "human"].pivot_table(index=["predictor", "domain"], columns="method", values="width_vs_human").round(3).to_string())


# ---------- 7. CNN on the VLM-B tiles (the analysis plan, 25/10/2026, post hoc; trained by server/cnn_b_train.py) ----------
def cnnpack():
    """Local: X uint8 [n, 24, 64, 64] from vlmb/tiles (per year: true RGB, false RGB), y and fold for the phase-2 points."""
    from PIL import Image
    pts = pd.read_csv(P.V / "points.csv", usecols=["pid"])
    lab = pd.read_csv(A.ROOT / "runs/land_audit/explore_bundle4/phase2.csv", usecols=["pid", "agri_2020", "fold"])
    t = pts.merge(lab, on="pid", how="left")
    X = np.zeros((len(t), 24, 64, 64), np.uint8)
    for i, p in enumerate(t.pid):
        with Image.open(P.V / "tiles" / f"{p}.png") as im:
            a = np.asarray(im.convert("RGB"))  # 128 x 256 x 3: row 0 true colour, row 1 false colour; 4 years
        for j in range(4):
            X[i, 6 * j:6 * j + 3] = a[:64, 64 * j:64 * (j + 1)].transpose(2, 0, 1)
            X[i, 6 * j + 3:6 * j + 6] = a[64:, 64 * j:64 * (j + 1)].transpose(2, 0, 1)
    out = A.ROOT / "runs/land_audit/cnn_b"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "cnn_b.npz", pid=t.pid.to_numpy(), X=X, y=t.agri_2020.fillna(-1).astype(int).to_numpy(),
             fold=t.fold.fillna(-1).astype(int).to_numpy())
    print(f"cnn_b.npz: {X.shape}, labelled {int((t.agri_2020.notna()).sum())}")


def bundle5():
    out = A.ROOT / "runs/land_audit/explore_bundle5"
    out.mkdir(parents=True, exist_ok=True)
    b = A.ROOT / "runs/land_audit/explore_bundle4"
    c = pd.read_csv(A.ROOT / "runs/land_audit/cnn_b/preds_cnn.csv")[["pid", "f_cnn"]]
    for name in ("phase2", "phase1"):
        t = pd.read_csv(b / f"{name}.csv").merge(c, on="pid", how="left", validate="1:1")
        assert t.f_cnn.notna().all()
        t.to_csv(out / f"{name}.csv", index=False)
    print("bundle5 ->", out)


def run6(a):
    from sklearn.metrics import roc_auc_score
    b, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p2 = pd.read_csv(b / "phase2.csv")
    p1 = pd.read_csv(b / "phase1.csv")
    feats = [c for c in p2.columns if c.startswith(("cell_A", "ring_A"))] + [f"p_{g}" for g in A.GROUPS]
    o2av, o1av = crossfit(p2, p1, feats)[:2]
    preds = {"C (CNN)": (p2.f_cnn.to_numpy(), p1.f_cnn.to_numpy()), "AV": (o2av, o1av),
             "C+AV (mean)": ((p2.f_cnn.to_numpy() + o2av) / 2, (p1.f_cnn.to_numpy() + o1av) / 2)}
    rows, diag = [], {}
    for tag, (o2, o1) in preds.items():
        q2, q1 = p2.assign(f=o2), p1.assign(f=o1)
        rf = q2[q2.record_forest == 1].copy()
        q1 = q1[q1.owner.isin(set(rf.owner))]
        diag[tag] = {"AUC all": roc_auc_score(q2.agri_2020, q2.f), "AUC record forest": roc_auc_score(rf.agri_2020, rf.f),
                     "corr record forest": float(np.corrcoef(rf.agri_2020, rf.f)[0, 1]),
                     "ppi_plus logit": ppi_plus(q1, rf, tag.split(" ")[0].replace("+", "p"), out)}
        domains = {"all record forest": (np.ones(len(q1), bool), np.ones(len(rf), bool))}
        domains.update({o: ((q1.owner == o).to_numpy(), (rf.owner == o).to_numpy()) for o in sorted(set(rf.owner))})
        for dname, (m1, m2) in domains.items():
            res = strat_share(q1, rf, m1.astype(float), m2.astype(float))
            se_h = np.sqrt(res["human"][1])
            for meth, v in res.items():
                rows.append({"predictor": tag, "domain": dname, "n_phase2": int(m2.sum()), "method": meth, "est": v[0],
                             "lo": v[0] - A.Z * np.sqrt(v[1]), "hi": v[0] + A.Z * np.sqrt(v[1]), "width_vs_human": np.sqrt(v[1]) / se_h})
    t = pd.DataFrame(rows)
    t.to_csv(out / "cnn_strat_ppi_shares.csv", index=False)
    (out / "explore_b6.json").write_text(json.dumps(diag, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: v[kk] for kk in ("AUC all", "AUC record forest", "corr record forest")} | {"lambda_hat": v["ppi_plus logit"]["lambda_hat"]}
                      for k, v in diag.items()}, indent=1))
    print(t[t.method.isin(["lambda 0", "StratPPI"])].pivot_table(index=["domain"], columns=["predictor", "method"],
                                                                 values="width_vs_human").round(3).to_string())


# ---------- 8. canopy height (the analysis plan, 25/10/2026, post hoc) ----------
CHM = [f"{z}_chm_{s}" for z in ("cell", "ring") for s in ("mean", "p50", "p90", "gt5")]


def chm():
    """Local (GEE): Meta/WRI 1 m canopy height in the 1-ha cell (1 m) and the 100-300 m ring (3 m) of every point."""
    import csv
    import time
    from concurrent.futures import ThreadPoolExecutor
    import ee
    from sample_maps_gee import init
    init()
    h = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight").mosaic().select("cover_code").toFloat()
    img = h.rename("h").addBands(h.gt(5).rename("gt5"))
    red = ee.Reducer.mean().combine(ee.Reducer.percentile([50, 90]), sharedInputs=True)
    pts = pd.read_csv(P.V / "points.csv", usecols=["pid", "lon", "lat"])
    out = A.D / "chm.csv"
    done = set(pd.read_csv(out, usecols=["pid"]).pid) if out.exists() else set()
    todo = pts[~pts.pid.isin(done)]
    parts = [todo.iloc[i:i + 100] for i in range(0, len(todo), 100)]

    def req(part):
        cells, rings = [], []
        for r in part.itertuples():
            pt = ee.Geometry.Point([float(r.lon), float(r.lat)])
            cells.append(ee.Feature(pt.buffer(50).bounds(), {"pid": int(r.pid)}))
            rings.append(ee.Feature(pt.buffer(300).difference(pt.buffer(100)), {"pid": int(r.pid)}))
        for k in range(8):
            try:
                rows = {}
                for zone, fc, scale in (("cell", cells, 1), ("ring", rings, 3)):
                    got = img.reduceRegions(ee.FeatureCollection(fc), red, scale, tileScale=8).getInfo()
                    for f in got["features"]:
                        p = f["properties"]
                        rows.setdefault(p["pid"], {"pid": p["pid"]}).update({
                            f"{zone}_chm_mean": p.get("h_mean"), f"{zone}_chm_p50": p.get("h_p50"), f"{zone}_chm_p90": p.get("h_p90"),
                            f"{zone}_chm_gt5": p.get("gt5_mean")})
                return list(rows.values())
            except Exception as e:
                print(f"retry {k + 1}: {str(e)[:100]}", flush=True)
                time.sleep(20 * (k + 1))
        raise RuntimeError("request failed")
    new = not out.exists()
    with open(out, "a", newline="") as f, ThreadPoolExecutor(4) as pool:
        w = csv.DictWriter(f, ["pid"] + CHM)
        if new:
            w.writeheader()
        for n, rows in enumerate(pool.map(req, parts), 1):
            w.writerows(rows)
            f.flush()
            if n % 10 == 0 or n == len(parts):
                print(f"{n}/{len(parts)}", flush=True)


def bundle6():
    out = A.ROOT / "runs/land_audit/explore_bundle6"
    out.mkdir(parents=True, exist_ok=True)
    b = A.ROOT / "runs/land_audit/explore_bundle5"
    c = pd.read_csv(A.D / "chm.csv").drop_duplicates("pid", keep="last")
    print("missing values per column:", c[CHM].isna().sum().to_dict())
    c[CHM] = c[CHM].fillna(c[CHM].median())
    for name in ("phase2", "phase1"):
        t = pd.read_csv(b / f"{name}.csv").merge(c, on="pid", how="left", validate="1:1")
        assert t[CHM].notna().all().all()
        t.to_csv(out / f"{name}.csv", index=False)
    print("bundle6 ->", out)


def run7(a):
    from sklearn.metrics import roc_auc_score
    b, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p2 = pd.read_csv(b / "phase2.csv")
    p1 = pd.read_csv(b / "phase1.csv")
    av = [c for c in p2.columns if c.startswith(("cell_A", "ring_A"))] + [f"p_{g}" for g in A.GROUPS]
    preds = {"H (canopy height)": crossfit(p2, p1, CHM)[:2], "AV": crossfit(p2, p1, av)[:2], "AV+H": crossfit(p2, p1, av + CHM)[:2]}
    rows, diag = [], {}
    for tag, (o2, o1) in preds.items():
        q2, q1 = p2.assign(f=o2), p1.assign(f=o1)
        rf = q2[q2.record_forest == 1].copy()
        q1 = q1[q1.owner.isin(set(rf.owner))]
        diag[tag] = {"AUC all": roc_auc_score(q2.agri_2020, q2.f), "AUC record forest": roc_auc_score(rf.agri_2020, rf.f),
                     "corr record forest": float(np.corrcoef(rf.agri_2020, rf.f)[0, 1]),
                     "ppi_plus logit": ppi_plus(q1, rf, tag.split(" ")[0].replace("+", "p"), out)}
        domains = {"all record forest": (np.ones(len(q1), bool), np.ones(len(rf), bool))}
        domains.update({o: ((q1.owner == o).to_numpy(), (rf.owner == o).to_numpy()) for o in sorted(set(rf.owner))})
        for dname, (m1, m2) in domains.items():
            res = strat_share(q1, rf, m1.astype(float), m2.astype(float))
            se_h = np.sqrt(res["human"][1])
            for meth, v in res.items():
                rows.append({"predictor": tag, "domain": dname, "n_phase2": int(m2.sum()), "method": meth, "est": v[0],
                             "lo": v[0] - A.Z * np.sqrt(v[1]), "hi": v[0] + A.Z * np.sqrt(v[1]), "width_vs_human": np.sqrt(v[1]) / se_h})
    t = pd.DataFrame(rows)
    t.to_csv(out / "chm_strat_ppi_shares.csv", index=False)
    (out / "explore_b7.json").write_text(json.dumps(diag, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: v[kk] for kk in ("AUC all", "AUC record forest", "corr record forest")} | {"lambda_hat": v["ppi_plus logit"]["lambda_hat"]}
                      for k, v in diag.items()}, indent=1))
    print(t[t.method == "StratPPI"].pivot_table(index="domain", columns="predictor", values="width_vs_human").round(3).to_string())


def run(a):
    b, out = Path(a.bundle), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p2 = pd.read_csv(b / "phase2.csv")
    p1 = pd.read_csv(b / "phase1.csv")
    p2["f"] = p2["p_perennial crops"] + p2["p_annual crops"]
    rf = p2[p2.record_forest == 1].copy()
    res = {"ppi_plus": ppi_plus(p1, rf, "groups_as_registered", out)}
    small = {o for o, n in rf.owner.value_counts().items() if n < 30}
    m1, m2 = p1.copy(), rf.copy()
    m1.loc[m1.owner.isin(small), "owner"] = m2.loc[m2.owner.isin(small), "owner"] = "nho_gop"
    res["ppi_plus_merged"] = {"merged": sorted(small), **ppi_plus(m1, m2, "small_groups_merged", out)}
    res["gft2020"] = gft_check(p2, out)
    (out / "explore_b.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gft", "bundle", "run", "aef", "bundle2", "run2", "run3", "extra", "bundle3", "run4",
                                     "bundle4", "run5", "cnnpack", "bundle5", "run6", "chm", "bundle6", "run7"])
    ap.add_argument("--bundle")
    ap.add_argument("--out")
    a = ap.parse_args()
    {"gft": gft, "bundle": bundle, "run": lambda: run(a), "aef": aef, "bundle2": bundle2, "run2": lambda: run2(a),
     "run3": lambda: run3(a), "extra": extra, "bundle3": bundle3, "run4": lambda: run4(a), "bundle4": bundle4,
     "run5": lambda: run5(a), "cnnpack": cnnpack, "bundle5": bundle5, "run6": lambda: run6(a), "chm": chm, "bundle6": bundle6,
     "run7": lambda: run7(a)}[a.what]()


if __name__ == "__main__":
    main()
