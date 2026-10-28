"""Paper B: the planned estimates (the analysis plan) from the final reference labels.

    .venv/Scripts/python paper/land_audit/scripts/analysis_b.py

RQ1 land use behind each record class, RQ2 agreement of the record and the global maps with the reference, interpreter
variability, the human-only part of RQ3, and the reports required by §8. The cross-PPI part of RQ3 needs the VLM-B
predictions of the 20,000 phase-1 points (§6) and is not here.

Estimator (Supplementary S1): two-phase sampling for stratification (Cochran 1977, ch. 12). Phase-1 stratum h (record
class x malr3) has area A_h and n1_h points; its cell c (x GFC2020 V4) has n1_hc of them and n2_c phase-2 points drawn by
simple random sampling. Per stratum, with W_c = n1_hc / n1_h:
    ybar_h = sum_c W_c ybar_c
    v1_h   = (sum_c W_c s2_c + sum_c W_c (ybar_c - ybar_h)^2) / n1_h      (phase 1)
    v2_h   = sum_c W_c s2_c (n1_hc / n2_c - 1) / n1_h                     (phase 2)
and over strata Ybar = sum_h P_h ybar_h, V = sum_h P_h^2 (v1_h + v2_h), P_h = A_h / A. Ratios (shares inside a domain,
shares that leave out F1 = unknown, user's and producer's accuracy) use the linearised residual y - R x (Stehman 2014).
Outputs: runs/land_audit/analysis_b.json and the CSV tables in runs/land_audit/analysis_b/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/interim/land_audit"
OUT = ROOT / "runs/land_audit/analysis_b"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from owner_groups import owner_group  # noqa: E402  (fixed before the labels were joined)

Z = 1.959964
GROUPS = ["natural forest", "planted forest", "forest land without trees", "perennial crops", "annual crops", "other"]
REGION = {  # the six socio-economic regions of the General Statistics Office, by province code (2020 boundaries)
    **{c: "Northern midlands and mountains" for c in (2, 4, 6, 8, 10, 11, 12, 14, 15, 17, 19, 20, 24, 25)},
    **{c: "Red River Delta" for c in (1, 22, 26, 27, 30, 31, 33, 34, 35, 36, 37)},
    **{c: "North Central and Central Coast" for c in (38, 40, 42, 44, 45, 46, 48, 49, 51, 52, 54, 56, 58, 60)},
    **{c: "Central Highlands" for c in (62, 64, 66, 67, 68)},
    **{c: "Southeast" for c in (70, 72, 74, 75, 77, 79)},
    **{c: "Mekong River Delta" for c in (80, 82, 83, 84, 86, 87, 89, 91, 92, 93, 94, 95, 96)}}


def group_of(code):
    if not isinstance(code, str):
        return None
    k, i = code[0], int(code[1:])
    if k == "R":
        return GROUPS[(i - 1) // 2]
    if k == "N":
        return "perennial crops" if i <= 8 else "annual crops"
    return "other"


# ---------- data ----------
def load():
    key = pd.read_csv(D / "phase2_key.csv", dtype={"ma_diem": str, "churung": str})
    key = key[key.ma_diem.str.startswith("B")]
    lab = pd.read_csv(D / "labels_final.csv", dtype={"f3": str, "f7": str})
    maps = pd.read_csv(D / "phase2_maps.csv").drop(columns=["gfcV4_cog", "gfcV4"])
    d = key.merge(lab, on="ma_diem", validate="1:1").merge(maps, on="pid", validate="1:1")
    terr = D / "phase1_terrain.csv"
    if terr.exists():
        d = d.merge(pd.read_csv(terr), on="pid", how="left", validate="1:1")
    p1 = pd.read_csv(D / "phase1_points.csv", usecols=["pid", "stratum"]).merge(
        pd.read_csv(D / "phase1_points_gfcV4.csv"), on="pid")
    p1["cell"] = p1.stratum + np.where(p1.gfcV4 == 1, "|gfc_forest", "|gfc_nonforest")
    strata = json.loads((D / "phase1_strata.json").read_text(encoding="utf-8"))["strata"]
    area = pd.Series({h: s["area_ha"] for h, s in strata.items()})
    n1h = p1.groupby("stratum").size()
    n1c = p1.groupby("cell").size()
    assert n1h.sort_index().equals(pd.Series({h: s["n1"] for h, s in strata.items()}).sort_index().astype(n1h.dtype))
    d["n1_h"] = d.stratum.map(n1h)
    d["n1_c"] = d.cell.map(n1c)
    d["n2_c"] = d.cell.map(d.groupby("cell").size())
    d["P_h"] = d.stratum.map(area / area.sum())
    d["w"] = d.stratum.map(area) * d.n1_c / d.n1_h / d.n2_c  # ha represented by each point
    d["record"] = d.stratum.str.split("|").str[0]
    d["malr3_cat"] = d.stratum.str.split("|").str[1]
    d["record_forest"] = (d.record != "nonforest").astype(int)
    d["owner"] = d.churung.map(owner_group)
    d["region"] = d.matinh.astype(int).map(REGION)
    d["lu20"] = d.f3.map(group_of)
    d["lu_now"] = d.f7.map(group_of)
    d["known"] = (d.f1 != "unknown").astype(int)
    return d, area


# ---------- estimators ----------
def mean_var(d, z):
    """Population mean (per ha of the whole population) of z and its phase-1 / phase-2 variance components."""
    t = d[["stratum", "cell", "n1_h", "n1_c", "n2_c", "P_h"]].assign(z=np.asarray(z, float))
    c = t.groupby("cell").agg(stratum=("stratum", "first"), n1_h=("n1_h", "first"), n1_c=("n1_c", "first"),
                              n2_c=("n2_c", "first"), P_h=("P_h", "first"), m=("z", "mean"), s2=("z", "var"))
    c["W"] = c.n1_c / c.n1_h
    est = v1 = v2 = 0.0
    for _, g in c.groupby("stratum"):
        yh = (g.W * g.m).sum()
        n1 = g.n1_h.iloc[0]
        P = g.P_h.iloc[0]
        est += P * yh
        v1 += P ** 2 * ((g.W * g.s2).sum() + (g.W * (g.m - yh) ** 2).sum()) / n1
        v2 += P ** 2 * (g.W * g.s2 * (g.n1_c / g.n2_c - 1)).sum() / n1
    return est, v1, v2


def ratio(d, y, x):
    """R = Ybar / Xbar with its standard error (linearisation), plus the phase-1 and phase-2 parts."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    my, _, _ = mean_var(d, y)
    mx, _, _ = mean_var(d, x)
    if mx == 0:
        return dict(est=np.nan, se=np.nan, se1=np.nan, se2=np.nan, n=0)
    r = my / mx
    _, v1, v2 = mean_var(d, y - r * x)
    return dict(est=r, se=np.sqrt(v1 + v2) / mx, se1=np.sqrt(v1) / mx, se2=np.sqrt(v2) / mx, n=int((x > 0).sum()))


def row(name, res, scale=1.0):
    return {"item": name, "est": res["est"] * scale, "lo": (res["est"] - Z * res["se"]) * scale,
            "hi": (res["est"] + Z * res["se"]) * scale, "se": res["se"] * scale, "se_phase1": res["se1"] * scale,
            "se_phase2": res["se2"] * scale, "n": res["n"]}


# ---------- RQ1 ----------
def rq1(d, area):
    A = area.sum()
    domains = {"all": d.index == d.index}
    for r in ("natural", "planted", "nonforest"):
        domains[f"record={r}"] = (d.record == r).to_numpy()
    domains["record=natural+planted"] = (d.record_forest == 1).to_numpy()
    for h in area.index:
        domains[f"stratum={h}"] = (d.stratum == h).to_numpy()
    dom_area = {k: area[sorted(set(d.stratum[m]))].sum() for k, m in domains.items()}  # domains are unions of strata
    rows = []
    for dom, m in domains.items():
        x = m.astype(float)
        a_dom = dom_area[dom]
        for g in GROUPS:
            for when, col in (("2020", "lu20"), ("now", "lu_now")):
                res = ratio(d, x * (d[col] == g), x)
                rows.append({"domain": dom, "when": when, **row(g, res), "kha": res["est"] * a_dom / 1e3})
        for when, col in (("2020", "f3"), ("now", "f7")):
            for name, codes in (("coffee (N1, N2)", ("N1", "N2")), ("rubber (N3)", ("N3",))):
                res = ratio(d, x * d[col].isin(codes), x)
                rows.append({"domain": dom, "when": when, **row(name, res), "kha": res["est"] * a_dom / 1e3})
        for name, y, xx in (("forest 2020, EUDR/FAO (F1)", x * (d.f1 == "forest"), x * d.known),
                            ("forest 2020, Vietnam definition", x * (d.forest_vn == 1), x * d.forest_vn.notna()),
                            ("agriculture 2020 (F3 in N1-N11)", x * (d.agri_2020 == 1), x),
                            ("agriculture now (F7 in N1-N11)", x * d.f7.str.startswith("N"), x),
                            ("forest 2020 (F1) and agriculture now", x * ((d.f1 == "forest") & d.f7.str.startswith("N")),
                             x * d.known)):
            res = ratio(d, y, xx)
            rows.append({"domain": dom, "when": "", **row(name, res), "kha": res["est"] * a_dom / 1e3})
    t = pd.DataFrame(rows)
    t.insert(1, "area_kha_domain", t.domain.map(dom_area) / 1e3)
    # owner groups: human labels only where the group has >= 30 phase-2 points (the analysis plan)
    own = []
    for rec, rm in (("record forest (natural + planted)", d.record_forest == 1), ("record nonforest", d.record_forest == 0)):
        for o, g in d[rm].groupby("owner"):
            x = (rm & (d.owner == o)).astype(float).to_numpy()
            base = {"record": rec, "owner": o, "n": len(g)}
            if len(g) < 30:
                own.append({**base, "note": "< 30 points: RQ3 only"})
                continue
            for name, y, xx in (("forest 2020, EUDR/FAO (F1)", x * (d.f1 == "forest"), x * d.known),
                                ("agriculture 2020", x * (d.agri_2020 == 1), x),
                                ("natural forest 2020", x * (d.lu20 == "natural forest"), x)):
                r = ratio(d, y, xx)
                own.append({**base, "item": name, "est": r["est"], "lo": r["est"] - Z * r["se"],
                            "hi": r["est"] + Z * r["se"]})
            r = mean_var(d, x)[0] * A
            own.append({**base, "item": "area (kha), estimated", "est": r / 1e3})
    return t, pd.DataFrame(own)


# ---------- RQ2 ----------
def map_calls(d):
    return {"record (natural or planted)": d.record_forest, "GFC2020 V1": d.gfcV1, "GFC2020 V2": d.gfcV2,
            "GFC2020 V3": d.gfcV3, "GFC2020 V4": d.gfcV4, "TMF 2020 (undisturbed or degraded)": d.tmf2020.isin([1, 2]).astype(int),
            "WorldCover 2020 (tree cover)": (d.wc2020 == 10).astype(int)}


def accuracy(d, m, ref, known):
    m, ref, known = np.asarray(m, float), np.nan_to_num(np.asarray(ref, float)), np.asarray(known, float)
    out = {"OA": ratio(d, known * (m == ref), known),
           "UA forest": ratio(d, known * m * ref, known * m), "PA forest": ratio(d, known * m * ref, known * ref),
           "UA nonforest": ratio(d, known * (1 - m) * (1 - ref), known * (1 - m)),
           "PA nonforest": ratio(d, known * (1 - m) * (1 - ref), known * (1 - ref)),
           "map forest share": ratio(d, known * m, known), "reference forest share": ratio(d, known * ref, known)}
    for i, j in ((1, 1), (1, 0), (0, 1), (0, 0)):
        out[f"p(map={i}, ref={j})"] = ratio(d, known * (m == i) * (ref == j), known)
    return out


def rq2(d):
    # F1 = unknown stays out of the denominators (the analysis plan): NaN, like forest_vn already is for those points
    refs = {"EUDR/FAO": (d.f1 == "forest").astype(float).where(d.known == 1), "Vietnam definition": d.forest_vn}
    rows = []
    for rname, ref in refs.items():
        known = ref.notna().astype(float)
        for mname, m in map_calls(d).items():
            for k, res in accuracy(d, m, ref, known).items():
                rows.append({"reference": rname, "map": mname, **row(k, res)})
    # sensitivity: every F1 = unknown point set to forest, then to nonforest (the analysis plan)
    sens = []
    for fill in (1.0, 0.0):
        ref = np.where(d.f1 == "unknown", fill, (d.f1 == "forest").astype(float))
        one = np.ones(len(d))
        sens.append({"unknown as": "forest" if fill else "nonforest", "map": "reference",
                     **row("forest share", ratio(d, ref, one))})
        for mname, m in map_calls(d).items():
            sens.append({"unknown as": "forest" if fill else "nonforest", "map": mname,
                         **row("OA", ratio(d, (np.asarray(m) == ref).astype(float), one))})
    # GFC2020 version changes, by actual land use 2020
    ver = []
    pairs = (("V1", "V2"), ("V2", "V3"), ("V3", "V4"), ("V1", "V4"))
    for a, b in pairs:
        ch = (d[f"gfc{a}"] != d[f"gfc{b}"]).astype(float)
        gain = ((d[f"gfc{a}"] == 0) & (d[f"gfc{b}"] == 1)).astype(float)
        loss = ((d[f"gfc{a}"] == 1) & (d[f"gfc{b}"] == 0)).astype(float)
        for g in ["all"] + GROUPS + ["coffee (N1, N2)", "rubber (N3)"]:
            x = (np.ones(len(d)) if g == "all" else d.f3.isin(["N1", "N2"]) if g.startswith("coffee")
                 else (d.f3 == "N3") if g.startswith("rubber") else (d.lu20 == g)).astype(float)
            rc, rg, rl = ratio(d, x * ch, x), ratio(d, x * gain, x), ratio(d, x * loss, x)
            # of the changed area with a known reference: how much moved towards the reference
            ref = (d.f1 == "forest").astype(float)
            to_ref = ratio(d, x * ch * d.known * (d[f"gfc{b}"] == ref), x * ch * d.known)
            ver.append({"from": a, "to": b, "land use 2020": g, "n": int(x.sum()), "changed": rc["est"],
                        "changed_lo": rc["est"] - Z * rc["se"], "changed_hi": rc["est"] + Z * rc["se"],
                        "nonforest->forest": rg["est"], "forest->nonforest": rl["est"],
                        "changed towards reference": to_ref["est"]})
    # forest calls on land actually under tree crops (EUDR commission)
    com = []
    for g, x in (("perennial crops", d.lu20 == "perennial crops"), ("coffee (N1, N2)", d.f3.isin(["N1", "N2"])),
                 ("rubber (N3)", d.f3 == "N3"), ("annual crops", d.lu20 == "annual crops")):
        x = x.astype(float)
        for mname, m in map_calls(d).items():
            com.append({"land use 2020": g, "map": mname, **row("called forest", ratio(d, x * m, x))})
    return pd.DataFrame(rows), pd.DataFrame(sens), pd.DataFrame(ver), pd.DataFrame(com)


# ---------- interpreter variability ----------
def kappa(a, b):
    cats = sorted(set(a) | set(b))
    po = np.mean(np.asarray(a) == np.asarray(b))
    pe = sum(np.mean(np.asarray(a) == c) * np.mean(np.asarray(b) == c) for c in cats)
    return po, (po - pe) / (1 - pe) if pe < 1 else np.nan


def interpreters(d):
    L = pd.read_csv(D / "labels_long.csv", dtype=str)
    L = L[L.kind == "main"]
    R = pd.read_csv(D / "labels_giaolai_long.csv", dtype=str)
    key = pd.read_csv(D / "phase2_key.csv", dtype=str)
    dbl = key[key.nguoi_2.notna() & key.ma_diem.str.startswith("B")].ma_diem
    out = {}
    for name, pairs in (("double coding (random 20%)", L[L.ma_diem.isin(dbl)]),
                        ("re-interpretation (low confidence)", pd.concat([L[L.ma_diem.isin(R.ma_diem)], R]))):
        p = pairs.sort_values("nguoi_gan").groupby("ma_diem")
        a = p.nth(0).set_index("ma_diem")
        b = p.nth(1).set_index("ma_diem").reindex(a.index)
        res = {"points": len(a)}
        for f, fn in (("F1", lambda s: s.f1), ("F3 code", lambda s: s.f3), ("F3 land-use group", lambda s: s.f3.map(group_of))):
            po, k = kappa(fn(a).fillna("-"), fn(b).fillna("-"))
            n = len(a)
            res[f] = {"agreement": po, "lo": po - Z * np.sqrt(po * (1 - po) / n), "hi": po + Z * np.sqrt(po * (1 - po) / n),
                      "kappa": k}
        if name.startswith("double"):
            # agreement weighted by the design weight of the point
            w = d.set_index("ma_diem").w.reindex(a.index)
            res["F1 agreement, area-weighted"] = float(np.average(a.f1 == b.f1, weights=w))
            res["F1 agreement by first interpreter's F1"] = {c: float((b.f1[a.f1 == c] == c).mean()) for c in ("forest", "nonforest")}
            out["effects"] = interpreter_effects(d, a, b)
        out[name] = res
    return out


def interpreter_effects(d, a, b):
    """Measurement model in the spirit of Stehman et al. (2022, RSE 269:112806): label = truth + interpreter effect.
    The effects b_j (sum to zero) come from the paired differences on the random double-coded points; their variance
    sigma_b^2 adds sigma_b^2 * sum_j omega_j^2 to the variance of an estimate, omega_j being the share of the estimator
    weight carried by labels of interpreter j (consensus labels carry none)."""
    J = sorted(set(a.nguoi_gan) | set(b.nguoi_gan))
    res = {}
    for yname, f in (("forest 2020 (F1)", lambda s: (s.f1 == "forest").astype(float).where(s.f1 != "unknown")),
                     ("agriculture 2020 (F3)", lambda s: s.f3.str.startswith("N").astype(float))):
        ya, yb = f(a), f(b)
        ok = ya.notna() & yb.notna()
        X = np.zeros((ok.sum(), len(J)))
        for i, (ja, jb) in enumerate(zip(a.nguoi_gan[ok], b.nguoi_gan[ok])):
            X[i, J.index(ja)] += 1
            X[i, J.index(jb)] -= 1
        diff = (ya - yb)[ok].to_numpy()
        XtX_pinv = np.linalg.pinv(X.T @ X)  # rows sum to 0, so the minimum-norm solution is the sum-to-zero one
        beta = XtX_pinv @ X.T @ diff
        resid = diff - X @ beta
        s2 = resid @ resid / max(1, len(diff) - (len(J) - 1))
        cov = s2 * XtX_pinv
        # E[sum_j b_hat_j^2] = (J - 1) sigma_b^2 + trace(cov) for sum-to-zero deviations of iid effects
        sb2 = max(0.0, (beta @ beta - np.trace(cov)) / (len(J) - 1))
        y = f(d)
        x =(d.known if yname.startswith("forest") else pd.Series(1.0, index=d.index)).to_numpy(float)
        wx = d.w.to_numpy() * x
        omega = {j: wx[(d["from"] == j).to_numpy()].sum() / wx.sum() for j in J}
        est = ratio(d, np.nan_to_num(y.to_numpy()) * x, x)
        v_int = sb2 * sum(o ** 2 for o in omega.values())
        res[yname] = {"pairs": int(ok.sum()), "effects": dict(zip(J, beta.round(4))),
                      "effect_se": dict(zip(J, np.sqrt(np.diag(cov)).round(4))), "sigma_b2": sb2, "omega": omega, "estimate": est["est"], "se_sampling": est["se"],
                      "se_total": float(np.sqrt(est["se"] ** 2 + v_int)), "interpreter_share_of_variance":
                      v_int / (est["se"] ** 2 + v_int)}
    return res


# ---------- RQ3, human labels only ----------
def logit_design(d, y, X, w):
    """Design-weighted logistic regression; linearisation variance with the phase-2 cells as strata (the phase-1
    component of the cell weights is left out)."""
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ beta))
        H = (X * (w * p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, X.T @ (w * (y - p)))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ beta))
    H = (X * (w * p * (1 - p))[:, None]).T @ X
    u = X * (w * (y - p))[:, None]
    G = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(d.cell):
        m = (d.cell == c).to_numpy()
        n = m.sum()
        uc = u[m] - u[m].mean(0)
        G += n / (n - 1) * uc.T @ uc
    Hi = np.linalg.inv(H)
    return beta, Hi @ G @ Hi


def rq3_human(d):
    s = d[(d.record_forest == 1) & d.elev.notna()].copy() if "elev" in d else None
    if s is None or s.empty:
        return pd.DataFrame(), {}
    ref = {"owner": "ubnd_xa", "malr3_cat": "production",
           "region": s.groupby("region").w.size().idxmax()}
    cols, X = ["intercept"], [np.ones(len(s))]
    for v, r in ref.items():
        for lvl in sorted(set(s[v]) - {r}):
            cols.append(f"{v}={lvl}")
            X.append((s[v] == lvl).to_numpy(float))
    cols += ["elevation (per 100 m)", "slope (per 10 degrees)"]
    X += [s.elev.to_numpy() / 100, s.slope.to_numpy() / 10]
    X = np.column_stack(X)
    y = s.agri_2020.to_numpy(float)
    beta, cov = logit_design(s, y, X, s.w.to_numpy())
    se = np.sqrt(np.diag(cov))
    t = pd.DataFrame({"term": cols, "coef": beta, "se": se})
    # national parks and nature reserves sit almost only in special-use forest, so the two terms are nearly aliased:
    # also report their sum (a VQG/KBT owner in special-use forest against a commune-owned production forest)
    i, j = cols.index("owner=vuon_quoc_gia_khu_bao_ton"), cols.index("malr3_cat=special")
    t.loc[len(t)] = ["owner=vuon_quoc_gia_khu_bao_ton + malr3_cat=special", beta[i] + beta[j],
                     np.sqrt(cov[i, i] + cov[j, j] + 2 * cov[i, j])]
    t["lo"], t["hi"] = t.coef - Z * t.se, t.coef + Z * t.se
    t["OR"] = np.exp(t.coef)
    lv = lambda c: (s[c.split("=")[0]] == c.split("=")[1]).to_numpy() if "=" in c and "+" not in c else None  # noqa: E731
    t["n_level"] = [int(lv(c).sum()) if lv(c) is not None else len(s) for c in t.term]
    t["events_level"] = [int(y[lv(c)].sum()) if lv(c) is not None else int(y.sum()) for c in t.term]
    t["ci_excludes_0"] = (t.lo > 0) | (t.hi < 0)
    t["small_group"] = t.n_level < 30  # too few human labels to read alone (the analysis plan: such groups go through PPI)
    overlap = float(((s.owner == "vuon_quoc_gia_khu_bao_ton") & (s.malr3_cat == "special")).sum()
                    / (s.owner == "vuon_quoc_gia_khu_bao_ton").sum())
    return t, {"n": len(s), "events": int(y.sum()), "reference_levels": ref,
               "share of VQG/KBT points in special-use forest": overlap}


# ---------- reports required by §8 ----------
def required(d):
    L = pd.read_csv(D / "labels_long.csv", dtype=str)
    L = L[L.kind == "main"]
    low = L[L.f4 == "low"]
    split = lambda s: s.dropna().str.split("; ").explode().str.strip()  # noqa: E731  (multiple-choice fields)
    return {"points per cell": d.groupby("cell").size().to_dict(),
            "low confidence share (all main labels)": len(low) / len(L),
            "low confidence reasons F4b (share of low-confidence labels citing it)":
                (split(low.F4b).value_counts() / len(low)).round(3).to_dict(),
            "F5 issues (share of all main labels citing it)": (split(L.F5).value_counts() / len(L)).round(3).to_dict(),
            "final label source": d.source.value_counts().to_dict(),
            "TMF no-data points (class 0, counted as not forest)": int((d.tmf2020 == 0).sum())}


def main():
    d, area = load()
    OUT.mkdir(parents=True, exist_ok=True)
    t1, own = rq1(d, area)
    acc, sens, ver, com = rq2(d)
    t3, info3 = rq3_human(d)
    for name, t in (("rq1_landuse", t1), ("rq1_owner", own), ("rq2_accuracy", acc), ("rq2_unknown_sensitivity", sens),
                    ("rq2_gfc_versions", ver), ("rq2_tree_crops_called_forest", com), ("rq3_human_logit", t3)):
        t.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")
    summary = {"points": len(d), "population_area_ha": float(area.sum()), "interpreters": interpreters(d),
               "rq3_human": info3, "required": required(d)}
    (ROOT / "runs/land_audit/analysis_b.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=float),
                                                         encoding="utf-8")
    pd.set_option("display.width", 220, "display.max_columns", 20, "display.max_rows", 400)
    f = lambda t: t.round(3).to_string(index=False)  # noqa: E731
    print(f(t1[t1.domain.isin(["all", "record=natural", "record=planted", "record=nonforest"]) & (t1.when != "now")]
            [["domain", "item", "est", "lo", "hi", "kha", "n"]]))
    print(f(acc[acc.item.isin(["OA", "UA forest", "PA forest", "map forest share", "reference forest share"])]
            .pivot_table(index=["reference", "map"], columns="item", values="est", sort=False).reset_index()))
    print(f(ver[ver["land use 2020"] == "all"]))
    print(f(t3))
    print(json.dumps(summary["interpreters"], ensure_ascii=False, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main()
