"""Paper B, RQ3 by cross-prediction-powered inference (the analysis plan; Zrnic & Candès 2024), and VLM-B accuracy.

    .venv/Scripts/python paper/land_audit/scripts/ppi_b.py

Population: phase-1 points that the record calls forest (nggocr 1, 2). f = VLM-B P(agriculture 2020) = P(perennial
crops) + P(annual crops): out of fold for the phase-2 points, the mean of the 5 adapters for the others.
Design-weighted logistic regression, two-phase sample: theta solves
    sum_{phase 1} w1 x (f - sigmoid(x'theta)) + sum_{phase 2} w2 x (y - f) = 0,
w1 = A_h / n1_h (phase-1 stratum h), w2 = the phase-2 weight of analysis_b.py. The rectifier does not depend on theta,
so H = sum w1 p (1 - p) x x' and Var(theta) = H^-1 (V1 + V2) H^-1, with V1 the variance of the phase-1 sum (strata h)
and V2 that of the rectifier (phase-2 cells as strata). Domain shares by owner group use the same two terms.
The human-only estimate (analysis_b.py) is refitted with the same terms for the CI comparison.
Outputs: runs/land_audit/analysis_b/rq3_ppi_logit.csv, rq3_ppi_owner_share.csv, vlmb_accuracy.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_b as A  # noqa: E402
from owner_groups import owner_group  # noqa: E402

V = A.D / "vlmb"
Z = A.Z


def phase1():
    p1 = pd.read_csv(A.D / "phase1_points.csv", usecols=["pid", "stratum", "matinh", "churung"], dtype={"churung": str})
    p1 = p1[~p1.stratum.str.startswith("nonforest")].merge(pd.read_csv(A.D / "phase1_terrain.csv"), on="pid")
    strata = json.loads((A.D / "phase1_strata.json").read_text(encoding="utf-8"))["strata"]
    p1["w1"] = p1.stratum.map({h: s["area_ha"] / s["n1"] for h, s in strata.items()})
    p1["owner"] = p1.churung.map(owner_group)
    p1["malr3_cat"] = p1.stratum.str.split("|").str[1]
    p1["region"] = p1.matinh.astype(int).map(A.REGION)
    pr = pd.read_csv(V / "preds.csv")
    pr["f"] = pr["p_perennial crops"] + pr["p_annual crops"]
    return p1.merge(pr[["pid", "f", "kind"]], on="pid", how="left", validate="1:1")


def design(s, ref, cols=None):
    """Design matrix with the same levels and reference categories as analysis_b.rq3_human."""
    if cols is None:
        cols = ["intercept"] + [f"{v}={l}" for v, r in ref.items() for l in sorted(set(s[v]) - {r})] + \
               ["elevation (per 100 m)", "slope (per 10 degrees)"]
    X = [np.ones(len(s))]
    for c in cols[1:]:
        if "=" in c:
            v, l = c.split("=")
            X.append((s[v] == l).to_numpy(float))
    X += [s.elev.to_numpy() / 100, s.slope.to_numpy() / 10]
    return np.column_stack(X), cols


def strat_var(u, groups):
    """Variance of sum_i u_i (rows) under stratified sampling with replacement within each group."""
    G = np.zeros((u.shape[1], u.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        n = m.sum()
        if n > 1:
            uc = u[m] - u[m].mean(0)
            G += n / (n - 1) * uc.T @ uc
    return G


class NoSolution(Exception):
    """The PPI logit equations have no finite root: for a 0/1 column the rectified target share lies outside [0, 1]."""


def feasibility(p1, p2, ref):
    """Per 0/1 column: VLM-B mean share on phase 1, rectifier, and their sum (the share the fitted model must reproduce).
    With lambda = 1 a sum outside [0, 1] means no finite solution (24/10/2026: found when the planned fit ran)."""
    X1, cols = design(p1, ref)
    X2, _ = design(p2, ref, cols)
    w1, f1 = p1.w1.to_numpy(), p1.f.to_numpy()
    w2, y2, f2 = p2.w.to_numpy(), p2.agri_2020.to_numpy(float), p2.f.to_numpy()
    rows = []
    for j, c in enumerate(cols[:-2]):
        m1, m2 = X1[:, j] == 1, X2[:, j] == 1
        W = (w1 * m1).sum()
        rows.append({"term": c, "n_phase1": int(m1.sum()), "n_phase2": int(m2.sum()), "vlm_share": (w1 * m1 * f1).sum() / W,
                     "rectifier": (w2 * m2 * (y2 - f2)).sum() / W})
    t = pd.DataFrame(rows)
    t["target"] = t.vlm_share + t.rectifier
    t["feasible"] = (t.target > 0) & (t.target < 1)
    return t


def ppi_logit(p1, p2, ref):
    X1, cols = design(p1, ref)
    X2, _ = design(p2, ref, cols)
    w1, f1 = p1.w1.to_numpy(), p1.f.to_numpy()
    w2, y2, f2 = p2.w.to_numpy(), p2.agri_2020.to_numpy(float), p2.f.to_numpy()
    fz = feasibility(p1, p2, ref)
    if not fz.feasible.all():
        raise NoSolution(", ".join(f"{r.term} {r.target:+.3f}" for r in fz[~fz.feasible].itertuples()))
    rect = X2.T @ (w2 * (y2 - f2))
    beta = np.zeros(X1.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X1 @ beta))
        H = (X1 * (w1 * p * (1 - p))[:, None]).T @ X1
        step = np.linalg.solve(H, X1.T @ (w1 * (f1 - p)) + rect)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X1 @ beta))
    H = (X1 * (w1 * p * (1 - p))[:, None]).T @ X1
    V1 = strat_var(X1 * (w1 * (f1 - p))[:, None], p1.stratum.to_numpy())
    V2 = strat_var(X2 * (w2 * (y2 - f2))[:, None], p2.cell.to_numpy())
    Hi = np.linalg.inv(H)
    return beta, Hi @ (V1 + V2) @ Hi, cols


def joint(t, beta, cov, cols, a, b, name):
    i, j = cols.index(a), cols.index(b)
    return {"term": name, "coef": beta[i] + beta[j], "se": np.sqrt(cov[i, i] + cov[j, j] + 2 * cov[i, j])}


def main():
    d, area = A.load()
    p1 = phase1()
    if p1.f.isna().any():
        raise SystemExit(f"{int(p1.f.isna().sum())} phase-1 points have no VLM-B prediction")
    pr = pd.read_csv(V / "preds.csv")
    d = d.merge(pr, on="pid", how="left")
    d["f"] = d["p_perennial crops"] + d["p_annual crops"]
    out = A.ROOT / "runs/land_audit/analysis_b"

    # VLM-B accuracy against the human labels, out of fold (the analysis plan)
    P = d[[f"p_{g}" for g in A.GROUPS]].to_numpy()
    pred = np.array(A.GROUPS)[P.argmax(1)]
    acc = {"points": len(d), "accuracy_6": float((pred == d.lu20).mean()),
           "accuracy_6_area_weighted": float(np.average(pred == d.lu20, weights=d.w)),
           "macro_f1_6": float(f1_score(d.lu20, pred, average="macro", labels=A.GROUPS)),
           "per_class_f1": dict(zip(A.GROUPS, f1_score(d.lu20, pred, average=None, labels=A.GROUPS).round(3).tolist())),
           "confusion (rows human, cols VLM)": pd.DataFrame(confusion_matrix(d.lu20, pred, labels=A.GROUPS),
                                                            index=A.GROUPS, columns=A.GROUPS).to_dict()}
    agri_pred = (d.f >= 0.5).astype(int)
    for name, m in (("all", d.index == d.index), ("record forest", d.record_forest == 1)):
        y, yp, w = d.agri_2020[m], agri_pred[m], d.w[m]
        acc[f"agriculture, {name}"] = {"n": int(m.sum()), "accuracy": float((y == yp).mean()),
                                       "sensitivity": float(yp[y == 1].mean()), "specificity": float(1 - yp[y == 0].mean()),
                                       "mean f (area-weighted)": float(np.average(d.f[m], weights=w)),
                                       "mean y (area-weighted)": float(np.average(y, weights=w))}
    (out / "vlmb_accuracy.json").write_text(json.dumps(acc, indent=1), encoding="utf-8")

    # RQ3: human-only vs cross-PPI, same specification
    p2 = d[(d.record_forest == 1)].copy()
    ref = {"owner": "ubnd_xa", "malr3_cat": "production", "region": p2.groupby("region").size().idxmax()}
    fz = feasibility(p1, p2, ref)
    fz.to_csv(out / "rq3_ppi_feasibility.csv", index=False)
    try:
        beta, cov, cols = ppi_logit(p1, p2, ref)
    except NoSolution as e:  # reported as such; the owner shares below are a linear estimator and always exist
        (out / "rq3_ppi_logit.csv").write_text(
            f"no finite PPI solution (lambda = 1); rectified target share outside [0, 1]: {e}", encoding="utf-8")
        print("PPI logit: no finite solution:", e)
        beta = None
    if beta is not None:
        rq3_logit_table(p1, p2, ref, beta, cov, cols, out)
    owner_shares(p1, p2, out)
    pd.set_option("display.width", 250)
    print(json.dumps({k: v for k, v in acc.items() if "confusion" not in k}, indent=1))
    print(fz.round(3).to_string(index=False))
    print(pd.read_csv(out / "rq3_ppi_owner_share.csv").round(3).to_string(index=False))


def rq3_logit_table(p1, p2, ref, beta, cov, cols, out):
    Xh, _ = design(p2, ref, cols)
    bh, covh = A.logit_design(p2, p2.agri_2020.to_numpy(float), Xh, p2.w.to_numpy())
    rows = []
    for c, b, s, b0, s0 in zip(cols, beta, np.sqrt(np.diag(cov)), bh, np.sqrt(np.diag(covh))):
        rows.append({"term": c, "ppi_coef": b, "ppi_se": s, "human_coef": b0, "human_se": s0})
    jp = joint(None, beta, cov, cols, "owner=vuon_quoc_gia_khu_bao_ton", "malr3_cat=special", "VQG/KBT + special (post hoc)")
    jh = joint(None, bh, covh, cols, "owner=vuon_quoc_gia_khu_bao_ton", "malr3_cat=special", "")
    rows.append({"term": jp["term"], "ppi_coef": jp["coef"], "ppi_se": jp["se"], "human_coef": jh["coef"], "human_se": jh["se"]})
    t = pd.DataFrame(rows)
    for k in ("ppi", "human"):
        t[f"{k}_OR"], t[f"{k}_lo"], t[f"{k}_hi"] = np.exp(t[f"{k}_coef"]), np.exp(t[f"{k}_coef"] - Z * t[f"{k}_se"]), \
            np.exp(t[f"{k}_coef"] + Z * t[f"{k}_se"])
        t[f"{k}_ci_excludes_0"] = (t[f"{k}_coef"] - Z * t[f"{k}_se"] > 0) | (t[f"{k}_coef"] + Z * t[f"{k}_se"] < 0)
    t["se_ratio_ppi_over_human"] = t.ppi_se / t.human_se
    t["n_phase1"] = [len(p1)] + [int((p1[c.split("=")[0]] == c.split("=")[1]).sum()) if "=" in c and "+" not in c
                                 else len(p1) for c in t.term[1:]]
    t["n_phase2"] = [len(p2)] + [int((p2[c.split("=")[0]] == c.split("=")[1]).sum()) if "=" in c and "+" not in c
                                 else len(p2) for c in t.term[1:]]
    t.to_csv(out / "rq3_ppi_logit.csv", index=False)
    print(t.round(3).to_string(index=False))


def owner_shares(p1, p2, out):
    """Agriculture share by owner group on record forest: PPI (difference estimator) vs human labels only."""
    sh = []
    for o in sorted(set(p1.owner)):
        m1, m2 = (p1.owner == o).to_numpy(), (p2.owner == o).to_numpy()
        w1, w2 = p1.w1.to_numpy(), p2.w.to_numpy()
        den = (w1 * m1).sum()
        num = (w1 * m1 * p1.f.to_numpy()).sum() + (w2 * m2 * (p2.agri_2020 - p2.f).to_numpy()).sum()
        r = num / den
        v1 = strat_var(((w1 * m1 * (p1.f.to_numpy() - r)) / den)[:, None], p1.stratum.to_numpy())[0, 0]
        v2 = strat_var(((w2 * m2 * (p2.agri_2020 - p2.f).to_numpy()) / den)[:, None], p2.cell.to_numpy())[0, 0]
        se = np.sqrt(v1 + v2)
        hum = A.ratio(p2, m2 * p2.agri_2020.to_numpy(float), m2.astype(float)) if m2.sum() else {"est": np.nan, "se": np.nan}
        sh.append({"owner": o, "n_phase1": int(m1.sum()), "n_phase2": int(m2.sum()), "ppi": r, "ppi_lo": r - Z * se,
                   "ppi_hi": r + Z * se, "human": hum["est"], "human_lo": hum["est"] - Z * hum["se"],
                   "human_hi": hum["est"] + Z * hum["se"]})
    s = pd.DataFrame(sh)
    s.to_csv(out / "rq3_ppi_owner_share.csv", index=False)


if __name__ == "__main__":
    main()
