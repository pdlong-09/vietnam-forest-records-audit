"""Paper B, checks added for the PeerJ version (post hoc, 28/10/2026): interpreter variance on the headline estimates,
agreement by land-use group, sensitivity to low-confidence labels, and the dates of the reference imagery.

    .venv/Scripts/python paper/land_audit/scripts/robustness_b.py

1. Interpreter variance. The measurement model of analysis_b.interpreter_effects (label = truth + effect b_j of
   interpreter j, sigma_b^2 from the random double-coded pairs) is applied to every headline estimate. For a ratio
   R = sum w y / sum w x whose label-dependent part shifts by g_i b_j(i) at point i, the shift is
   sum_j b_j c_j with c_j = sum_{i in j} w_i g_i / sum_i w_i x_i, so v_int = sigma_b^2 sum_j c_j^2.
   g = x for a share; g = x (2 m - 1) for the overall accuracy of map m (agreement = 1 - m - ref + 2 m ref);
   g = 2 x (m_A - m_B) for a paired difference. Consensus labels carry no interpreter effect.
2. Agreement on the 240 random double-coded points by land-use group (specific agreement 2a / (2a + b + c)) and for
   agriculture yes/no.
3. Low-confidence labels: the estimates recomputed without the points whose final label rests on a single
   interpretation with low confidence, treating them as missing at random within their cell (cell means over the
   remaining points; phase-one counts unchanged).
4. Reference imagery: the first acquisition date noted in F8 (image source) of each main interpretation, against
   31 December 2020.
Outputs: runs/land_audit/robustness_b/*.csv and robustness_b.json.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_b as A  # noqa: E402

OUT = A.ROOT / "runs/land_audit/robustness_b"
J = ["A1", "A2", "A3", "A4"]


def pairs():
    L = pd.read_csv(A.D / "labels_long.csv", dtype=str)
    L = L[L.kind == "main"]
    key = pd.read_csv(A.D / "phase2_key.csv", dtype=str)
    dbl = key[key.nguoi_2.notna() & key.ma_diem.str.startswith("B")].ma_diem
    p = L[L.ma_diem.isin(dbl)].sort_values("nguoi_gan").groupby("ma_diem")
    a = p.nth(0).set_index("ma_diem")
    return L, a, p.nth(1).set_index("ma_diem").reindex(a.index)


def sigma2(d):
    _, a, b = pairs()
    eff = A.interpreter_effects(d, a, b)
    return {"forest": eff["forest 2020 (F1)"]["sigma_b2"], "agri": eff["agriculture 2020 (F3)"]["sigma_b2"]}


def total(d, y, x, g, s2):
    """Ratio estimate with sampling SE, interpreter SE and total SE."""
    r = A.ratio(d, y, x)
    w = d.w.to_numpy()
    den = (w * np.asarray(x, float)).sum()
    c = [(w * np.asarray(g, float))[(d["from"] == j).to_numpy()].sum() / den for j in J]
    se_int = float(np.sqrt(s2 * sum(v ** 2 for v in c)))
    se_tot = float(np.sqrt(r["se"] ** 2 + se_int ** 2))
    return {"est": r["est"], "se_sampling": r["se"], "se_interpreter": se_int, "se_total": se_tot,
            "lo_sampling": r["est"] - A.Z * r["se"], "hi_sampling": r["est"] + A.Z * r["se"],
            "lo_total": r["est"] - A.Z * se_tot, "hi_total": r["est"] + A.Z * se_tot}


def headline(d, s2):
    rows = []
    known = d.known.to_numpy(float)
    forest = (d.f1 == "forest").to_numpy(float)
    agri = d.agri_2020.to_numpy(float)
    doms = {"record natural + planted": d.record_forest == 1, "record natural": d.record == "natural",
            "record planted": d.record == "planted", "record without forest": d.record == "nonforest",
            "all": d.record == d.record}
    for name, m in doms.items():
        x = m.to_numpy(float)
        rows.append({"item": "EUDR forest share", "domain": name, **total(d, x * forest * known, x * known, x * known, s2["forest"])})
        rows.append({"item": "agriculture share", "domain": name, **total(d, x * agri, x, x, s2["agri"])})
    for o in ("ubnd_xa", "doanh_nghiep", "ho_gia_dinh_ca_nhan_khac", "vuon_quoc_gia_khu_bao_ton", "ban_quan_ly_kiem_lam"):
        x = ((d.record_forest == 1) & (d.owner == o)).to_numpy(float)
        rows.append({"item": "agriculture share", "domain": f"record forest, owner {o}", **total(d, x * agri, x, x, s2["agri"])})
    calls = A.map_calls(d)
    ref = forest
    for mname, mm in calls.items():
        mm = np.asarray(mm, float)
        rows.append({"item": "overall accuracy (EUDR)", "domain": mname,
                     **total(d, known * (mm == ref), known, known * (2 * mm - 1), s2["forest"])})
    rec = np.asarray(calls["record (natural or planted)"], float)
    for mname, mm in calls.items():
        if mname.startswith("record"):
            continue
        mm = np.asarray(mm, float)
        y = known * ((mm == ref).astype(float) - (rec == ref).astype(float))
        rows.append({"item": "paired OA difference vs record (EUDR)", "domain": mname,
                     **total(d, y, known, 2 * known * (mm - rec), s2["forest"])})
    return pd.DataFrame(rows)


def agreement():
    _, a, b = pairs()
    out = {}
    ga, gb = a.f3.map(A.group_of), b.f3.map(A.group_of)
    for g in A.GROUPS:
        pa, pb = ga == g, gb == g
        both, either = (pa & pb).sum(), pa.sum() + pb.sum()
        out[g] = {"points (either interpreter)": int((pa | pb).sum()), "specific agreement": float(2 * both / either) if either else None}
    ya, yb = a.f3.str.startswith("N"), b.f3.str.startswith("N")
    po, k = A.kappa(ya.astype(str), yb.astype(str))
    both = (ya & yb).sum()
    out["agriculture yes/no"] = {"agreement": float(po), "kappa": float(k),
                                 "specific agreement (agriculture)": float(2 * both / (ya.sum() + yb.sum()))}
    return out


def low_confidence(d):
    L, _, _ = pairs()
    conf = L.groupby("ma_diem").f4.first()
    single_low = (d.source == "single") & d.ma_diem.map(conf).eq("low")
    s = d[~single_low.to_numpy()].copy()
    s["n2_c"] = s.cell.map(s.groupby("cell").size())
    assert s.cell.nunique() == d.cell.nunique()
    rows = []
    for tag, dd in (("all labels", d), ("without single low-confidence labels", s)):
        known, forest, agri = dd.known.to_numpy(float), (dd.f1 == "forest").to_numpy(float), dd.agri_2020.to_numpy(float)
        for name, m in (("record natural + planted", dd.record_forest == 1), ("record without forest", dd.record == "nonforest")):
            x = m.to_numpy(float)
            for item, y, xx in (("EUDR forest share", x * forest * known, x * known), ("agriculture share", x * agri, x)):
                r = A.ratio(dd, y, xx)
                rows.append({"labels": tag, "domain": name, "item": item, "n": len(dd), **A.row(item, r)})
        for mname in ("record (natural or planted)", "GFC2020 V4"):
            mm = np.asarray(A.map_calls(dd)[mname], float)
            r = A.ratio(dd, known * (mm == forest), known)
            rows.append({"labels": tag, "domain": mname, "item": "overall accuracy (EUDR)", "n": len(dd), **A.row("OA", r)})
    return pd.DataFrame(rows), int(single_low.sum())


DATE = re.compile(r"(\d{4})-(\d{2})-(\d{2})|(?:(\d{1,2})/)?(\d{1,2})/(\d{4})")


def first_date(text):
    if not isinstance(text, str):
        return None
    m = DATE.search(text)
    if not m:
        return None
    if m.group(1):
        y, mo = int(m.group(1)), int(m.group(2))
    else:
        y, mo = int(m.group(6)), int(m.group(5))
    return (y, mo) if 2000 <= y <= 2026 and 1 <= mo <= 12 else None


def imagery():
    L, _, _ = pairs()
    dt = L.F8.map(first_date)
    gap = dt.dropna().map(lambda t: (t[0] - 2020) * 12 + t[1] - 12)  # months after December 2020
    return {"main interpretations": len(L), "with a parsable date": int(dt.notna().sum()),
            "median absolute gap (months)": float(gap.abs().median()),
            "within 6 months": float((gap.abs() <= 6).mean()), "within 12 months": float((gap.abs() <= 12).mean()),
            "within 24 months": float((gap.abs() <= 24).mean()), "before 2021": float((gap <= 0).mean()),
            "share citing 'no imagery close to 2020' (F5)": float(L.F5.fillna("").str.contains("Không có ảnh gần năm 2020").mean()),
            "low confidence share": float((L.f4 == "low").mean())}


def main():
    d, _ = A.load()
    OUT.mkdir(parents=True, exist_ok=True)
    s2 = sigma2(d)
    h = headline(d, s2)
    h.to_csv(OUT / "interpreter_variance_headline.csv", index=False)
    lc, n_low = low_confidence(d)
    lc.to_csv(OUT / "low_confidence_sensitivity.csv", index=False)
    res = {"sigma_b2": s2, "agreement_by_group": agreement(), "single_low_confidence_points": n_low, "imagery": imagery()}
    (OUT / "robustness_b.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 220, "display.max_columns", 20)
    print((h.set_index(["item", "domain"])[["est", "lo_sampling", "hi_sampling", "lo_total", "hi_total"]] * 100).round(1).to_string())
    print(lc[["labels", "domain", "item", "n", "est", "lo", "hi"]].round(3).to_string(index=False))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
