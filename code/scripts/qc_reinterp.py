"""Paper B: QC of the blind re-interpretation sheets (the analysis plan; the interpretation sheet), then the comparison with the
first interpretation and the list for the consensus meeting. Same normalisation and consistency rules as the frozen
scripts/qc_labels.py (imported, not changed). No label is joined to the record class, GFC or the strata here.

    .venv/Scripts/python paper/land_audit/scripts/qc_reinterp.py

Input: data/interim/land_audit/labels_raw_giaolai/phieu_giaolai_A{1..4}.csv, reinterp_key.csv, labels_long.csv,
phase2_key.csv (coordinates only, for the links).
Output: labels_giaolai_long.csv, runs/land_audit/qc_reinterp.json, paper/land_audit/gan_nhan/thong_nhat/
DANH_SACH_THONG_NHAT.csv (points with different F1 between two interpretations: re-interpretation and double coding).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qc_labels as q  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/interim/land_audit"
RAW = D / "labels_raw_giaolai"
RUNS = ROOT / "runs/land_audit"
MEET = ROOT / "paper/land_audit/gan_nhan/thong_nhat"
KEEP = ["nguoi_gan", "ma_diem", "f1", "f2", "f3", "f4", "F4b", "F5", "F6", "f7", "F7b", "F8", "F9"]


def load(code):
    d = pd.read_csv(RAW / f"phieu_giaolai_{code}.csv", encoding="utf-8-sig", dtype=str)
    d = d.rename(columns={c: q.field(c) for c in d.columns if q.field(c)})
    d = d[[c for c in d.columns if c in ("ma_diem", "F1", "F2", "F3", "F4", "F4b", "F5", "F6", "F7", "F7b", "F8", "F9")]]
    d["nguoi_gan"] = code
    d["ma_diem"] = d.ma_diem.str.strip()
    return d


def main():
    key = pd.read_csv(D / "reinterp_key.csv", dtype=str)
    rows, rep = [], {"files": {}}
    for c in q.CODES:
        d = load(c)
        filled = d[d.F1.notna() & (d.F1.str.strip() != "")]
        assigned = set(key.ma_diem[key.nguoi_giao_lai == c])
        rep["files"][c] = {"rows": len(d), "filled": len(filled), "assigned": len(assigned),
                           "missing": sorted(assigned - set(filled.ma_diem)), "not_assigned": sorted(set(d.ma_diem) - assigned),
                           "duplicates": d.ma_diem[d.ma_diem.duplicated()].tolist()}
        rows.append(filled)
    L = pd.concat(rows, ignore_index=True)
    L["f1"] = L.F1.str.strip().map(q.F1).fillna("INVALID:" + L.F1.astype(str))
    L["f2"], L["f3"], L["f7"] = L.F2.map(q.norm_f2), L.F3.map(q.norm_f3), L.F7.map(q.norm_f3)
    L["f4"] = L.F4.str.strip().map({"Cao": "high", "Thấp": "low"})
    f3_forest = L.f3.str.startswith("R", na=False)
    f3_nonforest = L.f3.str.match(r"^[NK]\d", na=False)
    rep["issues"] = {
        "f1_invalid": int(L.f1.str.startswith("INVALID").sum()),
        "f2_invalid_or_missing_with_f1_known": int(((L.f2.isna() | L.f2.str.startswith("INVALID", na=False)) & (L.f1 != "unknown")).sum()),
        "f3_invalid_or_missing": int(L.f3.isna().sum() + L.f3.str.startswith("INVALID", na=False).sum()),
        "f4_missing": int(L.f4.isna().sum()),
        "forest_but_f3_crop_or_other": int(((L.f1 == "forest") & f3_nonforest).sum()),
        "nonforest_but_f3_forest_class": int(((L.f1 == "nonforest") & f3_forest).sum()),
        "forest_but_f2_nonforest_type": int(((L.f1 == "forest") & L.f2.notna() & ~L.f2.isin(q.FOREST_F2)).sum()),
        "nonforest_but_f2_forest_type": int(((L.f1 == "nonforest") & L.f2.isin(q.FOREST_F2)).sum()),
        "low_confidence_without_reason": int(((L.f4 == "low") & L.F4b.isna()).sum()),
    }
    rep["labels"] = len(L)
    rep["f1"], rep["f4"] = L.f1.value_counts().to_dict(), L.f4.value_counts().to_dict()
    L[KEEP].to_csv(D / "labels_giaolai_long.csv", index=False, encoding="utf-8")
    # comparison with the first interpretation (only after the QC above)
    first = pd.read_csv(D / "labels_long.csv", dtype=str)
    first = first[(first.kind == "main") & first.ma_diem.isin(key.ma_diem)]
    m = key.merge(first[KEEP].add_suffix("_1"), left_on=["ma_diem", "nguoi_dau"], right_on=["ma_diem_1", "nguoi_gan_1"]) \
           .merge(L[KEEP].add_suffix("_2"), left_on=["ma_diem", "nguoi_giao_lai"], right_on=["ma_diem_2", "nguoi_gan_2"])
    rep["comparison"] = {"points": len(m), "f1_agree": round(float((m.f1_1 == m.f1_2).mean()), 3),
                         "f3_agree": round(float((m.f3_1 == m.f3_2).mean()), 3),
                         "second_f4": m.f4_2.value_counts().to_dict(),
                         "f1_pairs": pd.crosstab(m.f1_1, m.f1_2).to_dict()}
    # consensus list: re-interpretation pairs and main double-coded pairs whose F1 differs (the analysis plan)
    coords = pd.read_csv(D / "phase2_key.csv", usecols=["ma_diem", "lat", "lon"], dtype={"ma_diem": str}).set_index("ma_diem")
    dis = m[m.f1_1 != m.f1_2].assign(nguon="giao lại")
    main = pd.read_csv(D / "labels_long.csv", dtype=str)
    main = main[main.kind == "main"]
    dbl = main.groupby("ma_diem").filter(lambda g: g.nguoi_gan.nunique() == 2 and g.f1.nunique() == 2)
    pairs = []
    for mid, g in dbl.groupby("ma_diem"):
        a, b = g.iloc[0], g.iloc[1]
        pairs.append({"ma_diem": mid, "nguon": "gán kép", **{f"{k}_1": a[k] for k in KEEP}, **{f"{k}_2": b[k] for k in KEEP}})
    lst = pd.concat([dis, pd.DataFrame(pairs)], ignore_index=True)
    out = pd.DataFrame({
        "Mã điểm": lst.ma_diem, "Nguồn": lst.nguon,
        "Người 1": lst.nguoi_gan_1, "Người 1: F1 / F3 / F4": lst.f1_1 + " / " + lst.f3_1.fillna("") + " / " + lst.f4_1.fillna(""),
        "Người 1: lý do, ghi chú": (lst.F4b_1.fillna("") + " | " + lst.F9_1.fillna("")).str.strip(" |"),
        "Người 2": lst.nguoi_gan_2, "Người 2: F1 / F3 / F4": lst.f1_2 + " / " + lst.f3_2.fillna("") + " / " + lst.f4_2.fillna(""),
        "Người 2: lý do, ghi chú": (lst.F4b_2.fillna("") + " | " + lst.F9_2.fillna("")).str.strip(" |"),
        "Link Esri Wayback 01/2021": [f"https://livingatlas.arcgis.com/wayback/#mapCenter={coords.at[x, 'lon']:.6f}%2C"
                                      f"{coords.at[x, 'lat']:.6f}%2C17&mode=explore&active=1049" for x in lst.ma_diem],
        "F1 thống nhất": "", "F3 thống nhất": "", "Lý do, ảnh đã xem lại": "", "Ngày, người tham dự": ""})
    MEET.mkdir(parents=True, exist_ok=True)
    out.sort_values(["Nguồn", "Mã điểm"]).to_csv(MEET / "DANH_SACH_THONG_NHAT.csv", index=False, encoding="utf-8-sig")
    rep["consensus_list"] = {"re_interpretation": int(len(dis)), "double_coded": int(len(pairs)), "total": int(len(out))}
    (RUNS / "qc_reinterp.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
