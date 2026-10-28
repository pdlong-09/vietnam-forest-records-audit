"""Paper B: quality control of the returned interpretation sheets (format, completeness, assignment, consistency,
agreement). No label is joined to the record class, GFC or the strata here: that analysis waits for the paper B
preregistration.

    .venv/Scripts/python paper/land_audit/scripts/qc_labels.py

Input: data/interim/land_audit/labels_raw/phieu_A{1..4}.csv (copies of the returned sheets, SHA256SUMS.txt) and the
assignment in data/interim/land_audit/phase2_key.csv (only ma_diem, nguoi_1, nguoi_2 are read).
Output: data/interim/land_audit/labels_long.csv (one row per interpreter x point, normalised codes) and
runs/land_audit/qc_labels.json.
"""
import itertools
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data/interim/land_audit/labels_raw"
OUT = ROOT / "data/interim/land_audit"
RUNS = ROOT / "runs/land_audit"
CODES = ["A1", "A2", "A3", "A4"]

F1 = {"Rừng": "forest", "Không rừng": "nonforest", "Không xác định được": "unknown"}
F2 = {"Rừng nguyên sinh hoặc tái sinh tự nhiên": "natural", "Rừng trồng": "planted",
      "Không có cây hoặc cây bụi": "no_trees", "Đất có cây gỗ khác": "owl", "Cây phục vụ nông nghiệp": "agri_trees",
      "Cây trong đô thị": "urban_trees", "Cây ngoài rừng": "tof", "Đất không rừng nằm trong vùng rừng": "nonforest_in_forest"}
F3_CODES = [f"R{i}" for i in range(1, 7)] + [f"N{i}" for i in range(1, 12)] + [f"K{i}" for i in range(1, 7)]
FOREST_F2 = {"natural", "planted"}


def field(col):
    """Map any header spelling ('F1 Rừng 2020', 'F1_Rung2020', 'F4b_LyDo', ...) to F1..F9 / F4b / F7b / id."""
    c = col.strip().lower()
    if c in ("mã điểm", "ma_diem", "ma diem"):
        return "ma_diem"
    m = re.match(r"f(\d)(b?)\b|f(\d)(b?)[ _]", c)
    if m:
        n, b = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
        return f"F{n}{b}"
    return None


def norm_f2(v):
    if pd.isna(v):
        return None
    for k, code in F2.items():
        if str(v).strip().startswith(k):
            return code
    return "INVALID:" + str(v).strip()[:40]


def norm_f3(v):
    if pd.isna(v):
        return None
    m = re.match(r"\s*([RNK]\d{1,2})\b", str(v))
    return m.group(1) if m and m.group(1) in F3_CODES else "INVALID:" + str(v).strip()[:40]


def load(code):
    d = pd.read_csv(RAW / f"phieu_{code}.csv", encoding="utf-8-sig", dtype=str)
    d = d.rename(columns={c: field(c) for c in d.columns if field(c)})
    d = d[[c for c in d.columns if c in ("ma_diem", "F1", "F2", "F3", "F4", "F4b", "F5", "F6", "F7", "F7b", "F8", "F9")]]
    d["nguoi_gan"] = code
    d["ma_diem"] = d.ma_diem.str.strip()
    return d


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    key = pd.read_csv(OUT / "phase2_key.csv", usecols=["ma_diem", "nguoi_1", "nguoi_2"], dtype=str).fillna("")
    assigned = {c: set(key.ma_diem[(key.nguoi_1 == c) | (key.nguoi_2 == c) | (key.nguoi_1 == "tất cả")]) for c in CODES}
    rows, report = [], {"files": {}}
    for c in CODES:
        d = load(c)
        filled = d[d.F1.notna() & (d.F1.str.strip() != "")].copy()
        dup = filled.ma_diem[filled.ma_diem.duplicated()].tolist()
        not_assigned = sorted(set(filled.ma_diem) - assigned[c])
        report["files"][c] = {"rows": len(d), "filled": len(filled), "duplicates": dup, "not_assigned": not_assigned,
                              "assigned_total": len(assigned[c]), "assigned_done": len(set(filled.ma_diem) & assigned[c])}
        rows.append(filled)
    L = pd.concat(rows, ignore_index=True)
    L["f1"] = L.F1.str.strip().map(F1).fillna("INVALID:" + L.F1.astype(str))
    L["f2"] = L.F2.map(norm_f2)
    L["f3"] = L.F3.map(norm_f3)
    L["f7"] = L.F7.map(norm_f3)
    L["f4"] = L.F4.str.strip().map({"Cao": "high", "Thấp": "low"})
    L["kind"] = L.ma_diem.str[:2].map(lambda s: "calibration" if s == "HC" else "main")
    # consistency rules of the guide (section 6): crops and other land are non-forest; F2 must match F1
    f3_forest = L.f3.str.startswith("R", na=False)
    f3_nonforest = L.f3.str.match(r"^[NK]\d", na=False)
    issues = {
        "f1_invalid": int(L.f1.str.startswith("INVALID").sum()),
        "f2_invalid_or_missing": int(L.f2.isna().sum() + L.f2.str.startswith("INVALID", na=False).sum()),
        "f3_invalid_or_missing": int(L.f3.isna().sum() + L.f3.str.startswith("INVALID", na=False).sum()),
        "f4_missing": int(L.f4.isna().sum()),
        "forest_but_f3_crop_or_other": int(((L.f1 == "forest") & f3_nonforest).sum()),
        "nonforest_but_f3_forest_class": int(((L.f1 == "nonforest") & f3_forest).sum()),
        "forest_but_f2_nonforest_type": int(((L.f1 == "forest") & L.f2.notna() & ~L.f2.isin(FOREST_F2)).sum()),
        "nonforest_but_f2_forest_type": int(((L.f1 == "nonforest") & L.f2.isin(FOREST_F2)).sum()),
        "low_confidence_without_reason": int(((L.f4 == "low") & L.F4b.isna()).sum()),
    }
    report["issues"] = issues
    report["labels"] = len(L)
    report["unique_points"] = int(L.ma_diem.nunique())
    report["unique_main_points"] = int(L[L.kind == "main"].ma_diem.nunique())
    report["f1"] = L.f1.value_counts().to_dict()
    report["f4"] = L.f4.value_counts().to_dict()
    report["f3_top"] = L.f3.value_counts().head(10).to_dict()
    # imagery actually used (F8): how often each source is named
    src = {"Wayback": r"wayback|esri", "Google Earth": r"google\s*earth|earth pro|lịch sử", "Google Maps": r"google\s*maps",
           "Sentinel-2": r"sentinel|copernicus", "Planet": r"planet|nicfi"}
    f8 = L.F8.fillna("").str.lower()
    report["f8_sources"] = {k: int(f8.str.contains(p).sum()) for k, p in src.items()}
    report["f8_empty"] = int((f8.str.strip() == "").sum())
    # agreement
    hc = L[L.kind == "calibration"].pivot_table(index="ma_diem", columns="nguoi_gan", values="f1", aggfunc="first")
    hc = hc.dropna()
    pair = [float((hc[a] == hc[b]).mean()) for a, b in itertools.combinations(hc.columns, 2)]
    report["calibration"] = {"points_with_all_4": len(hc), "all_4_agree_f1": float((hc.nunique(axis=1) == 1).mean()),
                             "mean_pairwise_f1": round(sum(pair) / len(pair), 3) if pair else None}
    main = L[L.kind == "main"]
    multi = main.groupby("ma_diem").filter(lambda g: g.nguoi_gan.nunique() >= 2)
    agree = multi.groupby("ma_diem").agg(f1=("f1", lambda s: s.nunique() == 1), f3=("f3", lambda s: s.nunique() == 1),
                                         n=("nguoi_gan", "nunique"))
    report["double_coded"] = {"points": len(agree), "f1_agree": round(float(agree.f1.mean()), 3) if len(agree) else None,
                              "f3_agree": round(float(agree.f3.mean()), 3) if len(agree) else None}
    # progress against the assignment: where each interpreter stopped
    for c in CODES:
        done = sorted(x for x in set(L.ma_diem[(L.nguoi_gan == c) & (L.kind == "main")]))
        report["files"][c]["main_range"] = [done[0], done[-1]] if done else None
    keep = ["nguoi_gan", "ma_diem", "kind", "f1", "f2", "f3", "f4", "F4b", "F5", "F6", "f7", "F7b", "F8", "F9"]
    L[keep].to_csv(OUT / "labels_long.csv", index=False, encoding="utf-8")
    (RUNS / "qc_labels.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
