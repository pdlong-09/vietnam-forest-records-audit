"""Paper B: final reference label of every main point (the analysis plan) and the derived variables (the analysis plan).
No join with the record, GFC or the strata here.

    .venv/Scripts/python paper/land_audit/scripts/build_final_labels.py

Rules (the analysis plan):
1. one interpretation: that label;
2. two interpretations (double coding or blind re-interpretation) with the same F1: that F1; F3 (and F6, F7) from
   the interpreter with the higher confidence, the first interpreter on a tie (double coding: nguoi_1 of the
   assignment; re-interpretation: the first interpreter);
3. different F1: the consensus row of gan_nhan/thong_nhat/DANH_SACH_THONG_NHAT.csv (two interpreters and the
   coordinator); "unknown" when they did not agree. F6/F7 then come from the interpreter whose F3 equals the consensus
   F3, else from the first interpreter.
Every individual label stays in labels_long.csv / labels_giaolai_long.csv for the inter-interpreter analysis.
Output: data/interim/land_audit/labels_final.csv and runs/land_audit/final_labels.json.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/interim/land_audit"
F1_MAP = {"forest": "forest", "nonforest": "nonforest", "unknown": "unknown"}
GROUP = {**{c: "natural forest" for c in ("R1", "R2")}, **{c: "planted forest" for c in ("R3", "R4")},
         **{c: "forest land without trees" for c in ("R5", "R6")}, **{f"N{i}": "perennial crops" for i in range(1, 9)},
         **{f"N{i}": "annual crops" for i in range(9, 12)}, **{f"K{i}": "other" for i in range(1, 7)}}


def derived(f1, f3, f6):
    big = f6 in ("Từ 0,3 đến 0,5 ha", "Trên 0,5 ha")
    return {"forest_eudr": None if f1 == "unknown" else int(f1 == "forest"),
            "forest_vn": None if f1 == "unknown" else int((f3 in ("R1", "R2", "R3", "R4", "N3")) and big),
            "landuse_2020": GROUP.get(f3), "agri_2020": int(str(f3).startswith("N")) if f3 else None}


def main():
    L1 = pd.read_csv(D / "labels_long.csv", dtype=str)
    L1 = L1[L1.kind == "main"]
    L2 = pd.read_csv(D / "labels_giaolai_long.csv", dtype=str)
    key = pd.read_csv(D / "phase2_key.csv", usecols=["ma_diem", "nguoi_1"], dtype=str).set_index("ma_diem").nguoi_1
    first_re = pd.read_csv(D / "reinterp_key.csv", dtype=str).set_index("ma_diem").nguoi_dau
    cons = pd.read_csv(ROOT / "paper/land_audit/gan_nhan/thong_nhat/DANH_SACH_THONG_NHAT.csv", encoding="utf-8-sig",
                       dtype=str).set_index("Mã điểm")
    rows = []
    for mid, g in pd.concat([L1.assign(round="first"), L2.assign(round="re")]).groupby("ma_diem"):
        g = g.reset_index(drop=True)
        if len(g) == 1:
            r = g.iloc[0]
            rec = {"source": "single", "f1": r.f1, "f3": r.f3, "f6": r.F6, "f7": r.f7, "from": r.nguoi_gan}
        else:
            first = first_re.get(mid) if (g["round"] == "re").any() else key.get(mid)
            kind = "re-interpretation" if (g["round"] == "re").any() else "double coding"
            if g.f1.nunique() == 1:
                rank = g.assign(c=(g.f4 == "high").astype(int), f=(g.nguoi_gan == first).astype(int)).sort_values(
                    ["c", "f"], ascending=False)
                r = rank.iloc[0]
                rec = {"source": f"{kind}, F1 agree", "f1": r.f1, "f3": r.f3, "f6": r.F6, "f7": r.f7, "from": r.nguoi_gan}
            else:
                c = cons.loc[mid]
                f3 = c["F3 thống nhất"].strip().split()[0]
                match = g[g.f3 == f3]
                r = match.iloc[0] if len(match) else g[g.nguoi_gan == first].iloc[0]
                rec = {"source": f"{kind}, consensus", "f1": F1_MAP[c["F1 thống nhất"].strip()], "f3": f3, "f6": r.F6,
                       "f7": r.f7, "from": "consensus"}
        rows.append({"ma_diem": mid, "n_interpretations": len(g), **rec, **derived(rec["f1"], rec["f3"], rec["f6"])})
    out = pd.DataFrame(rows).sort_values("ma_diem")
    out.to_csv(D / "labels_final.csv", index=False, encoding="utf-8")
    summary = {"points": len(out), "by_source": out.source.value_counts().to_dict(), "f1": out.f1.value_counts().to_dict(),
               "forest_eudr": out.forest_eudr.value_counts(dropna=False).to_dict(),
               "forest_vn": out.forest_vn.value_counts(dropna=False).to_dict(),
               "landuse_2020": out.landuse_2020.value_counts(dropna=False).to_dict()}
    (ROOT / "runs/land_audit/final_labels.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                                           encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
