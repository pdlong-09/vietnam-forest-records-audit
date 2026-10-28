"""Paper B: blind re-interpretation of low-confidence points (the analysis plan), drawn ONCE after all sheets are in.

    .venv/Scripts/python paper/land_audit/scripts/draw_reinterpretation.py

Candidates: main points with a single interpretation and F4 = low (data/interim/land_audit/labels_long.csv, built by
qc_labels.py from the full sheets). 20% are drawn at random (seed 20261023) and each goes to an interpreter other than
the first one, balanced over the other three. The sheets show no earlier label, no confidence and no reason why the
point was chosen. Outputs:
  data/interim/land_audit/reinterp_key.csv            research only (ma_diem, nguoi_dau, nguoi_giao_lai)
  paper/land_audit/gan_nhan/giao_lai/phieu_giaolai_A{1..4}.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data/interim/land_audit"
SHEETS = ROOT / "paper/land_audit/gan_nhan/giao_lai"
SEED = 20261023
CODES = ["A1", "A2", "A3", "A4"]
FIELDS = ["F1 Rừng 2020", "F2 Loại 2020", "F3 Sử dụng đất chi tiết 2020", "F4 Độ tin cậy", "F4b Lý do khi độ tin cậy thấp",
          "F5 Vấn đề", "F6 Kích thước mảng", "F7 Sử dụng đất hiện tại", "F7b Ngày ảnh mới nhất", "F8 Ảnh dùng cho 2020",
          "F9 Ghi chú"]


def links(lat, lon):
    la, lo = f"{lat:.6f}", f"{lon:.6f}"
    return {"Link Google Maps": f"https://www.google.com/maps/@{la},{lo},350m/data=!3m1!1e3",
            "Link Google Earth": f"https://earth.google.com/web/@{la},{lo},500a,700d,35y,0h,0t,0r",
            "Link Esri Wayback 01/2021": f"https://livingatlas.arcgis.com/wayback/#mapCenter={lo}%2C{la}%2C17&mode=explore&active=1049",
            "Link Sentinel-2 12/2020": (f"https://browser.dataspace.copernicus.eu/?zoom=16&lat={la}&lng={lo}&datasetId=S2_L2A_CDAS"
                                        "&fromTime=2020-12-01T00%3A00%3A00.000Z&toTime=2020-12-31T23%3A59%3A59.999Z&layerId=1_TRUE_COLOR")}


def main():
    key_path = OUT / "reinterp_key.csv"
    if key_path.exists():
        raise SystemExit(f"{key_path} exists: the list is drawn once (the analysis plan)")
    L = pd.read_csv(OUT / "labels_long.csv", dtype=str)
    main_ = L[L.kind == "main"]
    n_int = main_.groupby("ma_diem").nguoi_gan.nunique()
    single = main_[main_.ma_diem.isin(n_int.index[n_int == 1])]
    cand = sorted(single.ma_diem[single.f4 == "low"])
    rng = np.random.default_rng(SEED)
    n = round(0.20 * len(cand))
    pick = sorted(rng.choice(cand, n, replace=False))
    first = dict(zip(single.ma_diem, single.nguoi_gan))
    load = dict.fromkeys(CODES, 0)
    assign = {}
    for m in rng.permutation(pick):
        others = [c for c in CODES if c != first[m]]
        low = min(load[c] for c in others)
        c = rng.choice([c for c in others if load[c] == low])
        assign[m] = c
        load[c] += 1
    key = pd.DataFrame({"ma_diem": pick, "nguoi_dau": [first[m] for m in pick], "nguoi_giao_lai": [assign[m] for m in pick]})
    key.to_csv(key_path, index=False)
    coords = pd.read_csv(OUT / "phase2_key.csv", usecols=["ma_diem", "lat", "lon"], dtype={"ma_diem": str}).set_index("ma_diem")
    SHEETS.mkdir(parents=True, exist_ok=True)
    for c in CODES:
        rows = []
        for m in key.ma_diem[key.nguoi_giao_lai == c]:
            lat, lon = float(coords.at[m, "lat"]), float(coords.at[m, "lon"])
            rows.append({"Mã điểm": m, "Giao cho": c, "Vĩ độ": f"{lat:.6f}", "Kinh độ": f"{lon:.6f}", **links(lat, lon),
                         **dict.fromkeys(FIELDS, "")})
        pd.DataFrame(rows).to_csv(SHEETS / f"phieu_giaolai_{c}.csv", index=False, encoding="utf-8-sig")
    summary = {"candidates_single_low": len(cand), "drawn": n, "per_interpreter": load,
               "first_interpreter": key.nguoi_dau.value_counts().to_dict()}
    (ROOT / "runs/land_audit/reinterp_draw.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
