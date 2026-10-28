"""Paper B, phase 2: the 1,200 interpretation points, 25 calibration points and the interpreter package.

    python paper/land_audit/scripts/build_phase2.py [--seed 20261022]

Inputs (data/interim/land_audit/): phase1_points.csv, phase1_points_gfcV4.csv, phase1_strata.json.
Cells (24) = record class (nggocr) x GFC2020 V4 forest/non-forest x forest type (malr3). Phase 2 = simple random
sample of 50 phase-1 points per cell (all of them where a cell has fewer). Calibration = 25 further phase-1 points
(one per cell, then one more at random), never part of phase 2. Each phase-2 point gets one interpreter (A1-A4,
300 each) and a random 20% get a second one (60 per interpreter), never the first.

Outputs
- data/interim/land_audit/phase2_key.csv: research only (public id -> pid, cell, lot attributes, interpreters).
- paper/land_audit/gan_nhan/: what interpreters receive. No record class, owner, forest type or map value:
  DIEM_GAN_NHAN.md (all points with image links and assignments), diem_gan_nhan.kml (1 ha squares and the 10 m
  centre cell, for Google Earth Pro and its historical imagery), phieu_<code>.csv answer sheets per interpreter.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INT = ROOT / "data/interim/land_audit"
OUT = ROOT / "paper/land_audit/gan_nhan"
CODES = ["A1", "A2", "A3", "A4"]
PER_CELL, N_CAL, DOUBLE = 50, 25, 0.20
FIELDS = ["F1_rung_2020", "F2_loai_2020", "F3_sudung_2020", "F4_tin_cay", "F4b_ly_do", "F5_van_de", "F6_kich_thuoc",
          "F7_sudung_hien_tai", "F7b_ngay_anh_moi_nhat", "F8_anh_2020", "F9_ghi_chu"]


def links(lat, lon):
    return {
        "gmaps": f"https://www.google.com/maps/@{lat:.6f},{lon:.6f},350m/data=!3m1!1e3",
        "gearth": f"https://earth.google.com/web/@{lat:.6f},{lon:.6f},500a,700d,35y,0h,0t,0r",
        "wayback": f"https://livingatlas.arcgis.com/wayback/#mapCenter={lon:.6f}%2C{lat:.6f}%2C17&mode=explore&active=1049",
        "s2": (f"https://browser.dataspace.copernicus.eu/?zoom=16&lat={lat:.6f}&lng={lon:.6f}&datasetId=S2_L2A_CDAS"
               "&fromTime=2020-12-01T00%3A00%3A00.000Z&toTime=2020-12-31T23%3A59%3A59.999Z&layerId=1_TRUE_COLOR"),
    }


def square(lat, lon, half_m):
    dlat = half_m / 111320.0
    dlon = half_m / (111320.0 * math.cos(math.radians(lat)))
    c = [(lon - dlon, lat - dlat), (lon + dlon, lat - dlat), (lon + dlon, lat + dlat), (lon - dlon, lat + dlat),
         (lon - dlon, lat - dlat)]
    return " ".join(f"{x:.7f},{y:.7f},0" for x, y in c)


def kml(points):
    pm = []
    for r in points.itertuples():
        for half, style, tag in [(50, "ha", "1 ha"), (5, "cell", "ô 10 m")]:
            pm.append(f"<Placemark><name>{r.ma_diem if half == 5 else ''}</name><description>{r.ma_diem} – {tag}"
                      f"</description><styleUrl>#{style}</styleUrl><Polygon><outerBoundaryIs><LinearRing><coordinates>"
                      f"{square(r.lat, r.lon, half)}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            '<name>Điểm giải đoán bài B</name>'
            '<Style id="ha"><LineStyle><color>ff00ffff</color><width>2</width></LineStyle><PolyStyle><fill>0</fill>'
            '</PolyStyle></Style><Style id="cell"><LineStyle><color>ff0000ff</color><width>2</width></LineStyle>'
            '<PolyStyle><fill>0</fill></PolyStyle></Style>' + "".join(pm) + "</Document></kml>\n")


def table(rows, with_assign=True):
    head = "| Mã điểm | " + ("Giao cho | " if with_assign else "") + "Vĩ độ, kinh độ | Ảnh |\n|---|" + \
           ("---|" if with_assign else "") + "---|---|\n"
    lines = []
    for r in rows.itertuples():
        L = links(r.lat, r.lon)
        img = (f"[Google Maps]({L['gmaps']}) · [Google Earth]({L['gearth']}) · [Esri Wayback 01/2021]({L['wayback']})"
               f" · [Sentinel-2 12/2020]({L['s2']})")
        who = f"{r.giao_cho} | " if with_assign else ""
        lines.append(f"| {r.ma_diem} | {who}{r.lat:.6f}, {r.lon:.6f} | {img} |")
    return head + "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261022)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    p = pd.read_csv(INT / "phase1_points.csv", low_memory=False)
    g = pd.read_csv(INT / "phase1_points_gfcV4.csv")
    p = p.merge(g, on="pid", how="left")
    assert p.gfcV4.notna().all(), "GFC2020 V4 missing for some phase-1 points"
    p["cell"] = p.stratum + "|" + np.where(p.gfcV4 == 1, "gfc_forest", "gfc_nonforest")

    parts, cal = [], []
    for cell, d in p.groupby("cell"):
        d = d.sample(frac=1, random_state=int(rng.integers(2**31)))
        parts.append(d.iloc[:PER_CELL])
        if len(d) > PER_CELL:
            cal.append(d.iloc[PER_CELL:PER_CELL + 1])
    s2 = pd.concat(parts)
    cal = pd.concat(cal)
    rest = p[~p.pid.isin(s2.pid) & ~p.pid.isin(cal.pid)]
    cal = pd.concat([cal, rest.sample(N_CAL - len(cal), random_state=int(rng.integers(2**31)))]).iloc[:N_CAL]

    # public ids in random order, so the id says nothing about the cell
    s2 = s2.sample(frac=1, random_state=int(rng.integers(2**31))).reset_index(drop=True)
    s2["ma_diem"] = [f"B{n:04d}" for n in range(1, len(s2) + 1)]
    s2["nguoi_1"] = [CODES[n % 4] for n in range(len(s2))]
    dbl = rng.choice(len(s2), size=int(round(DOUBLE * len(s2))), replace=False)
    s2["nguoi_2"] = ""
    for k, n in enumerate(sorted(dbl)):
        first = CODES.index(s2.at[n, "nguoi_1"])
        s2.at[n, "nguoi_2"] = CODES[(first + 1 + k % 3) % 4]
    s2["giao_cho"] = s2.nguoi_1 + np.where(s2.nguoi_2 != "", ", " + s2.nguoi_2, "")
    cal = cal.reset_index(drop=True)
    cal["ma_diem"] = [f"HC{n:02d}" for n in range(1, len(cal) + 1)]
    cal["giao_cho"] = "tất cả"

    key_cols = ["ma_diem", "pid", "cell", "stratum", "gfcV4", "lon", "lat", "nguoi_1", "nguoi_2", "matinh", "tinh",
                "huyen", "xa", "tk", "khoanh", "lo", "dtich", "nggocr", "ldlr", "malr3", "mdsd", "churung", "sldlr"]
    key = pd.concat([cal.assign(nguoi_1="tất cả", nguoi_2=""), s2])[key_cols]
    key.to_csv(INT / "phase2_key.csv", index=False, encoding="utf-8")

    OUT.mkdir(parents=True, exist_ok=True)
    allpts = pd.concat([cal[["ma_diem", "lat", "lon"]], s2[["ma_diem", "lat", "lon"]]])
    (OUT / "diem_gan_nhan.kml").write_text(kml(allpts), encoding="utf-8")
    for c in CODES:
        mine = s2[(s2.nguoi_1 == c) | (s2.nguoi_2 == c)].sample(frac=1, random_state=int(rng.integers(2**31)))
        sheet = pd.concat([cal[["ma_diem", "lat", "lon"]], mine[["ma_diem", "lat", "lon"]]])
        for f in FIELDS:
            sheet[f] = ""
        sheet.insert(0, "nguoi_gan", c)
        sheet.to_csv(OUT / f"phieu_{c}.csv", index=False, encoding="utf-8-sig")

    counts = {c: int(((s2.nguoi_1 == c) | (s2.nguoi_2 == c)).sum()) for c in CODES}
    md = f"""# Danh sách điểm giải đoán (bài B)

Tạo ngày {pd.Timestamp.now():%d/%m/%Y}. Đọc **[HUONG_DAN_GIAI_DOAN.md](../HUONG_DAN_GIAI_DOAN.md)** trước khi bắt đầu.

## Cách dùng

1. **Ô cần gán:** mỗi điểm là tâm của một ô 100 × 100 m (1 ha). Nhãn là của **ô 10 × 10 m ở tâm**, còn diện tích, mật độ và mục đích sử dụng được xét trên cả ô 1 ha và vùng xung quanh (hướng dẫn, mục 2).
2. **Google Earth Pro (nên dùng):** mở file **`diem_gan_nhan.kml`**. File vẽ sẵn ô 1 ha (viền vàng) và ô 10 m (viền đỏ) cho mọi điểm. Dùng thanh **lịch sử ảnh** (biểu tượng đồng hồ) để xem ảnh quanh ngày 31/12/2020 và ảnh mới nhất.
3. **Link trong bảng:**
   - Google Maps (ảnh vệ tinh hiện tại);
   - Google Earth trên web;
   - **Esri Wayback 01/2021:** mở sẵn bản ảnh phát hành ngày 13/01/2021, tức ảnh có tại thời điểm sát mốc 2020.
     - Ngày phát hành **không phải** ngày chụp. Bấm vào bản đồ ("Click map for imagery details") để xem ngày chụp thật, rồi ghi vào F8.
     - Danh sách bên trái (đang chọn "Only versions with local changes") là các bản ảnh thật sự khác nhau tại điểm đó. Chọn bản mới nhất để xem hiện trạng hiện tại.
     - Bảng cookie: bấm "Reject All" hoặc "Accept".
   - **Sentinel-2 12/2020 (Copernicus Browser):** mở sẵn ảnh Sentinel-2 L2A màu thật, ngày 31/12/2020.
     - Lần đầu mở: bấm **"Use anonymously"**, rồi tắt bảng giới thiệu ("Remind me later").
     - Nếu ảnh ngày đó có mây hoặc mờ, dùng mũi tên ngày hoặc lịch ở bảng bên trái để chọn ngày quang mây gần nhất.
     - Ảnh 10 m chỉ dùng để xem diễn biến, không thay ảnh độ phân giải cao.
   - Ảnh Planet NICFI cần đăng nhập tài khoản Planet.
4. **Phiếu:** ghi nhãn vào phiếu CSV của mình, **`phieu_<mã>.csv`** (mở bằng Excel). Phiếu đã có sẵn đúng các điểm được giao, theo thứ tự đã xáo. Các cột F1–F9 theo mục 5 của hướng dẫn; dùng đúng tên lựa chọn trong hướng dẫn.
5. **Làm độc lập:** không xem phiếu của người khác và không bàn về từng điểm. Một số điểm được giao cho hai người; điều đó là bình thường. Các điểm có độ tin cậy thấp có thể được giao thêm cho người thứ hai sau này.
6. **Không mở bản đồ rừng hay bản đồ hiện trạng nào**, và **không dùng công cụ AI** để gợi ý nhãn.

## Phân công

| Người gán | Số điểm chính thức | Điểm hiệu chỉnh |
|---|---|---|
""" + "".join(f"| {c} | {counts[c]} | {len(cal)} |\n" for c in CODES) + f"""
Tổng cộng {len(s2)} điểm chính thức, trong đó {len(dbl)} điểm do hai người gán, và {len(cal)} điểm hiệu chỉnh do cả 4 người gán.

## 1. Điểm hiệu chỉnh: làm trước, sau đó cả nhóm thảo luận

{table(cal)}
## 2. Điểm chính thức

{table(s2)}"""
    (OUT / "DIEM_GAN_NHAN.md").write_text(md, encoding="utf-8")

    print(f"phase 2: {len(s2)} points in {s2.cell.nunique()} cells; calibration {len(cal)}; double {len(dbl)}")
    print(s2.cell.value_counts().sort_index().to_string())
    print("per interpreter:", counts)


if __name__ == "__main__":
    main()
