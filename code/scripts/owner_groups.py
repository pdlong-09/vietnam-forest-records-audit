"""Paper B: owner groups from the free-text forest-owner field `churung` of the 2020 records (the analysis plan).

Rules are applied in order; the first match wins. They read only the owner name, never a label. Names of persons
are never published: the output is the group only.

    .venv/Scripts/python paper/land_audit/scripts/owner_groups.py   # prints the group shares of the phase-2 points
"""
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

RULES = [  # (group, pattern on the lower-cased, accent-stripped name)
    ("vuon_quoc_gia_khu_bao_ton", r"\bvqg\b|vuon quoc gia|khu bao ton|\bkbt\b|khu du tru|bttn|rung dac dung|\bbqlrdd\b"),
    ("ban_quan_ly_kiem_lam", r"\bbql|ban quan ly|ban ql|phong ho|kiem lam"),
    ("luc_luong_vu_trang", r"quan doi|bo doi|quan khu|su doan|trung doan|lu doan|binh doan|don bien phong|bien phong|"
                           r"huyen doi|tinh doi|"
                           r"cong an|bo chi huy|\bbchqs\b|quan su|trai giam|quoc phong|\bdoan kt|kinh te quoc phong"),
    ("doanh_nghiep", r"cong ty|\bc\.? ?ty\b|\bcty\b|tnhh|co phan|\bdntn\b|doanh nghiep|tong cong ty|lam truong|"
                     r"nong truong|\bnlt\b|\bclt\b|lam nghiep|xi nghiep|tap doan|thuy dien"),
    ("hop_tac_xa", r"hop tac xa|\bhtx\b"),
    ("ubnd_xa", r"\bubnd\b|uy ban|\bub\b|xa\b.*ubnd"),
    ("cong_dong", r"cong dong|\bthon\b|\bbuon\b|\bban\b|\blang\b|\bto\b|\bbon\b|\bplei\b|\bnhom ho\b"),
    ("to_chuc_khac", r"truong|vien|trung tam|hoi\b|nha tho|chua\b|to chuc|don vi|tnxp|thanh nien xung phong"),
]


def strip(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s)).replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", "".join(c for c in s if unicodedata.category(c) != "Mn")).lower().strip()


def owner_group(name) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)) or not str(name).strip():
        return "khong_ro"
    s = strip(name)
    if len(re.sub(r"[^a-z]", "", s)) < 2:  # '0', '-', bare numbers
        return "khong_ro"
    for g, p in RULES:
        if re.search(p, s):
            return g
    # no organisation keyword: a household or person (the usual case in the records), or an unknown body
    return "ho_gia_dinh_ca_nhan_khac"


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    key = pd.read_csv(root / "data/interim/land_audit/phase2_key.csv", usecols=["ma_diem", "churung"], dtype=str)
    key = key[key.ma_diem.str.startswith("B")]
    g = key.churung.map(owner_group)
    print(g.value_counts().to_string())  # counts only: owner names are never printed
    sys.exit(0)
