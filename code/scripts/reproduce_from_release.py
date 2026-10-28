"""Paper B: the main estimates of the article recomputed from the public reference sample alone.

    python reproduce_from_release.py path/to/reference_sample_folder

The folder holds strata.csv, phase1_points.csv and phase2_points_labels.csv (Kaggle dataset, or data/ of the code
release). Uses the two-phase estimator of analysis_b.py (mean_var, ratio). The intervals are sampling intervals; where
the article also adds the interpreter component (Materials & Methods, Estimation), its intervals are slightly wider.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_b import Z, mean_var, ratio  # noqa: E402,F401


def load(folder):
    f = Path(folder)
    strata = pd.read_csv(f / "strata.csv").set_index("stratum")
    p1 = pd.read_csv(f / "phase1_points.csv", usecols=["pid", "stratum", "cell"])
    d = pd.read_csv(f / "phase2_points_labels.csv")
    n1h, n1c = p1.groupby("stratum").size(), p1.groupby("cell").size()
    assert n1h.sort_index().equals(strata.n_phase1.sort_index().astype(n1h.dtype))
    d["n1_h"], d["n1_c"] = d.stratum.map(n1h), d.cell.map(n1c)
    d["n2_c"] = d.cell.map(d.groupby("cell").size())
    d["P_h"] = d.stratum.map(strata.area_ha / strata.area_ha.sum())
    w = d.stratum.map(strata.area_ha) * d.n1_c / d.n1_h / d.n2_c
    assert np.allclose(w, d.weight_ha, rtol=1e-4), "design weights differ from weight_ha"
    return d, strata.area_ha.sum()


def show(name, d, y, x, scale=100):
    r = ratio(d, y, x)
    print(f"{name:<58} {r['est'] * scale:6.1f}  ({(r['est'] - Z * r['se']) * scale:.1f}-{(r['est'] + Z * r['se']) * scale:.1f})")


def main(folder):
    d, area = load(folder)
    print(f"population {area / 1e6:.2f} Mha, phase-2 points {len(d)}\n")
    known = (d.f1_forest_2020 != "unknown").astype(float)
    forest = (d.f1_forest_2020 == "forest").astype(float)
    agri = d.agriculture_2020.astype(float)
    for rec, m in (("natural + planted forest", d.record_origin != "without forest"),
                   ("natural forest", d.record_origin == "natural forest"),
                   ("planted forest", d.record_origin == "planted forest"),
                   ("without forest", d.record_origin == "without forest")):
        x = m.astype(float)
        show(f"record {rec}: EUDR/FAO forest 2020 (%)", d, x * forest, x * known)
        show(f"record {rec}: agriculture 2020 (%)", d, x * agri, x)
    print("\noverall accuracy against F1 (EUDR/FAO definition, %):")
    ref = forest
    calls = {"national records": (d.record_origin != "without forest").astype(int),
             **{f"GFC2020 V{v}": d[f"gfc2020_v{v}"] for v in (1, 2, 3, 4)},
             "TMF 2020": d.tmf_dec2020_class.isin([1, 2]).astype(int),
             "WorldCover 2020": (d.worldcover2020_class == 10).astype(int)}
    for name, c in calls.items():
        show(name, d, known * (c == ref), known)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
