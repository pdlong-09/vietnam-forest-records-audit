"""Paper B figures for the Land manuscript (Figures 2-6; Figure 1 is TikZ in manuscript/figures/fig1_design.tex).

    .venv/Scripts/python paper/land_audit/scripts/make_figures_b.py

Reads the analysis outputs (runs/land_audit/analysis_b/, explore_b5/) and, for Figure 2 and the paired differences in
Figure 4, the phase-2 key and analysis_b.load(). Writes manuscript/figures/fig<N>_<name>.pdf (vector) and .png (300 dpi).
Palette: the validated categorical slots (dataviz reference palette, light mode), assigned in fixed order.
"""
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis_b as A  # noqa: E402

matplotlib.use("Agg")
OUT = A.ROOT / "paper/land_audit/manuscript/figures"
RES = A.ROOT / "runs/land_audit/analysis_b"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "savefig.facecolor": SURF, "axes.facecolor": SURF,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", name)


def clean(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ---------- Figure 2: sample map ----------
def fig2_map():
    from shapely.geometry import shape
    g = shape(json.load(open(A.D / "vn_boundary_lsib.geojson"))["geometry"])
    key = pd.read_csv(A.D / "phase2_key.csv", usecols=["ma_diem", "pid", "lon", "lat", "nggocr"])
    key = key[key.ma_diem.str.startswith("B")]
    rec = key.nggocr.map({1: "Natural forest", 2: "Planted forest"}).fillna("Without forest")
    fig, ax = plt.subplots(figsize=(8.2 * CM, 12.5 * CM))
    polys = list(getattr(g, "geoms", [g]))
    for p in polys:
        for q in (p.geoms if hasattr(p, "geoms") else [p]):
            if q.geom_type == "Polygon":
                x, y = q.exterior.xy
                ax.fill(x, y, color="#f0efec", edgecolor="#9a9893", linewidth=0.5, zorder=1)
    for i, (lab, m) in enumerate((("Natural forest", rec == "Natural forest"), ("Planted forest", rec == "Planted forest"),
                                   ("Without forest", rec == "Without forest"))):
        ax.scatter(key.lon[m], key.lat[m], s=5, color=SLOTS[i], edgecolor="white", linewidth=0.25, zorder=3,
                   label=f"{lab} (n = {int(m.sum())})")
    ax.set_xlim(101.9, 110.0)
    ax.set_ylim(8.2, 23.6)
    ax.set_aspect(1 / np.cos(np.radians(16)))
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(title="Record class (phase-two points)", loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=1,
              frameon=False, markerscale=2.2, handletextpad=0.3, title_fontsize=7.5)
    save(fig, "fig2_sample_map")


# ---------- Figure 3: land use behind each record class (RQ1) ----------
def fig3_rq1():
    t = pd.read_csv(RES / "rq1_landuse.csv")
    t = t[(t["when"] == "2020")]
    groups = ["natural forest", "planted forest", "forest land without trees", "perennial crops", "annual crops", "other"]
    names = ["Natural forest", "Planted timber forest", "Forest land without trees", "Perennial crops", "Annual crops", "Other land"]
    rows = [("record=natural", "Recorded natural forest\n(9.65 Mha)"), ("record=planted", "Recorded planted forest\n(3.39 Mha)"),
            ("record=nonforest", "Recorded without forest\n(5.17 Mha)")]
    fig, ax = plt.subplots(figsize=(16 * CM, 5.6 * CM))
    for r, (dom, lab) in enumerate(rows):
        left = 0.0
        for k, g in enumerate(groups):
            v = float(t[(t.domain == dom) & (t.item == g)].est.iloc[0]) * 100
            ax.barh(r, v, left=left, height=0.62, color=SLOTS[k], edgecolor="white", linewidth=1.0)
            if v >= 6:
                ax.text(left + v / 2, r, f"{v:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if k in (0, 1, 5) else INK)
            left += v
    ax.set_yticks(range(len(rows)), [lab for _, lab in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of the recorded area in 2020 (%)")
    clean(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SLOTS[k]) for k in range(len(groups))]
    ax.legend(handles, names, ncol=3, loc="upper center", bbox_to_anchor=(0.42, -0.3), frameon=False, handlelength=1.2,
              columnspacing=1.2)
    save(fig, "fig3_rq1_landuse")


# ---------- Figure 4: accuracy and paired differences (RQ2) ----------
def paired_differences():
    d, _ = A.load()
    rows = []
    for rname, ref in {"EUDR/FAO": (d.f1 == "forest").astype(float).where(d.known == 1), "Vietnam": d.forest_vn}.items():
        known = ref.notna().astype(float).to_numpy()
        r = np.nan_to_num(ref.to_numpy(float))
        calls = {k: np.asarray(v, float) for k, v in A.map_calls(d).items()}
        ok = {k: known * (v == r) for k, v in calls.items()}
        base = ok["record (natural or planted)"]
        for k in calls:
            if k.startswith("record"):
                continue
            res = A.ratio(d, ok[k] - base, known)
            rows.append({"reference": rname, "map": k, "diff": 100 * res["est"], "lo": 100 * (res["est"] - A.Z * res["se"]),
                         "hi": 100 * (res["est"] + A.Z * res["se"])})
    t = pd.DataFrame(rows)
    t.to_csv(RES / "rq2_paired_differences.csv", index=False)
    return t


def fig4_rq2():
    a = pd.read_csv(RES / "rq2_accuracy.csv")
    a = a[(a.reference == "EUDR/FAO") & (a.item == "OA")]
    label = {"record (natural or planted)": "National records", "GFC2020 V1": "GFC2020 V1", "GFC2020 V2": "GFC2020 V2",
             "GFC2020 V3": "GFC2020 V3", "GFC2020 V4": "GFC2020 V4", "TMF 2020 (undisturbed or degraded)": "TMF 2020",
             "WorldCover 2020 (tree cover)": "WorldCover 2020"}
    order = ["GFC2020 V4", "GFC2020 V3", "WorldCover 2020 (tree cover)", "record (natural or planted)", "GFC2020 V1",
             "GFC2020 V2", "TMF 2020 (undisturbed or degraded)"]
    p = paired_differences()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16 * CM, 6.4 * CM), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.55})
    for i, m in enumerate(order):
        row = a[a["map"] == m].iloc[0]
        rec = m.startswith("record")
        ax1.errorbar(row.est * 100, i, xerr=[[100 * (row.est - row.lo)], [100 * (row.hi - row.est)]], fmt="o", ms=4.5,
                     color=INK if rec else SLOTS[0], ecolor=INK2 if rec else SLOTS[0], elinewidth=1.2, capsize=0)
        ax1.text(row.hi * 100 + 1.2, i, f"{row.est * 100:.1f}", va="center", fontsize=7, color=INK2)
    ax1.set_yticks(range(len(order)), [label[m] for m in order])
    ax1.invert_yaxis()
    ax1.set_xlim(50, 95)
    ax1.set_xlabel("Overall accuracy, EUDR/FAO (%)")
    ax1.set_title("(a) Agreement with the reference data", loc="left")
    clean(ax1)
    maps = [m for m in order if not m.startswith("record")]
    for j, (rname, mk, col, off) in enumerate((("EUDR/FAO", "o", SLOTS[0], -0.14), ("Vietnam", "s", SLOTS[1], 0.14))):
        for i, m in enumerate(maps):
            row = p[(p.reference == rname) & (p["map"] == m)].iloc[0]
            ax2.errorbar(row["diff"], i + off, xerr=[[row["diff"] - row.lo], [row.hi - row["diff"]]], fmt=mk, ms=4,
                         color=col, ecolor=col, elinewidth=1.1, capsize=0, label=("EUDR/FAO definition" if j == 0 else
                                                                                 "Vietnamese definition") if i == 0 else None)
    ax2.axvline(0, color=INK2, linewidth=0.8)
    ax2.set_yticks(range(len(maps)), [label[m] for m in maps])
    ax2.invert_yaxis()
    ax2.set_xlabel("Difference in overall accuracy, map − records (points)")
    ax2.set_title("(b) Paired difference with the national records", loc="left")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.2)
    ax2.set_xlim(-31, 12)
    clean(ax2)
    save(fig, "fig4_rq2_accuracy")


# ---------- Figure 5: odds ratios (RQ3) ----------
def fig5_rq3():
    t = pd.read_csv(RES / "rq3_human_logit.csv").set_index("term")
    rows = [("malr3_cat=protection", "Protection forest", "Forest function (vs production)"),
            ("malr3_cat=other", "Outside forest planning", "Forest function (vs production)"),
            ("owner=vuon_quoc_gia_khu_bao_ton + malr3_cat=special", "National park/reserve in special-use forest¹", "Forest function (vs production)"),
            ("owner=ban_quan_ly_kiem_lam", "Management boards", "Owner group (vs commune)"),
            ("owner=doanh_nghiep", "Enterprises", "Owner group (vs commune)"),
            ("owner=ho_gia_dinh_ca_nhan_khac", "Households and individuals", "Owner group (vs commune)"),
            ("owner=cong_dong", "Communities", "Owner group (vs commune)"),
            ("elevation (per 100 m)", "Elevation (per 100 m)", "Terrain"),
            ("slope (per 10 degrees)", "Slope (per 10°)", "Terrain")]
    fig, ax = plt.subplots(figsize=(12 * CM, 7.4 * CM))
    ylabels, y, prev = [], 0, None
    for term, name, grp in rows:
        if grp != prev:
            if prev is not None:
                y += 0.5
            ax.text(0.0035, y - 0.55, grp, fontsize=7, color=INK2, style="italic", va="center")
            y += 0.45
            prev = grp
        r = t.loc[term]
        sig = (r.lo > 0) or (r.hi < 0)
        col = SLOTS[0] if sig else "#9a9893"
        ax.errorbar(np.exp(r.coef), y, xerr=[[np.exp(r.coef) - np.exp(r.lo)], [np.exp(r.hi) - np.exp(r.coef)]], fmt="o",
                    ms=4.5, color=col, ecolor=col, elinewidth=1.2, capsize=0)
        fmt = (lambda v: f"{v:.3f}") if np.exp(r.hi) < 0.1 else (lambda v: f"{v:.2f}")
        ax.text(60, y, f"{fmt(np.exp(r.coef))} ({fmt(np.exp(r.lo))}–{fmt(np.exp(r.hi))})", va="center", fontsize=6.8, color=INK2)
        ylabels.append((y, name))
        y += 1
    ax.axvline(1, color=INK2, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.003, 30)
    ax.set_xticks([0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
    ax.minorticks_off()
    ax.set_yticks([v for v, _ in ylabels], [n for _, n in ylabels])
    ax.invert_yaxis()
    ax.set_xlabel("Odds ratio of agricultural use in 2020 (log scale)")
    clean(ax)
    ax.text(60, -0.45, "OR (95% CI)", fontsize=6.8, color=INK2, va="center", fontweight="bold")
    save(fig, "fig5_rq3_odds_ratios")


# ---------- Figure 6: AI-assisted estimation (exploratory) ----------
def fig6_ai():
    t = pd.read_csv(A.ROOT / "runs/land_audit/explore_b5/strat_ppi_shares.csv")
    t = t[t.predictor == "AV"]
    doms = [("all record forest", "All recorded forest"), ("ban_quan_ly_kiem_lam", "Management boards"),
            ("doanh_nghiep", "Enterprises"), ("vuon_quoc_gia_khu_bao_ton", "National parks and reserves"),
            ("ubnd_xa", "Commune People's Committees")]
    meths = [("lambda 0", "Phase-one area only"), ("PPI++", "PPI++ (one λ)"), ("StratPPI", "Stratified PPI")]
    fig, ax = plt.subplots(figsize=(12 * CM, 6.6 * CM))
    h = 0.26
    for j, (m, mlab) in enumerate(meths):
        for i, (d, _) in enumerate(doms):
            v = float(t[(t.domain == d) & (t.method == m)].width_vs_human.iloc[0])
            ax.barh(i + (j - 1) * h, v, height=h * 0.92, color=SLOTS[j], edgecolor="white", linewidth=0.8,
                    label=mlab if i == 0 else None)
            if m == "StratPPI":
                ax.text(v - 0.012, i + (j - 1) * h, f"{v:.2f}", va="center", ha="right", fontsize=6.8, color=INK)
    ax.axvline(1, color=INK2, linewidth=0.8)
    ax.text(0.99, -0.72, "human-only = 1", fontsize=6.8, color=INK2, ha="right", va="center")
    ax.set_yticks(range(len(doms)), [lab for _, lab in doms])
    ax.set_ylim(len(doms) - 0.5, -0.95)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Width of the 95% CI relative to the human-only estimator")
    clean(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=3, frameon=False, handlelength=1.2, columnspacing=1.0)
    save(fig, "fig6_ai_ci_width")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig2_map()
    fig3_rq1()
    fig4_rq2()
    fig5_rq3()
    fig6_ai()
