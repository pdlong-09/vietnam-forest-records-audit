"""Paper B: Supplementary Section S5 tables (checks added for the PeerJ version) from robustness_b.py and verify_b.py.

    .venv/Scripts/python paper/land_audit/scripts/robustness_b.py
    .venv/Scripts/python paper/land_audit/scripts/make_supp_s5.py
"""
import json
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_b  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RB = ROOT / "runs/land_audit/robustness_b"
TAB = ROOT / "paper/land_audit/manuscript/supplementary/tables"
OWNER = {"ubnd_xa": "communes", "doanh_nghiep": "enterprises", "ho_gia_dinh_ca_nhan_khac": "households",
         "vuon_quoc_gia_khu_bao_ton": "national parks, reserves", "ban_quan_ly_kiem_lam": "management boards"}
MAPS = {"record (natural or planted)": "national records", "TMF 2020 (undisturbed or degraded)": "TMF 2020",
        "WorldCover 2020 (tree cover)": "WorldCover 2020"}


def f(x, nd=1):
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)).replace("-", "$-$")


def ci(lo, hi, s=100):
    return f"{f(lo * s)}--{f(hi * s)}"


def table(name, caption, colspec, header, rows, note=None):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    tex = (f"\\begin{{table}}[H]\n\\caption{{{caption}\\label{{tab:{name}}}}}\n\\centering\n\\small\n"
           f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n")
    if note:
        tex += f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\scriptsize {note}}}\n"
    (TAB / f"{name}.tex").write_text(tex + "\\end{table}\n", encoding="utf-8")


def main():
    h = pd.read_csv(RB / "interpreter_variance_headline.csv")
    rows = []
    for r in h.itertuples():
        dom = r.domain
        for k, v in OWNER.items():
            dom = dom.replace(f"record forest, owner {k}", f"record forest: {v}")
        dom = MAPS.get(dom, dom)
        item = {"EUDR forest share": "EUDR forest share", "agriculture share": "Agricultural share",
                "overall accuracy (EUDR)": "Overall accuracy", "paired OA difference vs record (EUDR)": "OA difference vs records"}[r.item]
        rows.append([item, dom, f(r.est * 100), ci(r.lo_sampling, r.hi_sampling), ci(r.lo_total, r.hi_total),
                     f(100 * r.se_interpreter ** 2 / r.se_total ** 2, 0)])
    table("s5_interp_total", "Headline estimates with 95\\% confidence intervals from sampling variance only (as in the main tables) and with the interpreter component added (total). EUDR/FAO definition; percentages.",
          "llcccc", "\\textbf{Estimate} & \\textbf{Domain or source} & \\textbf{Est.} & \\textbf{CI, sampling} & \\textbf{CI, total} & \\textbf{Interp. \\%}",
          rows, "Interp. \\%: share of the total variance due to interpreters. Interpreter effects $b_j$ (label = truth + $b_j$) are estimated from the 240 random double-coded pairs; an estimate whose labels shift by $g_i b_{j(i)}$ gains the variance $\\sigma_b^2 \\sum_j c_j^2$, $c_j = \\sum_{i \\in j} w_i g_i / \\sum_i w_i x_i$ ($g = x$ for shares, $x(2m-1)$ for the overall accuracy of map $m$, $2x(m_A - m_B)$ for paired differences). Consensus labels carry no interpreter effect. Lower limits below 0 are those of the normal approximation.")
    js = json.loads((RB / "robustness_b.json").read_text(encoding="utf-8"))
    ag = js["agreement_by_group"]
    rows = [[g, str(v["points (either interpreter)"]), f(v["specific agreement"], 2)] for g, v in ag.items() if "specific agreement" in v]
    a = ag["agriculture yes/no"]
    rows.append(["agriculture (yes/no)", "--", f(a["specific agreement (agriculture)"], 2)])
    table("s5_agreement_groups", "Agreement between the two interpreters of the 240 random double-coded points, by land-use group in 2020.",
          "lcc", "\\textbf{Land-use group (F3)} & \\textbf{Points (either interpreter)} & \\textbf{Specific agreement}",
          rows, f"Specific agreement for a class: $2a/(2a + b + c)$, with $a$ the points both interpreters put in the class and $b$, $c$ those only one did. For agriculture yes/no, overall agreement {f(a['agreement'] * 100)}\\% and Cohen's $\\kappa = {f(a['kappa'], 2)}$. On the 78 double-coded points of lots recorded as planted forest, both interpreters chose planted forest for 2 points, and one chose natural and the other planted forest for 7.")
    lc = pd.read_csv(RB / "low_confidence_sensitivity.csv")
    rows = []
    name = {"EUDR forest share": "EUDR forest share", "agriculture share": "Agricultural share", "OA": "Overall accuracy"}
    for (dom, item), g in lc.groupby(["domain", "item"], sort=False):
        allr = g[g.labels == "all labels"].iloc[0]
        sub = g[g.labels != "all labels"].iloc[0]
        rows.append([name[item], MAPS.get(dom, dom), f"{f(allr.est * 100)} ({ci(allr.lo, allr.hi)})", f"{f(sub.est * 100)} ({ci(sub.lo, sub.hi)})"])
    table("s5_lowconf", f"Sensitivity of the main estimates to low-confidence labels: all 1198 labels, and without the {js['single_low_confidence_points']} points whose final label rests on a single low-confidence interpretation (treated as missing at random within their phase-two cell). Percentages with 95\\% CI.",
          "llcc", "\\textbf{Estimate} & \\textbf{Domain or source} & \\textbf{All labels} & \\textbf{Without single low-confidence}", rows)
    im = js["imagery"]
    rows = [["Interpretations with a parsable acquisition date", f"{im['with a parsable date']} of {im['main interpretations']}"],
            ["Median absolute gap to 31 December 2020", f"{f(im['median absolute gap (months)'], 0)} months"],
            ["Within 6 / 12 / 24 months", f"{f(im['within 6 months'] * 100)}\\% / {f(im['within 12 months'] * 100)}\\% / {f(im['within 24 months'] * 100)}\\%"],
            ["Acquired in or before December 2020", f"{f(im['before 2021'] * 100)}\\%"],
            ["Issue ``no imagery close to 2020'' recorded (F5)", f"{f(im['share citing \'no imagery close to 2020\' (F5)'] * 100)}\\%"],
            ["Low confidence recorded (F4)", f"{f(im['low confidence share'] * 100)}\\%"]]
    table("s5_imagery", "Reference imagery used for the 2020 label: the first acquisition date noted in F8 of each main interpretation.",
          "lc", "\\textbf{Item} & \\textbf{Value}", rows)
    t, checks = verify_b.main(str(ROOT / "release/land_audit_kaggle"), str(ROOT / "runs/land_audit/analysis_b"))
    rows = []
    for r in t.itertuples():
        pub = "--" if pd.isna(r._8) else f"{f(r._8 * 100)} ({ci(r._9, r._10)})"
        rows.append([r.estimate.replace("commune People's Committees", "communes"), pub, f"{f(r.est * 100)} ({ci(r.lo, r.hi)})",
                     f(r.se_analytic * 100, 2), f(r.se_bootstrap * 100, 2)])
    table("s5_verify", "Independent recomputation of key estimates from the public reference sample (verify\\_b.py in the code release): published values, the independent analytic estimate (Cochran's form of the two-phase variance) and the two-phase bootstrap standard error (2000 replicates). Percentages.",
          "p{5.4cm}cccc", "\\textbf{Estimate} & \\textbf{Published} & \\textbf{Independent} & \\textbf{SE} & \\textbf{SE, bootstrap}",
          rows, f"Design weights recomputed from the strata and the phase-one cell counts differ from the released weights by at most {checks['max relative difference to weight_ha']:.0e} (relative) and sum to the population area ({checks['weights sum (Mha)']:.2f} Mha). The paired difference GFC2020 V4 $-$ records is published in Table 3 of the article: +6.7 (4.4 to 9.0).")
    print("written", [p.name for p in TAB.glob("s5_*.tex")])


if __name__ == "__main__":
    main()
