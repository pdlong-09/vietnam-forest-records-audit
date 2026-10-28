"""Paper B, Supplementary Materials tables: reads the frozen result files and writes LaTeX tables for S1, S3 and S4 to
manuscript/supplementary/tables/. Owner groups only (no names, no lot identifiers, no coordinates).

    python paper/land_audit/scripts/make_supp_b.py
"""
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/interim/land_audit"
R = ROOT / "runs/land_audit"
OUT = ROOT / "paper/land_audit/manuscript/supplementary/tables"

ORIGIN = {"natural": "Natural forest", "planted": "Planted forest", "nonforest": "Without forest"}
FUNC = {"protection": "Protection", "special": "Special use", "production": "Production", "other": "Outside planning$^{a}$"}
OWNER = {"ubnd_xa": "Commune People's Committees", "ban_quan_ly_kiem_lam": "Management boards, forest rangers",
         "vuon_quoc_gia_khu_bao_ton": "National parks, nature reserves", "doanh_nghiep": "Enterprises",
         "ho_gia_dinh_ca_nhan_khac": "Households, individuals, other", "cong_dong": "Communities",
         "luc_luong_vu_trang": "Armed forces", "to_chuc_khac": "Other organisations", "hop_tac_xa": "Cooperatives",
         "khong_ro": "Unknown"}
MAPS = {"record (natural or planted)": "National records", "GFC2020 V1": "GFC2020 V1", "GFC2020 V2": "GFC2020 V2",
        "GFC2020 V3": "GFC2020 V3", "GFC2020 V4": "GFC2020 V4", "TMF 2020 (undisturbed or degraded)": "TMF 2020",
        "WorldCover 2020 (tree cover)": "WorldCover 2020"}
LANDUSE = {"natural forest": "Natural forest", "planted forest": "Planted timber forest",
           "forest land without trees": "Forest land without trees", "perennial crops": "Perennial crops",
           "coffee (N1, N2)": "\\quad of which coffee", "rubber (N3)": "\\quad of which rubber",
           "annual crops": "Annual crops", "other": "Other land", "all": "All land uses"}
STRATA = [f"{o}|{f}" for o in ORIGIN for f in FUNC]


def rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def pct(x, d=1):
    """Percentage with a typeset minus; a rounded zero never carries a sign."""
    v = f"{100 * float(x):.{d}f}"
    if float(v) == 0:
        v = f"{0:.{d}f}"
    return v.replace("-", "$-$")


def span(r, d=1):
    """CI bounds joined by an en dash, or by 'to' when the lower bound is negative."""
    lo, hi = pct(r["lo"], d), pct(r["hi"], d)
    return f"{lo} to {hi}" if lo.startswith("$-$") else f"{lo}--{hi}"


def ci(r, d=1, k="est"):
    return f"{pct(r[k], d)} ({span(r, d)})"


def ci2(r, d=1, k="est"):
    """Estimate over its CI, for wide tables."""
    return "\\makecell{" + pct(r[k], d) + "\\\\(" + span(r, d) + ")}"


def tex(s):
    return s.replace("&", "\\&").replace("%", "\\%").replace("−", "$-$")


def table(name, caption, head, body, spec=None, small=True, notes=None):
    spec = spec or "l" + "c" * (len(head) - 1)
    lines = ["\\begin{table}[H]", f"\\caption{{{caption}\\label{{tab:{name}}}}}", "\\centering",
             "\\footnotesize" if small else "\\small", f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & ".join(f"\\textbf{{\\makecell{{{h}}}}}" if "\\\\" in h else f"\\textbf{{{h}}}" for h in head)
             + " \\\\", "\\midrule"]
    for b in body:
        lines.append("\\midrule" if b == "MID" else " & ".join(b) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines.append(f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\scriptsize {notes}}}")
    lines.append("\\end{table}")
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def s1_strata():
    st = json.load(open(D / "phase1_strata.json", encoding="utf-8"))
    p1 = {r["pid"]: r["stratum"] for r in rows(D / "phase1_points.csv")}
    gfc = collections.defaultdict(list)
    for r in rows(D / "phase1_points_gfcV4.csv"):
        gfc[p1[r["pid"]]].append(int(float(r["gfcV4"] or 0)))
    key = [r for r in rows(D / "phase2_key.csv") if r["ma_diem"].startswith("B")]
    cells = collections.Counter(r["cell"] for r in key)
    body = []
    for s in STRATA:
        o, f = s.split("|")
        v = st["strata"][s]
        g = gfc[s]
        body.append([ORIGIN[o], FUNC[f], f"{v['lots']:,}", f"{v['area_ha'] / 1e3:,.0f}", f"{v['n1']:,}",
                     pct(sum(g) / len(g)), str(cells[f"{s}|gfc_forest"]), str(cells[f"{s}|gfc_nonforest"])])
    tot_lots = sum(st["strata"][s]["lots"] for s in STRATA)
    body += ["MID", ["Total", "", f"{tot_lots:,}", f"{st['area_ha_total'] / 1e3:,.0f}", f"{st['n1']:,}",
                     pct(sum(sum(g) for g in gfc.values()) / sum(len(g) for g in gfc.values())),
                     str(sum(v for k, v in cells.items() if k.endswith('gfc_forest'))),
                     str(sum(v for k, v in cells.items() if k.endswith('gfc_nonforest')))]]
    table("s1_strata", "Phase-one strata (record origin $\\times$ forest function) and phase-two cells (stratum "
          "$\\times$ GFC2020 version~4 call).",
          ["Origin", "Function", "Lots", "Area (kha)", "Phase 1 n", "GFC V4 forest (\\%)", "Cell F n", "Cell NF n"],
          body, spec="llrrrrrr",
          notes="$^{a}$ Forest function code 0 (outside forest planning) or not recorded. Area from the lot polygons in "
                "an equal-area projection. GFC V4 forest: share of phase-one points called forest by GFC2020 version 4. "
                "Cell F/NF: phase-two points drawn in the GFC-forest and GFC-non-forest cell of the stratum.")
    return tot_lots


def s1_calibration():
    lab = rows(D / "labels_long.csv")
    cal = [r for r in lab if r["kind"] == "calibration"]
    low = sum(r["f4"] == "low" for r in cal) / len(cal)
    unk = sum(r["f1"] == "unknown" for r in cal) / len(cal)
    qc = json.load(open(R / "qc_labels.json", encoding="utf-8"))["calibration"]
    return {"pairwise": qc["mean_pairwise_f1"], "all4": qc["all_4_agree_f1"], "low": low, "unknown": unk,
            "labels": len(cal)}


def s3_landuse():
    lu = rows(R / "analysis_b/rq1_landuse.csv")
    get = {(r["domain"], r["when"], r["item"]): r for r in lu}
    for (d, w, it), r in list(get.items()):  # derived items carry no `when`; they are listed under both periods
        if not w:
            get[(d, "2020", it)] = get[(d, "now", it)] = r
    doms = [("record=natural", "Natural forest"), ("record=planted", "Planted forest"),
            ("record=nonforest", "Without forest"), ("record=natural+planted", "Natural + planted"), ("all", "All")]
    items = ["natural forest", "planted forest", "forest land without trees", "perennial crops", "coffee (N1, N2)",
             "rubber (N3)", "annual crops", "other", "MID", "forest 2020, EUDR/FAO (F1)",
             "forest 2020, Vietnam definition", "agriculture 2020 (F3 in N1-N11)"]
    names = dict(LANDUSE, **{"forest 2020, EUDR/FAO (F1)": "Forest, EUDR/FAO (F1)",
                             "forest 2020, Vietnam definition": "Forest, Vietnamese definition",
                             "agriculture 2020 (F3 in N1-N11)": "Agriculture (N1--N11)"})
    body = []
    for it in items:
        if it == "MID":
            body.append("MID")
            continue
        body.append([names[it]] + [ci2(get[(d, "2020", it)]) for d, _ in doms])
    head_area = [f"{n}\\\\({float(get[(d, '2020', 'natural forest')]['area_kha_domain']) / 1e3:.2f} Mha)"
                 for d, n in doms]
    table("s3_landuse2020", "Land use on 31 December 2020 by record class (\\% of the recorded area, 95\\% CI).",
          ["Land use"] + [h.replace("\\newline", " ") for h in head_area], body, spec="lccccc")
    items_now = ["natural forest", "planted forest", "forest land without trees", "perennial crops", "annual crops",
                 "other", "MID", "agriculture now (F7 in N1-N11)", "forest 2020 (F1) and agriculture now"]
    names_now = dict(LANDUSE, **{"agriculture now (F7 in N1-N11)": "Agriculture, latest imagery",
                                 "forest 2020 (F1) and agriculture now": "Forest in 2020, agriculture now"})
    body = []
    for it in items_now:
        if it == "MID":
            body.append("MID")
            continue
        body.append([names_now[it]] + [ci2(get[(d, "now", it)]) for d, _ in doms])
    table("s3_landuse_now", "Land use in the latest imagery (field F7) by record class (\\% of the recorded area, "
          "95\\% CI).", ["Land use"] + [h.replace("\\newline", " ") for h in head_area], body, spec="lccccc")
    body = []
    for s in STRATA:
        o, f = s.split("|")
        g = lambda it: get[(f"stratum={s}", "2020", it)]  # noqa: E731
        body.append([ORIGIN[o], FUNC[f], ci(g("forest 2020, EUDR/FAO (F1)")), ci(g("agriculture 2020 (F3 in N1-N11)")),
                     pct(g("perennial crops")["est"]), pct(g("annual crops")["est"]), pct(g("natural forest")["est"]),
                     pct(g("planted forest")["est"]), g("natural forest")["n"]])
    table("s3_strata", "Forest (EUDR/FAO) and land use in 2020 by phase-one stratum (\\% of the stratum area; 95\\% CI "
          "for the first two columns).",
          ["Origin", "Function", "Forest (F1)", "Agriculture", "Perennial", "Annual", "Natural f.", "Planted f.", "n"],
          body, spec="llccccccr",
          notes="$^{a}$ Forest function code 0 (outside forest planning) or not recorded.")


def s3_owner():
    o = rows(R / "analysis_b/rq1_owner.csv")
    by = collections.defaultdict(dict)
    for r in o:
        by[(r["record"], r["owner"])][r["item"]] = r
    body = []
    for rec, lab in [("record forest (natural + planted)", "Recorded forest"), ("record nonforest", "Without forest")]:
        if body:
            body.append("MID")
        for k, name in OWNER.items():
            d = by.get((rec, k))
            if not d:
                continue
            n = next(iter(d.values()))["n"]
            if "agriculture 2020" not in d:
                body.append([lab, name, n, "\\multicolumn{3}{c}{fewer than 30 phase-two points}"])
                continue
            body.append([lab, name, n, ci2(d["forest 2020, EUDR/FAO (F1)"]), ci2(d["natural forest 2020"]),
                         ci2(d["agriculture 2020"])])
            lab = ""
    table("s3_owner", "Forest (EUDR/FAO), natural forest and agricultural use in 2020 by owner group (\\% of the area "
          "of the owner group within the record class, 95\\% CI; human labels only).",
          ["Record class", "Owner group", "n", "Forest (F1)", "Natural forest", "Agriculture"], body,
          spec="llrccc", notes="Owner groups were assigned by fixed rules from the owner field; owner names are not "
          "reported. Groups with fewer than 30 phase-two points are reported only through the model in Table~\\ref{tab:s3_logit}.")


def s3_accuracy():
    a = rows(R / "analysis_b/rq2_accuracy.csv")
    get = {(r["reference"], r["map"], r["item"]): r for r in a}
    for ref, tag, cap in [("EUDR/FAO", "eudr", "EUDR/FAO definition"), ("Vietnam definition", "vn", "Vietnamese definition")]:
        body = []
        for m, name in MAPS.items():
            g = lambda it: get[(ref, m, it)]  # noqa: E731
            body.append([name, ci2(g("OA")), ci2(g("UA forest")), ci2(g("PA forest")), ci2(g("UA nonforest")),
                         ci2(g("PA nonforest")), pct(g("map forest share")["est"])])
        refshare = get[(ref, list(MAPS)[0], "reference forest share")]
        body += ["MID", ["Reference forest share", "", "", "", "", "", ci2(refshare)]]
        table(f"s3_accuracy_{tag}", f"Accuracy against the reference data, {cap} (\\%, area-weighted, 95\\% CI).",
              ["Source", "OA", "UA forest", "PA forest", "UA non-forest", "PA non-forest", "Forest share"], body,
              spec="lcccccc")
    body = []
    for m, name in MAPS.items():
        g = lambda it: get[("EUDR/FAO", m, it)]  # noqa: E731
        body.append([name, pct(g("p(map=1, ref=1)")["est"]), pct(g("p(map=1, ref=0)")["est"]),
                     pct(g("p(map=0, ref=1)")["est"]), pct(g("p(map=0, ref=0)")["est"])])
    table("s3_confusion", "Error matrices in area proportions, EUDR/FAO definition (\\% of the population area with a "
          "determined label).", ["Source", "Map F, ref. F", "Map F, ref. NF", "Map NF, ref. F", "Map NF, ref. NF"], body)
    pdiff = rows(R / "analysis_b/rq2_paired_differences.csv")
    pg = {(r["reference"], r["map"]): r for r in pdiff}
    body = []
    for m, name in MAPS.items():
        if m.startswith("record"):
            continue
        cells = []
        for ref in ("EUDR/FAO", "Vietnam"):
            r = pg[(ref, m)]
            cells.append(f"{float(r['diff']):+.1f} ({float(r['lo']):+.1f} to {float(r['hi']):+.1f})")
        body.append([name] + [tex(c.replace("-", "−")) for c in cells])
    table("s3_paired", "Paired differences in overall accuracy, map minus national records (percentage points, 95\\% "
          "CI), estimated on the same points.", ["Map $-$ records", "EUDR/FAO", "Vietnamese definition"], body)
    tc = rows(R / "analysis_b/rq2_tree_crops_called_forest.csv")
    tg = {(r["land use 2020"], r["map"]): r for r in tc}
    body = []
    for m, name in MAPS.items():
        body.append([name] + [ci(tg[(lu, m)]) for lu in ("perennial crops", "coffee (N1, N2)", "rubber (N3)")])
    n = {lu: tg[(lu, list(MAPS)[0])]["n"] for lu in ("perennial crops", "coffee (N1, N2)", "rubber (N3)")}
    table("s3_treecrops", "Share of the area of tree crops in 2020 called forest by each source (\\%, 95\\% CI).",
          ["Source", f"Perennial crops (n = {n['perennial crops']})", f"Coffee (n = {n['coffee (N1, N2)']})",
           f"Rubber (n = {n['rubber (N3)']})"], body)
    gv = rows(R / "analysis_b/rq2_gfc_versions.csv")
    body = []
    order = ["all", "natural forest", "planted forest", "forest land without trees", "perennial crops",
             "coffee (N1, N2)", "rubber (N3)", "annual crops", "other"]
    gg = {(r["from"], r["to"], r["land use 2020"]): r for r in gv}
    for lu in order:
        cells = []
        for fr, to in [("V1", "V2"), ("V2", "V3"), ("V3", "V4"), ("V1", "V4")]:
            r = gg[(fr, to, lu)]
            cells.append(f"{pct(r['changed'])} ({pct(r['changed towards reference'], 0)})")
        body.append([LANDUSE[lu], gg[("V1", "V2", lu)]["n"]] + cells)
    table("s3_versions", "Area changing its forest call between GFC2020 versions, by land use in 2020 (\\% of the area "
          "of the land use; in brackets, \\% of the changed area that moved towards the reference label).",
          ["Land use 2020", "n", "V1$\\to$V2", "V2$\\to$V3", "V3$\\to$V4", "V1$\\to$V4"], body, spec="lrcccc")
    us = rows(R / "analysis_b/rq2_unknown_sensitivity.csv")
    ug = {(r["unknown as"], r["map"], r["item"]): r for r in us}
    body = [["Reference forest share", pct(ug[("nonforest", "reference", "forest share")]["est"]),
             pct(get[("EUDR/FAO", list(MAPS)[0], "reference forest share")]["est"]),
             pct(ug[("forest", "reference", "forest share")]["est"])], "MID"]
    for m, name in MAPS.items():
        body.append([f"OA, {name}", pct(ug[("nonforest", m, "OA")]["est"]), pct(get[("EUDR/FAO", m, "OA")]["est"]),
                     pct(ug[("forest", m, "OA")]["est"])])
    table("s3_unknown", "Sensitivity to the 26 points left undetermined for F1 (\\%, EUDR/FAO definition).",
          ["Quantity", "Undetermined = non-forest", "Excluded (main)", "Undetermined = forest"], body)


def s3_logit():
    lg = rows(R / "analysis_b/rq3_human_logit.csv")
    lab = {"intercept": "Intercept", "malr3_cat=protection": "Protection forest (production)",
           "malr3_cat=special": "Special-use forest (production)$^{b}$",
           "malr3_cat=other": "Outside forest planning (production)$^{a}$",
           "elevation (per 100 m)": "Elevation, per 100 m", "slope (per 10 degrees)": "Slope, per 10$^\\circ$",
           "owner=vuon_quoc_gia_khu_bao_ton + malr3_cat=special":
               "National parks/reserves in special-use forest (commune, production)$^{c}$"}
    body = []
    for r in lg:
        t = r["term"]
        if t.startswith("owner=") and "+" not in t:
            name = OWNER[t[6:]] + " (commune)" + ("$^{b}$" if t.endswith("vuon_quoc_gia_khu_bao_ton") else "")
        elif t.startswith("region="):
            name = t[7:] + " (Northern midlands, mountains)"
        else:
            name = lab[t]
        orr = f"{float(r['OR']):.3f}" if float(r["OR"]) < 0.1 else f"{float(r['OR']):.2f}"
        lo, hi = (2.718281828 ** float(r["lo"]), 2.718281828 ** float(r["hi"]))
        f = (lambda x: f"{x:.3f}") if hi < 0.1 else (lambda x: f"{x:.2f}")
        body.append([name, f"{float(r['coef']):+.3f}".replace("-", "$-$"), f"{float(r['se']):.3f}", orr,
                     f"{f(lo)}--{f(hi)}", r["n_level"] if t != "intercept" else "798",
                     r["events_level"] if t != "intercept" else "116"])
    table("s3_logit", "Design-weighted logistic regression of agricultural use in 2020 on points recorded as forest "
          "(n = 798, 116 in agricultural use), all terms; reference levels in brackets.",
          ["Term", "Coef.", "SE", "OR", "95\\% CI", "n", "Agric."], body, spec="lrrrcrr",
          notes="$^{a}$ Function code 0 or not recorded. $^{b}$ 91\\% of national-park and nature-reserve points lie in "
                "special-use forest, so these two terms are not separately interpretable. $^{c}$ Sum of the two terms. "
                "Linearised variances with the phase-two cells as strata. Armed forces and other "
                "organisations have fewer than 30 points and are not interpreted on their own.")


def s3_interpreters():
    d = json.load(open(R / "analysis_b.json", encoding="utf-8"))["interpreters"]
    dc = d["double coding (random 20%)"]
    body = [["F1 (forest 2020)", ci({"est": dc["F1"]["agreement"], "lo": dc["F1"]["lo"], "hi": dc["F1"]["hi"]}),
             f"{dc['F1']['kappa']:.2f}"],
            ["F3, detailed class", ci({"est": dc["F3 code"]["agreement"], "lo": dc["F3 code"]["lo"],
                                       "hi": dc["F3 code"]["hi"]}), f"{dc['F3 code']['kappa']:.2f}"],
            ["F3, land-use group", ci({"est": dc["F3 land-use group"]["agreement"], "lo": dc["F3 land-use group"]["lo"],
                                       "hi": dc["F3 land-use group"]["hi"]}), f"{dc['F3 land-use group']['kappa']:.2f}"],
            ["F1, area-weighted", pct(dc["F1 agreement, area-weighted"]), ""]]
    table("s3_agreement", "Agreement between two independent interpretations on the 240 randomly double-interpreted "
          "points (\\%, 95\\% CI).", ["Field", "Agreement", "Cohen's $\\kappa$"], body, small=False)
    body = []
    for q, name in [("forest 2020 (F1)", "Forest share (F1)"), ("agriculture 2020 (F3)", "Agricultural share (F3)")]:
        e = d["effects"][q]
        eff = ", ".join(f"{k} {v:+.3f}" for k, v in e["effects"].items()).replace("-", "$-$")
        body.append([name, pct(e["estimate"]), f"{100 * e['se_sampling']:.2f}", f"{100 * e['se_total']:.2f}",
                     pct(e["interpreter_share_of_variance"], 0), eff])
    table("s3_interp_var", "Interpreter effects and their contribution to the variance (population level, percentage "
          "points).", ["Quantity", "Estimate", "SE sampling", "SE total", "Interp. share (\\%)", "Interpreter effects"],
          body, spec="lccccl")


def s4():
    v = json.load(open(R / "analysis_b/vlmb_accuracy.json", encoding="utf-8"))
    groups = list(v["per_class_f1"])
    cm = v["confusion (rows human, cols VLM)"]
    short = {"natural forest": "Nat.", "planted forest": "Plant.", "forest land without trees": "FL w/o trees",
             "perennial crops": "Perenn.", "annual crops": "Annual", "other": "Other"}
    # the stored matrix is keyed by the VLM-B prediction first (its column sums equal the reference counts)
    body = [[LANDUSE[g]] + [str(cm.get(h, {}).get(g, 0)) for h in groups] + [f"{v['per_class_f1'][g]:.2f}"]
            for g in groups]
    table("s4_vlm_confusion", f"VLM-B against the reference labels, out-of-fold on the 1198 phase-two points (rows: "
          f"reference; columns: VLM-B). Accuracy {v['accuracy_6']:.3f} (area-weighted {v['accuracy_6_area_weighted']:.3f}),"
          f" macro-F1 {v['macro_f1_6']:.3f}.", ["Reference"] + [short[g] for g in groups] + ["F1"], body,
          spec="lrrrrrrc")
    fe = rows(R / "analysis_b/rq3_ppi_feasibility.csv")
    body = []
    for r in fe:
        t = r["term"]
        name = ("Intercept (all recorded forest)" if t == "intercept" else OWNER[t[6:]] if t.startswith("owner=")
                else {"malr3_cat=protection": "Protection forest", "malr3_cat=special": "Special-use forest",
                      "malr3_cat=other": "Outside planning"}.get(t, t.replace("region=", "")))
        body.append([name, r["n_phase1"], r["n_phase2"], f"{float(r['vlm_share']):.3f}",
                     f"{float(r['rectifier']):+.3f}".replace("-", "$-$"), f"{float(r['target']):+.3f}".replace("-", "$-$"),
                     "yes" if r["feasible"] == "True" else "\\textbf{no}"])
    table("s4_feasibility", "Feasibility of the PPI logistic model with VLM-B ($\\lambda = 1$): rectified target "
          "share of agricultural use for each binary term (VLM-B mean prediction on phase one plus the phase-two "
          "rectifier). A finite solution requires every target in (0, 1).",
          ["Level", "n phase 1", "n phase 2", "VLM share", "Rectifier", "Target", "Feasible"], body, spec="lrrrrrc")
    ps = rows(R / "analysis_b/rq3_ppi_owner_share.csv")
    body = [[OWNER.get(r["owner"], r["owner"]), r["n_phase1"], r["n_phase2"],
             tex(f"{float(r['ppi']):.3f} ({float(r['ppi_lo']):.3f} to {float(r['ppi_hi']):.3f})".replace("-", "−")),
             tex(f"{float(r['human']):.3f} ({float(r['human_lo']):.3f} to {float(r['human_hi']):.3f})".replace("-", "−"))]
            for r in ps if r["owner"] in OWNER and r["human"]]
    table("s4_ppi_owner", "Difference estimator with VLM-B ($\\lambda = 1$) of the agricultural share on recorded "
          "forest by owner group, against the human-only ratio estimator (95\\% CI).",
          ["Owner group", "n phase 1", "n phase 2", "PPI, VLM-B", "Human only"], body, spec="lrrcc")
    b2 = json.load(open(R / "explore_b2/explore_b2.json", encoding="utf-8"))
    b4 = json.load(open(R / "explore_b4/explore_b4.json", encoding="utf-8"))
    b5 = json.load(open(R / "explore_b5/explore_b5.json", encoding="utf-8"))
    b6 = json.load(open(R / "explore_b6/explore_b6.json", encoding="utf-8"))
    b7 = json.load(open(R / "explore_b7/explore_b7.json", encoding="utf-8"))
    preds = []
    for k, name in [("V (VLM-B, as registered)", "V: VLM-B (step 1)"), ("A (AlphaEarth only)", "A: AlphaEarth 2020"),
                    ("A + V (AlphaEarth + VLM-B, main)", "A + V")]:
        e = b2[k]["record forest"]
        preds.append([name, f"{e['AUC']:.3f}", f"{e['corr']:.3f}"])
    for k, name in [("F", "F: FDP commodity probabilities"), ("S", "S: Sentinel-1/2 seasonality 2020"),
                    ("ALL", "A + V + F + S")]:
        preds.append([name, f"{b4[k]['AUC record forest']:.3f}", f"{b4[k]['corr record forest']:.3f}"])
    preds.append(["A + V trained on high-confidence labels", f"{b5['H (high-confidence training)']['AUC record forest']:.3f}",
                  f"{b5['H (high-confidence training)']['corr record forest']:.3f}"])
    for js, k, name in [(b6, "C (CNN)", "C: CNN (ResNet18, Sentinel-2 tiles)"), (b6, "C+AV (mean)", "C and A + V, mean"),
                        (b7, "H (canopy height)", "H: canopy height (Meta/WRI)"), (b7, "AV+H", "A + V + H")]:
        preds.append([name, f"{js[k]['AUC record forest']:.3f}", f"{js[k]['corr record forest']:.3f}"])
    b3 = json.load(open(R / "explore_b3/explore_b3.json", encoding="utf-8"))
    gk = next(k for k in b3 if k.startswith("G"))
    g = b3[gk]
    auc = g.get("record forest", g).get("AUC", g.get("AUC record forest"))
    cor = g.get("record forest", g).get("corr", g.get("corr record forest"))
    preds.insert(3, ["G: gradient boosting on A + V", f"{auc:.3f}", f"{cor:.3f}"])
    table("s4_predictors", "Predictors of agricultural use in 2020 on recorded forest, out-of-fold on the 798 phase-two "
          "points (all cross-fitted with the same five folds), in the order tested.",
          ["Predictor", "AUC", "Correlation with label"], preds, spec="lcc", small=False,
          notes="For every predictor, PPI++ for the multivariable logistic model of RQ3 selected $\\hat\\lambda = 0$, "
                "i.e., the human-only estimate.")
    sp = rows(R / "explore_b5/strat_ppi_shares.csv")
    sg = {(r["predictor"], r["domain"], r["method"]): r for r in sp if r["predictor"] == "AV"}
    doms = ["all record forest"] + list(OWNER)
    body = []
    for dm in doms:
        if ("AV", dm, "human") not in sg:
            continue
        h = sg[("AV", dm, "human")]
        name = "All recorded forest" if dm == "all record forest" else OWNER[dm]
        body.append([name, sg[("AV", dm, "StratPPI")]["n_phase2"],
                     ci(h), ci(sg[("AV", dm, "StratPPI")]),
                     f"{float(sg[('AV', dm, 'lambda 0')]['width_vs_human']):.3f}",
                     f"{float(sg[('AV', dm, 'PPI++')]['width_vs_human']):.3f}",
                     f"{float(sg[('AV', dm, 'StratPPI')]['width_vs_human']):.3f}"])
    table("s4_stratppi", "Agricultural share in 2020 on recorded forest by domain (\\%, 95\\% CI): human-only and "
          "stratified PPI with the A + V predictor, and confidence-interval widths relative to the human-only "
          "estimator.", ["Domain", "n ph. 2", "Human only", "Stratified PPI", "$\\lambda = 0$", "PPI++",
                                    "StratPPI"], body, spec="lrcc ccc")
    cs = rows(R / "explore_b6/cnn_strat_ppi_shares.csv")
    hs = rows(R / "explore_b7/chm_strat_ppi_shares.csv")
    w = lambda rs, p, dm: next((f"{float(r['width_vs_human']):.3f}" for r in rs if r["predictor"] == p  # noqa: E731
                                and r["domain"] == dm and r["method"] == "StratPPI"), "--")
    preds_c = sorted({r["predictor"] for r in cs})
    preds_h = sorted({r["predictor"] for r in hs})
    hi_name = "H (high-confidence training)"
    body = []
    for dm in doms:
        if ("AV", dm, "human") not in sg:
            continue
        name = "All recorded forest" if dm == "all record forest" else OWNER[dm]
        body.append([name, w(sp, "AV", dm), w(sp, hi_name, dm)] + [w(cs, p, dm) for p in preds_c if p != "AV"]
                    + [w(hs, p, dm) for p in preds_h if p != "AV"])
    heads = ["Domain", "A + V", "A + V, high conf."] + [p.split(" (")[0].replace("CpAV", "C, A + V") for p in preds_c
                                                         if p != "AV"] + [p.split(" (")[0] for p in preds_h if p != "AV"]
    table("s4_stratppi_all", "Stratified PPI confidence-interval width relative to the human-only estimator, for each "
          "predictor tested.", heads, body, spec="l" + "c" * (len(heads) - 1))
    return preds_c, preds_h


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s1_strata()
    cal = s1_calibration()
    s3_landuse()
    s3_owner()
    s3_accuracy()
    s3_logit()
    s3_interpreters()
    pc, ph = s4()
    json.dump({"calibration": cal, "cnn_predictors": pc, "chm_predictors": ph},
              open(OUT / "facts.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(cal), pc, ph)


if __name__ == "__main__":
    main()
