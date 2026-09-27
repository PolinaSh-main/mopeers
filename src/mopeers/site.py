"""Этап 9: сборка лендинга — один автономный HTML-файл `site/index.html`.

Шаблон `site/template.html` получает данные конвейера (JSON), границы МО (TopoJSON) и библиотеки d3 и
topojson-client, встроенные в файл: страница работает с диска и с GitHub Pages без внешних запросов.
"""
import json

import numpy as np
import pandas as pd
import yaml

from .config import ROOT, load_config, path
from .product import LABELS

VENDOR = {"d3": "https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js",
          "topojson": "https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/dist/topojson-client.min.js"}


def _r(v, n=4):
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return None
    return round(float(v), n)


def collect(cfg):
    out = path(cfg, "outputs")
    proc = path(cfg, "processed")
    types_cfg = yaml.safe_load((ROOT / "configs" / "types.yaml").read_text(encoding="utf-8"))
    mo = pd.read_parquet(proc / "mo.parquet")
    feats = pd.read_parquet(proc / "features.parquet")
    ctx = pd.read_parquet(proc / "context.parquet")
    types = pd.read_parquet(out / "typology" / "types.parquet").type
    lab_w = pd.read_parquet(out / "dynamics" / "labels_by_window.parquet")
    flags = pd.read_parquet(out / "dynamics" / "divergence.parquet")
    peers = pd.read_parquet(out / "dataset" / "mo_peers.parquet").set_index("territory_id")
    prof_z = pd.read_csv(out / "interpret" / "profile_z.csv", index_col=0)
    prof_m = pd.read_csv(out / "interpret" / "profile_median.csv", index_col=0)
    pat = pd.DataFrame(json.loads((out / "interpret" / "interval_patterns.json").read_text(encoding="utf-8")))
    pat["type"] = pat["type"].astype(str)
    end = feats.window_end.max()
    f = feats[feats.window_end == end].set_index("territory_id")
    ends = sorted(feats.window_end.unique())

    tdata = {}
    for code, t in types_cfg.items():
        row = pat[pat.type == code].iloc[0]
        tdata[code] = {**t, "size": int((types == code).sum()),
                       "pattern": {"intervals": row.intervals, "precision": _r(row.precision_test, 3),
                                   "recall": _r(row.recall_test, 3), "stability": _r(row.stability, 3),
                                   "extent": int(row.extent_size)},
                       "z": {k: _r(v, 3) for k, v in prof_z[code].items()},
                       "median": {k: _r(v, 4) for k, v in prof_m[code].items()}}

    mos = []
    for tid in peers.index:
        rec = json.loads((out / "dataset" / "api" / f"{tid}.json").read_text(encoding="utf-8"))
        tw = lab_w.loc[tid, ends].tolist() if tid in lab_w.index else []
        mos.append({
            "id": int(tid), "n": rec["name"], "r": rec["region"], "t": rec["type"],
            "lat": _r(mo.loc[tid, "lat"], 3), "lon": _r(mo.loc[tid, "lon"], 3),
            "full": mo.loc[tid, "full_name"], "d": int(rec["divergence"]),
            "sp": _r(rec["spend_month"], 0), "g": _r(rec["growth_2024"], 4),
            "ss": _r(f.loc[tid, "spend_to_salary"], 4) if tid in f.index else None,
            "sal": _r(ctx.salary.get(tid), 0), "pop": _r(ctx.population.get(tid), 0),
            "pay": _r(ctx.payroll_pc_month.get(tid), 0), "urb": _r(ctx.urban_share.get(tid), 3),
            "sh": {c.replace("share_", ""): _r(f.loc[tid, c], 4) for c in f.columns if c.startswith("share_")},
            "pe": [p["territory_id"] for p in rec["peers"]], "pd": [p["distance"] for p in rec["peers"]],
            "lp": rec["local_peers"],
            "pot": {m: [v["value"], v["rank"], v["of"], v["best_peer"]] for m, v in rec["potential"].items()},
            "ex": [[e["territory_id"], e["growth_2024"], [[d["feature"], d["direction"], d["z"]] for d in e["differs_by"]]]
                   for e in rec["experience"]],
            "tw": [x if isinstance(x, str) else None for x in tw],
            "mv": int(bool(flags.moved.get(tid, False))),
        })

    net = pd.read_csv(out / "network" / "rules_comparison.csv")
    grid = pd.read_csv(out / "cluster" / "icvi_grid.csv")
    synth = pd.concat([pd.read_csv(out / "cluster" / "synthetic_k_accuracy.csv", index_col=0)["all"].rename("strong"),
                       pd.read_csv(out / "cluster" / "synthetic_k_accuracy_weak.csv", index_col=0)["all"].rename("weak")],
                      axis=1)
    synth_ari = pd.concat([pd.read_csv(out / "cluster" / "synthetic_methods_ari.csv").assign(features="strong"),
                           pd.read_csv(out / "cluster" / "synthetic_methods_ari_weak.csv").assign(features="weak")])
    synth_ari = synth_ari.groupby(["method", "features", "edges"]).ari.mean().round(3)
    null = pd.read_csv(out / "cluster" / "null_structure.csv")
    thesis = pd.read_csv(out / "dataset" / "thesis_check.csv")
    seeds_p = out / "cluster" / "methods_seeds_summary.csv"
    seeds = pd.read_csv(seeds_p, header=[0, 1], index_col=0) if seeds_p.exists() else None
    xi = pd.read_csv(out / "cluster" / "kefrin_xi.csv")
    macro = pd.read_csv(out / "typology" / "icvi_macro.csv")
    final = pd.read_csv(out / "typology" / "icvi_final.csv", index_col=0).iloc[:, 0]
    trans = pd.read_csv(out / "dynamics" / "transitions_first_last.csv", index_col=0)
    centers = pd.read_parquet(out / "dynamics" / "centers_by_window.parquet")
    # сдвиг профиля типов по окнам в исходных единицах: доля маркетплейсов в тратах
    fw = feats.merge(lab_w.stack().rename("tw").reset_index().rename(columns={"level_0": "territory_id",
                                                                              "level_1": "window_end"}),
                     on=["territory_id", "window_end"], how="inner") if len(lab_w) else feats
    track = fw.groupby(["tw", "window_end"])["share_Маркетплейсы"].median().unstack()

    mob = pd.read_csv(out / "interpret" / "mobility_by_type.csv", dtype={"type": str})
    mobt = pd.read_csv(out / "interpret" / "mobility_test.csv", index_col=0).iloc[:, 0]
    stab = out / "typology" / "stability.csv"
    stab = pd.read_csv(stab)[["ari_macro", "ari_types"]].describe().round(3).to_dict() if stab.exists() else None
    nw = out / "news" / "divergence_vs_control.csv"
    news = None
    if nw.exists():
        nt = pd.read_csv(nw).set_index("category")
        nn = pd.read_parquet(out / "news" / "news.parquet")
        a = nt.loc["любое экономическое событие"]
        from .news import mo_list
        ag = pd.read_csv(out / "news" / "labeler_agreement.csv").set_index("model").kappa
        adj = pd.read_csv(out / "news" / "divergence_adjusted.csv", index_col=0)
        news = {"n_titles": len(nn), "n_mo": int(nn.territory_id.nunique()), "div": _r(a.share_divergent, 3),
                "ctrl": _r(a.share_control, 3), "p": _r(a.p_value, 3), "n_group": len(mo_list(cfg)) // 2,
                "kappa": {k: _r(v, 2) for k, v in ag.items()}, "p_adj": _r(adj.logit_p_divergent.iloc[0], 3),
                "titles_div": _r(adj.loc["расхождение", "mean_titles"], 1),
                "titles_ctrl": _r(adj.loc["контроль", "mean_titles"], 1)}
    ext = None
    if (out / "interpret" / "sme_by_type.csv").exists() and (out / "interpret" / "nightlights_by_type.csv").exists():
        by = pd.read_csv(out / "interpret" / "sme_by_type.csv", dtype={"type": str}).merge(
            pd.read_csv(out / "interpret" / "nightlights_by_type.csv", dtype={"type": str}), on="type")
        smt = pd.read_csv(out / "interpret" / "sme_test.csv").set_index(["x", "y"])
        nlt = pd.read_csv(out / "interpret" / "nightlights_test.csv").set_index("test")
        ext = {"by_type": by.round(4).to_dict("records"),
               "sme_spend": _r(smt.loc[("sme_emp_pc", "spend"), "spearman"], 3),
               "ie_spend": _r(smt.loc[("ie_per_1000", "spend"), "spearman"], 3),
               "sme_ratio": _r(smt.loc[("sme_emp_pc", "spend_to_payroll"), "spearman"], 3),
               "light_spend": _r(nlt.spearman.iloc[0], 3), "light_growth": _r(nlt.spearman.iloc[-2], 3),
               "light_flag_p": _r(nlt.p_value.iloc[-1], 3)}
    method = {
        "news": news,
        "ext": ext,
        "mobility": {"by_type": mob.round(3).to_dict("records"), "test": {k: _r(v, 6) for k, v in mobt.items()}},
        "stability": stab,
        "rules": net.round(4).to_dict("records"),
        "grid": grid.round(4).to_dict("records"),
        "synth": {k: {"strong": _r(r.strong, 3), "weak": _r(r.weak, 3)} for k, r in synth.iterrows()},
        "synth_n": {"strong": len(pd.read_csv(out / "cluster" / "synthetic_k_choice.csv")),
                    "weak": len(pd.read_csv(out / "cluster" / "synthetic_k_choice_weak.csv"))},
        "synth_ari": [{"method": m, "features": f_, "edges": e, "ari": float(v)} for (m, f_, e), v in synth_ari.items()],
        "null": null.round(4).to_dict("records"),
        "thesis": thesis.round(4).to_dict("records"),
        "seeds": None if seeds is None else {m: {k: [_r(seeds.loc[m, (k, "mean")], 4), _r(seeds.loc[m, (k, "std")], 4)]
                                                 for k in ["SW", "CH", "S_Dbw", "AVI", "AVU", "ANUI", "MQ"]}
                                             for m in seeds.index},
        "k_rule": cfg["typology"]["k_rule"],
        "xi": xi.round(4).to_dict("records"),
        "macro": macro.round(4).to_dict("records"),
        "final": {k: _r(v, 4) for k, v in final.items()},
        "trans": {"index": list(trans.index.astype(str)), "columns": list(trans.columns.astype(str)),
                  "values": trans.to_numpy().tolist()},
        "track_market": {"windows": [str(c) for c in track.columns],
                         "types": {str(k): [_r(v, 4) for v in row] for k, row in track.iterrows()}},
    }
    # находки для первого экрана — считаются из данных при сборке
    import re
    reg = mo.region_name
    same = pd.Series({i: sum(reg.get(int(x)) == reg.get(i) for x in r.split(";")) for i, r in peers.peers.items()})
    same = same[~reg.reindex(same.index).isin(["Москва", "Санкт-Петербург"])]
    first = re.search(r"\|--- (.+?) <= ([\d.]+)", (out / "interpret" / "tree_rules.txt").read_text(encoding="utf-8"))
    nums = re.findall(r": ([\d.]+)\n", (out / "interpret" / "tree_rules.txt").read_text(encoding="utf-8"))
    tree_acc, tree_cv = float(nums[0]), float(nums[1])
    findings = {"same_region_median": float(same.median()), "same_region_n": int(len(same)),
                "tree_feature": first.group(1), "tree_threshold": float(first.group(2)), "tree_acc": tree_acc,
                "tree_cv": tree_cv}
    meta = {"findings": findings, "n_mo": len(mos), "months": f"{ends[0][:4]}-01 — {ends[-1]}", "first_month": "2023-01",
            "last_month": ends[-1], "windows": ends, "n_moved": int(flags.moved.sum()),
            "n_div": int(flags.divergence.sum()), "labels": LABELS, "cfg": {
                "k": cfg["network"]["k"], "xi": cfg["typology"]["xi"], "n_peers": cfg["product"]["n_peers"],
                "radius_km": cfg["product"]["radius_km"]}}
    return {"meta": meta, "types": tdata, "mo": mos, "method": method}


def build(cfg=None):
    import urllib.request
    cfg = cfg or load_config()
    site = ROOT / "site"
    vendor = site / "vendor"
    vendor.mkdir(parents=True, exist_ok=True)
    libs = {}
    for name, url in VENDOR.items():
        p = vendor / f"{name}.min.js"
        if not p.exists():
            p.write_bytes(urllib.request.urlopen(url, timeout=60).read())
        libs[name] = p.read_text(encoding="utf-8")
    data = collect(cfg)
    geo = (path(cfg, "outputs") / "geo" / "mo.topojson").read_text(encoding="utf-8")
    html = (site / "template.html").read_text(encoding="utf-8")
    html = html.replace("/*__D3__*/", libs["d3"]).replace("/*__TOPOJSON__*/", libs["topojson"])
    html = html.replace('"__DATA__"', json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    html = html.replace('"__GEO__"', geo)
    (site / "index.html").write_text(html, encoding="utf-8")
    print("site/index.html", round(len(html.encode("utf-8")) / 1e6, 2), "МБ")
