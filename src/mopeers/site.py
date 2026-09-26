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
    fca = pd.DataFrame(json.loads((out / "interpret" / "fca_descriptions.json").read_text(encoding="utf-8")))
    fca["type"] = fca["type"].astype(str)
    end = feats.window_end.max()
    f = feats[feats.window_end == end].set_index("territory_id")
    ends = sorted(feats.window_end.unique())

    tdata = {}
    for code, t in types_cfg.items():
        row = fca[fca.type == code].iloc[0]
        tdata[code] = {**t, "size": int((types == code).sum()),
                       "fca": {"intent": row.intent, "precision": _r(row.precision, 3), "recall": _r(row.recall, 3),
                               "f1": _r(row.f1, 3), "extent": int(row.extent_size)},
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
    synth = pd.read_csv(out / "cluster" / "synthetic_k_accuracy.csv", index_col=0).iloc[:, 0]
    xi = pd.read_csv(out / "cluster" / "kefrin_xi.csv")
    macro = pd.read_csv(out / "typology" / "icvi_macro.csv")
    sub = pd.read_csv(out / "typology" / "icvi_sub.csv")
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
    method = {
        "mobility": {"by_type": mob.round(3).to_dict("records"), "test": {k: _r(v, 6) for k, v in mobt.items()}},
        "stability": stab,
        "rules": net.round(4).to_dict("records"),
        "grid": grid.round(4).to_dict("records"),
        "synth": {k: _r(v, 3) for k, v in synth.items()},
        "xi": xi.round(4).to_dict("records"),
        "macro": macro.round(4).to_dict("records"), "sub": sub.round(4).to_dict("records"),
        "final": {k: _r(v, 4) for k, v in final.items()},
        "trans": {"index": list(trans.index.astype(str)), "columns": list(trans.columns.astype(str)),
                  "values": trans.to_numpy().tolist()},
        "track_market": {"windows": [str(c) for c in track.columns],
                         "types": {str(k): [_r(v, 4) for v in row] for k, row in track.iterrows()}},
    }
    meta = {"n_mo": len(mos), "months": f"{ends[0][:4]}-01 — {ends[-1]}", "first_month": "2023-01",
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
