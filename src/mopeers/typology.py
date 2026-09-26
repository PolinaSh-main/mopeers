"""Двухуровневая типология МО по окну `window_end` (по умолчанию — последнее, 2024 год).

Уровень 1 — макротипы; уровень 2 — подтипы внутри каждого макротипа. На обоих уровнях метод один
(`typology.method`, по умолчанию KEFRiN с весом сети ξ), число кластеров выбирается голосованием SW, CH и MQ
(правило из `mopeers.synth`: эти три индекса находят истинное K на синтетике; при равенстве голосов — меньшее K).
Подтипы получают коды «<макротип>.<подтип>», нумерация — по убыванию размера.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from .cluster import feature_matrix, icvi, run_method
from .config import load_config, path
from .network import knn_graph, road_distances, similarity, window_months

VOTERS = ["SW", "CH", "MQ"]


def choose_k(x, a, ks, method, cfg):
    cc = cfg["cluster"]
    rows, labs = [], {}
    for k in ks:
        if k >= x.shape[0]:
            continue
        lab = run_method(method, k, x, a, seed=cfg.get("seed", 0), xi=cfg["typology"]["xi"], n_init=cc["kefrin_n_init"])
        if len(set(lab)) < 2:
            continue
        labs[k] = lab
        rows.append({"k": k, **icvi(x, a, lab)})
    t = pd.DataFrame(rows).set_index("k")
    votes = pd.Series([t[m].idxmax() for m in VOTERS]).value_counts()
    k = int(votes[votes == votes.max()].index.min())
    return k, labs[k], t


def _order_by_size(lab):
    sizes = pd.Series(lab).value_counts()
    remap = {old: new for new, old in enumerate(sizes.index)}
    return np.array([remap[v] for v in lab])


def build(cfg=None, end=None):
    cfg = cfg or load_config()
    tc = cfg["typology"]
    proc = path(cfg, "processed")
    panel = pd.read_parquet(proc / "panel.parquet")
    feats = pd.read_parquet(proc / "features.parquet")
    ctx = pd.read_parquet(proc / "context.parquet")
    mo = pd.read_parquet(proc / "mo.parquet")
    end = end or feats.window_end.max()
    f = feats[(feats.window_end == end) & feats.territory_id.isin(mo.index)]
    ids = list(f.territory_id)
    xdf, cols = feature_matrix(f, ctx, ids, cfg)
    x = xdf.to_numpy()
    fw = f.set_index("territory_id").loc[ids]
    a = knn_graph(similarity(tc["rule"], fw, panel, ids, window_months(panel, end), road_distances(cfg, ids), cfg),
                  cfg["network"]["k"])
    k1, lab1, t1 = choose_k(x, a, tc["k_macro"], tc["method"], cfg)
    lab1 = _order_by_size(lab1)
    print("макротипов:", k1, "| размеры", np.bincount(lab1).tolist())
    codes = np.empty(len(ids), dtype=object)
    level2 = []
    for m in range(k1):
        idx = np.where(lab1 == m)[0]
        sub_a = sparse.csr_matrix(a[idx][:, idx])
        if len(idx) < tc["min_split_size"]:
            codes[idx] = f"{m + 1}.1"
            continue
        k2, lab2, t2 = choose_k(x[idx], sub_a, tc["k_sub"], tc["method"], cfg)
        lab2 = _order_by_size(lab2)
        codes[idx] = [f"{m + 1}.{s + 1}" for s in lab2]
        t2 = t2.assign(macro=m + 1)
        level2.append(t2)
        print(f"  макротип {m + 1}: подтипов {k2} | размеры {np.bincount(lab2).tolist()}")
    out = path(cfg, "outputs") / "typology"
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame({"territory_id": ids, "macro": lab1 + 1, "type": codes}).set_index("territory_id")
    res.to_parquet(out / "types.parquet")
    t1.to_csv(out / "icvi_macro.csv")
    if level2:
        pd.concat(level2).to_csv(out / "icvi_sub.csv")
    # индексы качества итогового разбиения на подтипы
    final = pd.Series(pd.factorize(res.type)[0], index=ids)
    q = icvi(x, a, final.to_numpy())
    pd.Series(q).to_csv(out / "icvi_final.csv")
    centers = xdf.groupby(res.type.to_numpy()).mean()
    centers.to_parquet(out / "centers.parquet")
    print("итоговых типов:", res.type.nunique(), "| ICVI:", {k: round(float(v), 3) for k, v in q.items()})
    return res
