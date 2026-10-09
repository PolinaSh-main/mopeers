"""Типология МО по окну `window_end` (по умолчанию — последнее, 2024 год).

Метод — `typology.method` (KEFRiN с весом сети ξ). Число типов K выбирается правилом `typology.rule`
из `mopeers.synth` (по умолчанию — Борда по блокам: признаковый блок SW и CH и сетевой блок MQ с равным весом;
на синтетике это правило вместе с Бордой по всем индексам точнее остальных). При `levels: 2` каждый тип
делится ещё раз тем же правилом. Типы нумеруются по возрастанию медианных трат жителя: 1 — самые низкие.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from .cluster import feature_matrix, icvi, run_method
from .config import load_config, path
from .network import knn_graph, road_distances, similarity, window_months
from .synth import choose


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
    k = choose(t, cfg["typology"]["k_rule"])
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
    spend = fw.log_spend.to_numpy()
    order = pd.Series(spend).groupby(lab1).median().sort_values().index  # по возрастанию медианных трат
    lab1 = np.array([list(order).index(v) for v in lab1])
    print("типов первого уровня:", k1, "| размеры", np.bincount(lab1).tolist())
    codes = np.empty(len(ids), dtype=object)
    level2 = []
    for m in range(k1):
        idx = np.where(lab1 == m)[0]
        sub_a = sparse.csr_matrix(a[idx][:, idx])
        if tc["levels"] == 1 or len(idx) < tc["min_split_size"]:
            codes[idx] = f"{m + 1}"
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
    t1.to_csv(out / "icvi_macro.csv", lineterminator="\n")
    if level2:
        pd.concat(level2).to_csv(out / "icvi_sub.csv", lineterminator="\n")
    # индексы качества итогового разбиения на подтипы
    final = pd.Series(pd.factorize(res.type)[0], index=ids)
    q = icvi(x, a, final.to_numpy())
    pd.Series(q).to_csv(out / "icvi_final.csv", lineterminator="\n")
    centers = xdf.groupby(res.type.to_numpy()).mean()
    centers.to_parquet(out / "centers.parquet")
    print("итоговых типов:", res.type.nunique(), "| ICVI:", {k: round(float(v), 3) for k, v in q.items()})
    return res


def stability(cfg=None):
    """Устойчивость типологии: `n_boot` подвыборок по `boot_share` МО; на каждой — та же двухуровневая процедура
    с числами кластеров итоговой типологии; согласие с итоговыми типами на общих МО — ARI (1 — полное совпадение)."""
    from sklearn.metrics import adjusted_rand_score
    cfg = cfg or load_config()
    tc = cfg["typology"]
    proc = path(cfg, "processed")
    panel = pd.read_parquet(proc / "panel.parquet")
    feats = pd.read_parquet(proc / "features.parquet")
    ctx = pd.read_parquet(proc / "context.parquet")
    mo = pd.read_parquet(proc / "mo.parquet")
    ref = pd.read_parquet(path(cfg, "outputs") / "typology" / "types.parquet")
    end = feats.window_end.max()
    f = feats[(feats.window_end == end) & feats.territory_id.isin(ref.index)]
    ids = list(f.territory_id)
    x = feature_matrix(f, ctx, ids, cfg)[0].to_numpy()
    fw = f.set_index("territory_id").loc[ids]
    a = knn_graph(similarity(tc["rule"], fw, panel, ids, window_months(panel, end), road_distances(cfg, ids), cfg),
                  cfg["network"]["k"]).tocsr()
    k1 = ref.macro.nunique()
    k2 = ref.groupby("macro").type.nunique()
    rng = np.random.default_rng(cfg.get("seed", 0))
    rows = []
    for b in range(tc["n_boot"]):
        idx = np.sort(rng.choice(len(ids), int(len(ids) * tc["boot_share"]), replace=False))
        xa, aa = x[idx], a[idx][:, idx]
        lab1 = run_method(tc["method"], k1, xa, aa, seed=b, xi=tc["xi"], n_init=cfg["cluster"]["kefrin_n_init"])
        codes = np.empty(len(idx), dtype=object)
        # макротипы подвыборки сопоставляются с итоговыми по большинству
        ref_macro = ref.macro.to_numpy()[idx]
        for m in range(k1):
            sel = np.where(lab1 == m)[0]
            major = pd.Series(ref_macro[sel]).mode().iloc[0]
            kk = int(k2.get(major, 1))
            if kk > 1 and len(sel) > kk:
                lab2 = run_method(tc["method"], kk, xa[sel], aa[sel][:, sel], seed=b, xi=tc["xi"],
                                  n_init=cfg["cluster"]["kefrin_n_init"])
            else:
                lab2 = np.zeros(len(sel), dtype=int)
            codes[sel] = [f"{m}.{s}" for s in lab2]
        rows.append({"boot": b, "ari_macro": adjusted_rand_score(ref_macro, lab1),
                     "ari_types": adjusted_rand_score(ref.type.to_numpy()[idx], codes)})
        print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(path(cfg, "outputs") / "typology" / "stability.csv", index=False, lineterminator="\n")
    print(t[["ari_macro", "ari_types"]].describe().round(3).to_string())
    return t
