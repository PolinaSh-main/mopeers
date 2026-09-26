"""Этап 3: сеть МО. Правило ребра задаёт сходство двух МО, разрежение оставляет k ближайших соседей.

Правила (веса рёбер в [0, 1]):
- structure  — косинусное сходство отклонений структуры трат от средней по стране;
- corr       — корреляция Пирсона месячных приростов трат (лог-разности);
- lagcorr    — максимум корреляции приростов по сдвигам ±L месяцев (один МО опережает другой);
- dtw        — сходство формы рядов трат по DTW-расстоянию (динамическое выравнивание во времени);
- road       — близость по дорожному расстоянию между центрами МО;
- fused      — среднее рангов structure и road: похожая экономика и достижимость.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from .config import load_config, path
from .data import TOTAL
from .features import SHARE_COLS


def _series(panel, ids, months):
    """Матрица МО × месяц: лог-траты; пропуски заполняются по соседним месяцам."""
    s = panel[panel.date.isin(months)].pivot(index="territory_id", columns="date", values=TOTAL)
    s = np.log(s.reindex(ids)[months]).T.interpolate(limit_direction="both").T
    return s.to_numpy()


def _corr(a, b=None):
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-12)
    b = a if b is None else (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-12)
    return a @ b.T / a.shape[1]


def similarity(rule, feats, panel, ids, months, dist, cfg):
    nc = cfg["network"]
    if rule == "structure":
        x = feats.loc[ids, SHARE_COLS].to_numpy()
        x = x - x.mean(0)
        x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        return x @ x.T
    s = _series(panel, ids, months)
    d = np.diff(s, axis=1)
    if rule == "corr":
        return _corr(d)
    if rule == "lagcorr":
        best = _corr(d)
        for lag in range(1, nc["max_lag"] + 1):
            c = _corr(d[:, lag:], d[:, :-lag])  # c[i, j]: i сдвинут на lag после j
            best = np.maximum(best, np.maximum(c, c.T))
        return best
    if rule == "dtw":
        from tslearn.metrics import cdist_dtw
        z = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-12)
        dd = cdist_dtw(z[:, :, None], global_constraint="sakoe_chiba", sakoe_chiba_radius=nc["dtw_radius"],
                       n_jobs=-1)
        return np.exp(-dd / np.median(dd[dd > 0]))
    if rule == "road":
        return np.exp(-dist / nc["road_scale_km"])
    if rule == "fused":
        a = similarity("structure", feats, panel, ids, months, dist, cfg)
        b = similarity("road", feats, panel, ids, months, dist, cfg)
        ra = a.argsort(1).argsort(1) / a.shape[1]
        rb = b.argsort(1).argsort(1) / b.shape[1]
        return (ra + rb) / 2
    raise ValueError(rule)


def knn_graph(sim, k):
    """Симметричный граф k ближайших соседей; вес ребра — сходство, приведённое к [0, 1]."""
    s = np.array(sim, dtype=float)
    np.fill_diagonal(s, -np.inf)
    n = s.shape[0]
    idx = np.argpartition(-s, k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    vals = s[rows, idx.ravel()]
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    w = (vals - lo) / (hi - lo + 1e-12) + 1e-3
    a = sparse.csr_matrix((w, (rows, idx.ravel())), shape=(n, n))
    return a.maximum(a.T)


def road_distances(cfg, ids):
    """Матрица дорожных расстояний (км) между МО; отсутствующие пары — бесконечность."""
    c = pd.read_parquet(path(cfg, "connection"))
    c = c[c.type == "highway"]
    pos = pd.Series(np.arange(len(ids)), index=ids)
    c = c[c.territory_id_x.isin(pos.index) & c.territory_id_y.isin(pos.index)]
    d = np.full((len(ids), len(ids)), np.inf)
    i, j = pos[c.territory_id_x].to_numpy(), pos[c.territory_id_y].to_numpy()
    d[i, j] = c.distance.to_numpy()
    d[j, i] = c.distance.to_numpy()
    np.fill_diagonal(d, 0)
    return d


def window_months(panel, end):
    months = sorted(panel.date.unique())
    i = months.index(end)
    return months[i - 11:i + 1]


def compare(cfg=None, end=None):
    """Сравнение правил ребра на одном окне: плотность, география, однородность соседей, Leiden, ARI."""
    import igraph as ig
    import leidenalg
    from sklearn.metrics import adjusted_rand_score

    cfg = cfg or load_config()
    proc = path(cfg, "processed")
    panel = pd.read_parquet(proc / "panel.parquet")
    feats = pd.read_parquet(proc / "features.parquet")
    mo = pd.read_parquet(proc / "mo.parquet")
    end = end or feats.window_end.max()
    f = feats[feats.window_end == end].set_index("territory_id")
    ids = [i for i in f.index if i in mo.index]
    f = f.loc[ids]
    months = window_months(panel, end)
    dist = road_distances(cfg, ids)
    from .cluster import feature_matrix
    ctx = pd.read_parquet(proc / "context.parquet")
    xdf = feature_matrix(f.reset_index(), ctx, ids, cfg)[0]
    x = xdf.to_numpy()  # те же 30 признаков, что в кластеризации
    # проверка без замкнутости: только признаки, которые не участвуют ни в одном правиле ребра
    # (контекст Росстата и доступность рынков; доли трат и ряды трат исключены)
    xc = xdf[[c for c in cfg["cluster"]["context_features"] if c in xdf.columns]].to_numpy()
    region = mo.loc[ids, "region_code"].to_numpy()
    rng = np.random.default_rng(0)
    ri, rj = rng.integers(0, len(ids), 20000), rng.integers(0, len(ids), 20000)
    base = np.abs(x[ri] - x[rj]).mean()
    base_c = np.abs(xc[ri] - xc[rj]).mean()
    rows, parts, graphs = [], {}, {}
    for rule in cfg["network"]["rules"]:
        a = knn_graph(similarity(rule, f, panel, ids, months, dist, cfg), cfg["network"]["k"])
        graphs[rule] = a
        i, j = sparse.triu(a, 1).nonzero()
        g = ig.Graph(n=len(ids), edges=list(zip(i, j)), edge_attrs={"weight": np.asarray(a[i, j]).ravel()})
        p = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights="weight", seed=0)
        parts[rule] = np.array(p.membership)
        rows.append({
            "rule": rule, "edges": len(i),
            "same_region_share": float((region[i] == region[j]).mean()),
            "feature_homophily": float(1 - np.abs(x[i] - x[j]).mean() / base),
            "context_homophily": float(1 - np.abs(xc[i] - xc[j]).mean() / base_c),
            "leiden_modularity": float(p.modularity), "leiden_communities": len(set(p.membership)),
            "median_edge_km": float(np.median(dist[i, j][np.isfinite(dist[i, j])])),
        })
    res = pd.DataFrame(rows).set_index("rule")
    ari = pd.DataFrame({r: {s: adjusted_rand_score(parts[r], parts[s]) for s in parts} for r in parts})
    out = path(cfg, "outputs") / "network"
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "rules_comparison.csv")
    ari.to_csv(out / "rules_ari.csv")
    print(res.round(3).to_string())
    print("ARI разбиений Leiden между правилами:")
    print(ari.round(2).to_string())
    return res, ari, graphs, ids
