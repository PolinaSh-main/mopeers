"""Этап 4: методы кластеризации и внутренние индексы качества (ICVI).

Методы:
- kefrin   — KEFRiN (Shalileh, Mirkin 2022): k-средних сразу по признакам МО и по строкам матрицы сети;
- leiden   — Leiden по сети, разрешение подбирается под заданное K;
- spectral — спектральная кластеризация по сети;
- kmeans, gmm, ward — только по признакам, без сети.

ICVI (направление «лучше»): SW ↑, CH ↑, S_Dbw ↓ — по признакам; AVI ↑, AVU ↓, ANUI ↑, MQ (модульность) ↑ — по сети.
SW, CH, AVI, AVU, ANUI и модульность считаются кодом библиотеки Pattern (Шалилех), S_Dbw — пакетом s_dbw.
"""
import numpy as np
import pandas as pd
from scipy import sparse

from .config import load_config, path
from .third_party import load

HIGHER_BETTER = {"SW": True, "CH": True, "S_Dbw": False, "AVI": True, "AVU": False, "ANUI": True, "MQ": True}
FEATURE_METHODS = {"kmeans", "gmm", "ward"}


def feature_matrix(feats, ctx, ids, cfg):
    """Стандартизованная матрица признаков для кластеризации.

    Пропуск заполняется средним по `impute_k` МО, ближайшим по известным признакам (KNNImputer, стандартизованные
    единицы): МО без данных Росстата получает значение похожих МО, а не медиану страны.
    """
    if len(set(ids)) == len(ids):
        f = feats.set_index("territory_id").loc[ids, cfg["cluster"]["window_features"]]
    else:  # несколько окон: строки берутся в порядке таблицы, ids = feats.territory_id
        f = feats[cfg["cluster"]["window_features"]]
    c = ctx.reindex(ids)[cfg["cluster"]["context_features"]]
    x = pd.concat([f.reset_index(drop=True), c.reset_index(drop=True)], axis=1)
    x.index = list(ids)
    for col in cfg["cluster"].get("log_features", []):
        x[col] = np.log(x[col])
    x = x.replace([np.inf, -np.inf], np.nan)
    from sklearn.impute import KNNImputer
    z = (x - x.mean()) / x.std()
    z = pd.DataFrame(KNNImputer(n_neighbors=cfg["cluster"]["impute_k"]).fit_transform(z), index=z.index, columns=z.columns)
    z = z.clip(-cfg["cluster"]["clip_z"], cfg["cluster"]["clip_z"])
    return z, list(x.columns)


def _relabel(labels):
    return np.unique(labels, return_inverse=True)[1]


def run_method(method, k, x, a, seed=0, xi=1.0, n_init=3):
    if method == "kefrin":
        kf = load("kefrin", "kefrin")
        conf = kf.KEFRiNConfig(n_clusters=k, rho=1.0, xi=xi, distance_metric=kf.DistanceMetric.EUCLIDEAN,
                               random_state=seed, n_init=n_init,
                               preprocessing_y=kf.PreprocessingMethod.NONE,
                               preprocessing_p=kf.PreprocessingMethod.NONE)
        return _relabel(kf.KEFRiN(conf).fit_predict(x, a.toarray()))
    if method == "leiden":
        return _leiden_k(a, k, seed)
    if method == "spectral":
        from sklearn.cluster import SpectralClustering
        return SpectralClustering(k, affinity="precomputed", random_state=seed).fit_predict(a)
    if method == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(k, n_init=10, random_state=seed).fit_predict(x)
    if method == "gmm":
        from sklearn.mixture import GaussianMixture
        return _relabel(GaussianMixture(k, covariance_type="diag", random_state=seed).fit_predict(x))
    if method == "ward":
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(k, linkage="ward").fit_predict(x)
    raise ValueError(method)


def _leiden_k(a, k, seed):
    """Leiden с RB-модульностью; разрешение ищется бисекцией, чтобы получить K сообществ (или ближайшее)."""
    import igraph as ig
    import leidenalg
    i, j = sparse.triu(a, 1).nonzero()
    g = ig.Graph(n=a.shape[0], edges=list(zip(i, j)), edge_attrs={"weight": np.asarray(a[i, j]).ravel()})
    lo, hi, best = 0.01, 5.0, None
    for _ in range(30):
        r = (lo + hi) / 2
        p = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                     resolution_parameter=r, seed=seed)
        m = np.array(p.membership)
        # мелкие сообщества (< 1% узлов) присоединяются к самому связанному крупному
        m = _merge_small(m, a, max(5, a.shape[0] // 100))
        n = len(set(m))
        if best is None or abs(n - k) < abs(len(set(best)) - k):
            best = m
        if n == k:
            break
        lo, hi = (r, hi) if n < k else (lo, r)
    return _relabel(best)


def _merge_small(m, a, min_size):
    m = m.copy()
    sizes = pd.Series(m).value_counts()
    big = sizes[sizes >= min_size].index.to_numpy()
    for c in sizes[sizes < min_size].index:
        nodes = np.where(m == c)[0]
        w = np.asarray(a[nodes].sum(0)).ravel()
        score = [w[m == b].sum() for b in big]
        m[nodes] = big[int(np.argmax(score))] if len(big) else c
    return m


def icvi(x, a, labels):
    pm = load("pattern", "clustering_metrics")
    from s_dbw import S_Dbw
    labels = _relabel(labels)
    fm = pm.FeaturesClusteringMetrics().get_metrics(x, labels)
    am = pm.AdjacencyClusteringMetrics().get_metric(a.toarray(), labels)
    return {"SW": fm["SW"], "CH": fm["CH"], "S_Dbw": S_Dbw(x, labels),
            "AVI": am["AVI"], "AVU": am["AVU"], "ANUI": am["ANUI"], "MQ": am["modularity"]}


def rank_score(table):
    """Средний ранг варианта по всем ICVI (1 — лучший), с учётом направления каждого индекса."""
    ranks = pd.DataFrame({m: table[m].rank(ascending=not up) for m, up in HIGHER_BETTER.items()})
    return ranks.mean(axis=1)


def experiment(cfg=None, end=None):
    """Методы × правила рёбер × K на одном окне. Результат — outputs/cluster/icvi_grid.csv."""
    from .network import knn_graph, road_distances, similarity, window_months
    cfg = cfg or load_config()
    cc = cfg["cluster"]
    proc = path(cfg, "processed")
    panel = pd.read_parquet(proc / "panel.parquet")
    feats = pd.read_parquet(proc / "features.parquet")
    ctx = pd.read_parquet(proc / ("context.parquet" if (proc / "context.parquet").exists() else "context_base.parquet"))
    mo = pd.read_parquet(proc / "mo.parquet")
    end = end or feats.window_end.max()
    f = feats[feats.window_end == end]
    ids = [i for i in f.territory_id if i in mo.index]
    xdf, cols = feature_matrix(f, ctx, ids, cfg)
    x = xdf.to_numpy()
    fw = f.set_index("territory_id").loc[ids]
    dist = road_distances(cfg, ids)
    months = window_months(panel, end)
    out = path(cfg, "outputs") / "cluster"
    out.mkdir(parents=True, exist_ok=True)
    rows, labels_store = [], {}
    for rule in cc["rules"]:
        a = knn_graph(similarity(rule, fw, panel, ids, months, dist, cfg), cfg["network"]["k"])
        for method in cc["methods"]:
            if method in FEATURE_METHODS and rule != cc["rules"][0]:
                continue  # методы без сети не зависят от правила: считаем один раз, индексы сети — по первому правилу
            for k in cc["k_range"]:
                lab = run_method(method, k, x, a, seed=cfg.get("seed", 0), xi=cc["kefrin_xi"],
                                 n_init=cc["kefrin_n_init"])
                row = {"rule": rule, "method": method, "k_target": k, "k": len(set(lab)), **icvi(x, a, lab)}
                rows.append(row)
                labels_store[f"{rule}|{method}|{k}"] = lab
                print(rule, method, k, {m: round(v, 3) for m, v in row.items() if m in HIGHER_BETTER})
    t = pd.DataFrame(rows)
    t["mean_rank"] = rank_score(t)
    t.to_csv(out / "icvi_grid.csv", index=False)
    pd.DataFrame(labels_store, index=ids).to_parquet(out / "labels_grid.parquet")
    print(t.sort_values("mean_rank").head(15).round(3).to_string())
    return t


def xi_sweep(cfg=None, end=None):
    """KEFRiN при разном весе сети ξ: компромисс между качеством по признакам (SW, CH) и по сети (MQ, AVI)."""
    from .network import knn_graph, road_distances, similarity, window_months
    cfg = cfg or load_config()
    cc = cfg["cluster"]
    proc = path(cfg, "processed")
    panel = pd.read_parquet(proc / "panel.parquet")
    feats = pd.read_parquet(proc / "features.parquet")
    ctx = pd.read_parquet(proc / "context.parquet")
    mo = pd.read_parquet(proc / "mo.parquet")
    end = end or feats.window_end.max()
    f = feats[feats.window_end == end]
    ids = [i for i in f.territory_id if i in mo.index]
    x = feature_matrix(f, ctx, ids, cfg)[0].to_numpy()
    fw = f.set_index("territory_id").loc[ids]
    a = knn_graph(similarity(cc["rules"][0], fw, panel, ids, window_months(panel, end), road_distances(cfg, ids), cfg),
                  cfg["network"]["k"])
    rows = []
    for k in cc["xi_k"]:
        for xi in cc["xi_grid"]:
            lab = run_method("kefrin", k, x, a, seed=cfg.get("seed", 0), xi=xi, n_init=cc["kefrin_n_init"])
            rows.append({"k": k, "xi": xi, **icvi(x, a, lab)})
            print({m: round(float(v), 3) for m, v in rows[-1].items()})
    t = pd.DataFrame(rows)
    out = path(cfg, "outputs") / "cluster"
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "kefrin_xi.csv", index=False)
    return t


def methods_seeds(cfg=None, end=None):
    """Методы при итоговом K на правиле типологии: `seeds` запусков с разными сидами, равный бюджет —
    `n_init` перезапусков у KEFRiN и k-средних. Среднее и стандартное отклонение ICVI по сидам."""
    from .network import knn_graph, road_distances, similarity, window_months
    cfg = cfg or load_config()
    mc = cfg["cluster"]["seeds_check"]
    proc = path(cfg, "processed")
    panel = pd.read_parquet(proc / "panel.parquet")
    feats = pd.read_parquet(proc / "features.parquet")
    ctx = pd.read_parquet(proc / "context.parquet")
    mo = pd.read_parquet(proc / "mo.parquet")
    k = pd.read_parquet(path(cfg, "outputs") / "typology" / "types.parquet").type.nunique()
    end = end or feats.window_end.max()
    f = feats[(feats.window_end == end) & feats.territory_id.isin(mo.index)]
    ids = list(f.territory_id)
    x = feature_matrix(f, ctx, ids, cfg)[0].to_numpy()
    fw = f.set_index("territory_id").loc[ids]
    a = knn_graph(similarity(cfg["typology"]["rule"], fw, panel, ids, window_months(panel, end),
                             road_distances(cfg, ids), cfg), cfg["network"]["k"])
    rows = []
    for method in cfg["cluster"]["methods"]:
        for seed in range(mc["seeds"]):
            if method == "kmeans":
                from sklearn.cluster import KMeans
                lab = KMeans(k, n_init=mc["n_init"], random_state=seed).fit_predict(x)
            else:
                lab = run_method(method, k, x, a, seed=seed, xi=cfg["typology"]["xi"], n_init=mc["n_init"])
            rows.append({"method": method, "seed": seed, "k": len(set(lab)), **icvi(x, a, lab)})
        print(method, "готово", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(path(cfg, "outputs") / "cluster" / "methods_seeds.csv", index=False)
    s = t.groupby("method")[list(HIGHER_BETTER)].agg(["mean", "std"])
    s.to_csv(path(cfg, "outputs") / "cluster" / "methods_seeds_summary.csv")
    print(s.round(3).to_string())
    return t
