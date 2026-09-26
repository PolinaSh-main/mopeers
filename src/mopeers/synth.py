"""Синтетические атрибутированные сети с известным числом групп: какие индексы и какое правило агрегирования
находят истинное K, и какой метод лучше восстанавливает истинное разбиение.

Генератор — `synthetic_data_generator.py` из репозитория KEFRiN (Shalileh, Mirkin), без изменений. Идея —
как в докладе Шалилеха и Антонова на AI Journey 2025: при контролируемых параметрах сети смотреть, какие ICVI
указывают на истинное K. Параметры перебора — `configs/base.yaml`, раздел `synthetic`.

Правила агрегирования индексов в выбор K:
- `plurality` — голосование SW, CH, MQ по первым местам, при равенстве — меньшее K;
- `borda_all` — сумма рангов K по всем семи индексам (с учётом направления каждого);
- `borda_3` — сумма рангов по SW, CH, MQ;
- `borda_blocks` — блоки с равным весом: признаковый (средний ранг SW и CH) и сетевой (ранг MQ).

`null_structure` — нулевая модель для реальных данных: SW и CH на признаках, где каждый столбец независимо
перемешан по МО (связи между признаками разрушены, распределения сохранены).
"""
import sys
import types

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import adjusted_rand_score

from .cluster import HIGHER_BETTER, icvi, run_method
from .config import load_config, path
from .third_party import load

RULES = ["plurality", "borda_all", "borda_3", "borda_blocks"]


def _generator():
    sys.modules.setdefault("evaluation", types.ModuleType("evaluation"))  # импортируется, но не используется
    return load("kefrin", "synthetic_data_generator")


def _ranks(t, cols):
    """Ранг каждого K по индексу (1 — лучший) с учётом направления индекса."""
    return pd.DataFrame({m: t[m].rank(ascending=not HIGHER_BETTER[m]) for m in cols}, index=t.index)


def choose(t, rule):
    """K по правилу агрегирования; t — таблица индексов с индексом K."""
    if rule == "plurality":
        votes = pd.Series([t[m].idxmax() for m in ["SW", "CH", "MQ"]]).value_counts()
        return int(votes[votes == votes.max()].index.min())
    if rule == "borda_all":
        s = _ranks(t, list(HIGHER_BETTER)).sum(axis=1)
    elif rule == "borda_3":
        s = _ranks(t, ["SW", "CH", "MQ"]).sum(axis=1)
    elif rule == "borda_blocks":
        r = _ranks(t, ["SW", "CH", "MQ"])
        s = (r.SW + r.CH) / 2 + r.MQ
    else:
        raise ValueError(rule)
    return int(s[s == s.min()].index.min())


def run(cfg=None, tag=""):
    cfg = cfg or load_config()
    sc = cfg["synthetic"]
    xi = cfg["typology"]["xi"]
    gen = _generator()
    rows, meth_rows = [], []
    for k_true in sc["k_true"]:
        for pr_v in sc["pr_v"]:
            for level, (p_in, p_out) in sc["edge_levels"].items():
                for rep in range(sc["repeats"]):
                    np.random.seed(1000 * k_true + 100 * rep + int(pr_v * 10))
                    n = sc["n_nodes"]
                    card = [n // k_true] * (k_true - 1) + [n - (n // k_true) * (k_true - 1)]
                    _, y = gen.generate_Y(n, sc["n_features"], k_true, pr_v, card, "Q", sc["noise_features"])
                    p = gen.generate_P(n, card, p_in, p_out)
                    truth = np.repeat(np.arange(k_true), card)
                    y = (y - y.mean(0)) / y.std(0)
                    a = sparse.csr_matrix(p)
                    res = []
                    for k in sc["k_range"]:
                        lab = run_method("kefrin", k, y, a, seed=rep, xi=xi, n_init=cfg["cluster"]["kefrin_n_init"])
                        res.append({"k": k, **icvi(y, a, lab)})
                    t = pd.DataFrame(res).set_index("k")
                    picks = {m: int(t[m].idxmax() if up else t[m].idxmin()) for m, up in HIGHER_BETTER.items()}
                    picks.update({r: choose(t, r) for r in RULES})
                    base = {"k_true": k_true, "pr_v": pr_v, "edges": level, "rep": rep}
                    rows.append({**base, **picks})
                    for m in sc["methods"]:  # восстановление истинного разбиения при истинном K
                        lab = run_method(m, k_true, y, a, seed=rep, xi=xi, n_init=cfg["cluster"]["kefrin_n_init"])
                        meth_rows.append({**base, "method": m, "ari": adjusted_rand_score(truth, lab)})
                    print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    d = path(cfg, "outputs") / "cluster"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / f"synthetic_k_choice{tag}.csv", index=False)
    cols = list(HIGHER_BETTER) + RULES
    hit = pd.DataFrame({c: out.groupby("edges").apply(lambda g: (g[c] == g.k_true).mean()) for c in cols}).T
    hit["all"] = [(out[c] == out.k_true).mean() for c in cols]
    hit.to_csv(d / f"synthetic_k_accuracy{tag}.csv")
    meth = pd.DataFrame(meth_rows)
    meth.to_csv(d / f"synthetic_methods_ari{tag}.csv", index=False)
    ms = meth.groupby(["method", "edges"]).ari.agg(["mean", "std"]).unstack()
    ms.to_csv(d / f"synthetic_methods_ari_summary{tag}.csv")
    print("доля наборов с истинным K:\n", hit.round(2).to_string())
    print("ARI с истинным разбиением:\n", ms.round(3).to_string())
    return out


def null_structure(cfg=None, end=None):
    """SW и CH реальных признаков против перемешанных (каждый столбец независимо), K-средних при K = 2…8."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, silhouette_score
    from .cluster import feature_matrix
    cfg = cfg or load_config()
    proc = path(cfg, "processed")
    feats = pd.read_parquet(proc / "features.parquet")
    ctx = pd.read_parquet(proc / "context.parquet")
    mo = pd.read_parquet(proc / "mo.parquet")
    end = end or feats.window_end.max()
    f = feats[(feats.window_end == end) & feats.territory_id.isin(mo.index)]
    x = feature_matrix(f, ctx, list(f.territory_id), cfg)[0].to_numpy()
    rng = np.random.default_rng(cfg.get("seed", 0))
    rows = []
    for k in range(2, 9):
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(x)
        real = (silhouette_score(x, lab), calinski_harabasz_score(x, lab))
        nulls = []
        for b in range(cfg["synthetic"]["null_perms"]):
            xp = np.column_stack([rng.permutation(col) for col in x.T])
            lp = KMeans(k, n_init=10, random_state=b).fit_predict(xp)
            nulls.append((silhouette_score(xp, lp), calinski_harabasz_score(xp, lp)))
        nulls = np.array(nulls)
        rows.append({"k": k, "SW": real[0], "SW_null_mean": nulls[:, 0].mean(), "SW_null_max": nulls[:, 0].max(),
                     "CH": real[1], "CH_null_mean": nulls[:, 1].mean(), "CH_null_max": nulls[:, 1].max()})
    t = pd.DataFrame(rows)
    t.to_csv(path(cfg, "outputs") / "cluster" / "null_structure.csv", index=False)
    print(t.round(3).to_string())
    return t


def run_weak(cfg=None):
    """Тот же эксперимент при слабых признаках и информативной сети (`synthetic.weak_features`)."""
    cfg = cfg or load_config()
    w = cfg["synthetic"]["weak_features"]
    cfg["synthetic"].update(pr_v=w["pr_v"], k_true=w["k_true"])
    return run(cfg, tag="_weak")
