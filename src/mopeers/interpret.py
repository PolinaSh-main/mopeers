"""Этап 6: интерпретация типов МО.

1. Профиль типа — медианы признаков в исходных единицах и отклонения от страны в стандартных отклонениях.
2. Описание через анализ формальных понятий (fcapy). Признаки бинаризуются по терцилям страны
   («доля маркетплейсов: верхняя треть»). Для типа строится решётка понятий на его самых характерных атрибутах
   и выбирается понятие, чей объём (все МО с этими атрибутами) лучше всего совпадает с типом.
   Точность описания — доля объёма, попавшая в тип; полнота — доля типа, попавшая в объём.
3. Пороговые правила — дерево решений глубины `tree_depth`, обученное различать типы по исходным признакам.
4. Сверка с категориями МО, которые использует лаборатория СберИндекса (пост от 29.12.2025): центры регионов,
   сельские территории, добывающие территории, ЗАТО.
"""
import numpy as np
import pandas as pd
from fcapy.context import FormalContext
from sklearn.tree import DecisionTreeClassifier, export_text

from .cluster import feature_matrix
from .config import load_config, path
from .product import LABELS


def binarize(raw):
    """Терцили по стране → атрибуты «нижняя треть» / «верхняя треть» (средняя треть атрибута не даёт).

    Терциль — по процентильному рангу со средним рангом для равных значений: у признака, где у большинства МО 0
    (занятость в добыче), нули получают один ранг и попадают в одну треть, а не в обе.
    """
    cols = {}
    for c in raw.columns:
        r = raw[c].rank(pct=True, method="average")
        name = LABELS.get(c, c)
        cols[f"{name}: нижняя треть"] = (r <= 1 / 3).fillna(False)
        cols[f"{name}: верхняя треть"] = (r > 2 / 3).fillna(False)
    return pd.DataFrame(cols)


def interval_patterns(raw, z, labels, ic, seed=0):
    """Описание типов интервальными узорными структурами (interval pattern structures; Kaytoue, Kuznetsov et al., 2011).

    Описание множества МО A — вектор интервалов [min, max] значений признаков по A (замыкание в интервальной
    узорной структуре); объём описания — все МО страны, чьи значения попадают во все интервалы. Для типа:
    признаки — `pattern_features` самых отличающихся от страны (по медиане стандартизованных значений), без
    остаточной категории «прочее» и признаков с покрытием ниже `min_coverage`; A — ядро типа, доля
    `core_share` его МО, ближайших к центру типа.

    Качество: точность (доля объёма в типе) и полнота (доля типа в объёме) на отложенной половине МО — узор
    строится по ядру обучающей половины; устойчивость объёма — средний индекс Жаккара объёмов на `n_boot`
    подвыборках; пересечения объёмов разных типов.
    """
    rng = np.random.default_rng(seed)
    cover = raw.notna().mean()
    usable = [c for c in raw.columns if c not in ic["exclude_features"] and cover[c] >= ic["min_coverage"]]

    def pattern(members, cols):
        zz = z.loc[members, cols]
        core = members[np.argsort(np.linalg.norm(zz - zz.mean(), axis=1))[:max(5, int(len(members) * ic["core_share"]))]]
        box = raw.loc[core, cols]
        return box.min(), box.max()

    def extent(lo, hi, index):
        r = raw.loc[index, lo.index]
        return index[((r >= lo) & (r <= hi)).all(axis=1).to_numpy()]

    ids = raw.index
    half = rng.permutation(len(ids))
    train, test = ids[half[: len(ids) // 2]], ids[half[len(ids) // 2:]]
    rows, extents = [], {}
    for t in sorted(labels.unique()):
        members = labels.index[labels == t]
        diff = z.loc[members, usable].median().abs().sort_values(ascending=False)
        cols = list(diff.index[: ic["pattern_features"]])
        lo, hi = pattern(members, cols)
        ext = extent(lo, hi, ids)
        extents[t] = set(ext)
        # отложенная выборка
        lo_t, hi_t = pattern(members.intersection(train), cols)
        ext_t = extent(lo_t, hi_t, test)
        in_t = labels.reindex(ext_t).eq(t)
        tp = int(in_t.sum())
        n_type_test = int(labels.reindex(test).eq(t).sum())
        # устойчивость объёма
        jac = []
        for _ in range(ic["n_boot"]):
            sub = members[rng.random(len(members)) < 0.8]
            lb, hb = pattern(sub, cols)
            eb = set(extent(lb, hb, ids))
            jac.append(len(eb & extents[t]) / max(len(eb | extents[t]), 1))
        rows.append({"type": t, "size": len(members), "features": cols,
                     "intervals": {c: [float(lo[c]), float(hi[c])] for c in cols},
                     "extent_size": len(ext), "precision_test": tp / max(len(ext_t), 1),
                     "recall_test": tp / max(n_type_test, 1), "stability": float(np.mean(jac))})
    types = sorted(extents)
    overlap = pd.DataFrame([[len(extents[a] & extents[b]) for b in types] for a in types], index=types, columns=types)
    return pd.DataFrame(rows), overlap


def lab_categories(mo, ctx):
    c = pd.DataFrame(index=mo.index)
    c["центр региона"] = mo.status.eq("административный_центр_субъекта")
    c["ЗАТО"] = mo.status.eq("ЗАТО")
    c["сельская территория"] = ctx.urban_share.reindex(mo.index) < 0.3
    c["добывающая территория"] = ctx.emp_mining_share.reindex(mo.index) >= 0.15
    c["внутригородская (Москва, СПб, Севастополь)"] = mo.mo_type.str.startswith("внутригородская")
    return c


def build(cfg=None, labels=None):
    cfg = cfg or load_config()
    ic = cfg["interpret"]
    proc = path(cfg, "processed")
    feats = pd.read_parquet(proc / "features.parquet")
    ctx = pd.read_parquet(proc / "context.parquet")
    mo = pd.read_parquet(proc / "mo.parquet")
    end = feats.window_end.max()
    if labels is None:
        labels = pd.read_parquet(path(cfg, "outputs") / "typology" / "types.parquet").type
    f = feats[(feats.window_end == end) & feats.territory_id.isin(labels.index)]
    ids = list(f.territory_id)
    labels = labels.loc[ids]
    z, cols = feature_matrix(f, ctx, ids, cfg)
    raw = pd.concat([f.set_index("territory_id").loc[ids], ctx.reindex(ids)], axis=1)[cols]
    raw["spend_month"] = np.exp(raw.log_spend)

    prof_z = z.groupby(labels).mean().T
    prof_raw = raw.groupby(labels).median().T
    prof_raw["Россия"] = raw.median()
    raw_desc = raw[cols].copy()
    raw_desc["spend_month"] = raw["spend_month"]
    raw_desc = raw_desc.drop(columns=["log_spend"])
    fca, overlap = interval_patterns(raw_desc, z.assign(spend_month=z.log_spend).drop(columns=["log_spend"]),
                                     labels, ic)
    # пороговые правила — в исходных единицах (траты в рублях), без остаточной категории
    from sklearn.model_selection import cross_val_score
    tree_x = raw_desc.drop(columns=[c for c in ic["exclude_features"] if c in raw_desc.columns])
    tree_x = tree_x.fillna(tree_x.median())
    make_tree = lambda: DecisionTreeClassifier(max_depth=ic["tree_depth"], min_samples_leaf=ic["tree_min_leaf"],
                                               random_state=0)
    tree = make_tree().fit(tree_x, labels)
    rules = export_text(tree, feature_names=[LABELS.get(c, c) for c in tree_x.columns], decimals=3)
    tree_acc = float(tree.score(tree_x, labels))
    tree_cv = float(cross_val_score(make_tree(), tree_x, labels, cv=5).mean())
    cats = lab_categories(mo.loc[ids], ctx)
    cross = pd.DataFrame({c: cats[c].groupby(labels).mean() for c in cats}).T

    out = path(cfg, "outputs") / "interpret"
    out.mkdir(parents=True, exist_ok=True)
    prof_z.to_csv(out / "profile_z.csv")
    prof_raw.to_csv(out / "profile_median.csv")
    fca.to_json(out / "interval_patterns.json", orient="records", force_ascii=False, indent=1)
    overlap.to_csv(out / "interval_patterns_overlap.csv")
    (out / "tree_rules.txt").write_text(f"точность дерева на обучении: {tree_acc:.3f}\n"
                                        f"точность при 5-кратной перекрёстной проверке: {tree_cv:.3f}\n\n{rules}",
                                        encoding="utf-8")
    cross.to_csv(out / "lab_categories_by_type.csv")
    sizes = labels.value_counts().sort_index()
    for t in sizes.index:
        top = prof_z[t].sort_values()
        print(f"тип {t} ({sizes[t]} МО): выше страны — {', '.join(LABELS.get(c, c) for c in top.index[-4:][::-1])}; "
              f"ниже — {', '.join(LABELS.get(c, c) for c in top.index[:3])}")
    print(fca[["type", "size", "features", "extent_size", "precision_test", "recall_test", "stability"]].round(2).to_string())
    print("пересечения объёмов:"); print(overlap.to_string())
    print("дерево глубины", ic["tree_depth"], "— точность", round(tree_acc, 3), "| 5-кратная проверка", round(tree_cv, 3))
    print(cross.round(2).to_string())
    return prof_z, fca, cross


NW_REGIONS = ["Республика Карелия", "Республика Коми", "Архангельская", "Ненецкий", "Вологодская", "Калининградская",
              "Ленинградская", "Мурманская", "Новгородская", "Псковская", "Санкт-Петербург"]


def external_mobility(cfg=None):
    """Внешняя проверка: индекс мобильности СберИндекса (в кластеризации не участвовал) по типам.

    Индекс опубликован для 297 МО Северо-Западного ФО (2024 г.); сопоставление — по полному названию МО
    внутри регионов СЗФО (внутри округа названия уникальны). Тест Краскела — Уоллиса и доля дисперсии,
    объяснённая типом (η², по рангам).
    """
    from scipy.stats import kruskal
    cfg = cfg or load_config()
    mo = pd.read_parquet(path(cfg, "processed") / "mo.parquet")
    types = pd.read_parquet(path(cfg, "outputs") / "typology" / "types.parquet").type
    m = pd.read_parquet(path(cfg, "mobility"))
    m = m[m.period == m.period.min()]
    nw = mo[mo.region_name.str.contains("|".join(NW_REGIONS))].reset_index()
    full = nw.drop_duplicates("full_name", keep=False).set_index("full_name").territory_id
    m = m.assign(territory_id=m.ref_area.str.replace(r"\s+", " ", regex=True).map(full)).dropna(subset=["territory_id"])
    m["type"] = m.territory_id.astype(int).map(types)
    m = m.dropna(subset=["type"])
    groups = [g.value.to_numpy() for _, g in m.groupby("type") if len(g) >= 5]
    h, p = kruskal(*groups)
    n, k = len(m), len(groups)
    eta2 = (h - k + 1) / (n - k)
    res = m.groupby("type").value.agg(["count", "median"]).rename(columns={"median": "mobility_km"})
    # устойчивость вывода: без районов Санкт-Петербурга и против базовой модели «только доля горожан»
    from scipy.stats import spearmanr
    ctx = pd.read_parquet(path(cfg, "processed") / "context.parquet")
    inner = mo.mo_type.str.startswith("внутригородская")
    rest = m[~m.territory_id.astype(int).map(inner).fillna(False).astype(bool)]  # без районов Санкт-Петербурга
    g2 = [g.value.to_numpy() for _, g in rest.groupby("type") if len(g) >= 5]
    h2, p2 = kruskal(*g2)
    eta2_rest = (h2 - len(g2) + 1) / (len(rest) - len(g2))
    rank = m.value.rank()
    type_means = rank.groupby(m.type).transform("mean")
    urb = ctx.urban_share.reindex(m.territory_id.astype(int)).to_numpy()
    ok = ~np.isnan(urb)
    rho, _ = spearmanr(urb[ok], m.value.to_numpy()[ok])
    extra = {"eta2_without_spb": eta2_rest, "p_without_spb": p2, "n_without_spb": len(rest),
             "urban_share_spearman_r2": rho ** 2, "n_urban_known": int(ok.sum())}
    out = path(cfg, "outputs") / "interpret"
    res.to_csv(out / "mobility_by_type.csv")
    pd.Series({"n": n, "H": h, "p_value": p, "eta2": eta2, **extra}).to_csv(out / "mobility_test.csv")
    print(res.round(2).to_string()); print(f"n = {n}, H = {h:.1f}, p = {p:.2e}, η² = {eta2:.2f}", extra)
    return res
