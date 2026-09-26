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


def fca_descriptions(bin_df, labels, n_attrs):
    """Понятие решётки, лучше всего описывающее тип.

    Для типа берутся `n_attrs` атрибутов с наибольшим подъёмом (доля в типе / доля в стране). На контексте
    «все МО × эти атрибуты» строится решётка формальных понятий (fcapy); выбирается понятие с непустым
    содержанием, чей объём лучше всего совпадает с типом по F1.
    """
    from fcapy.lattice import ConceptLattice
    out = []
    base = bin_df.mean()
    for t in sorted(labels.unique()):
        inside = labels == t
        lift = (bin_df[inside.to_numpy()].mean() / base.replace(0, np.nan)).dropna().sort_values(ascending=False)
        attrs = list(lift.index[:n_attrs])
        sub = bin_df[attrs]
        ctx = FormalContext(data=sub.to_numpy().tolist(), object_names=[str(i) for i in sub.index],
                            attribute_names=attrs)
        lattice = ConceptLattice.from_context(ctx)
        best = None
        for c in lattice:
            if not c.intent:
                continue
            ext = pd.Index([int(i) for i in c.extent])
            tp = inside.reindex(ext).sum()
            prec, rec = tp / max(len(ext), 1), tp / inside.sum()
            f1 = 2 * prec * rec / (prec + rec) if tp else 0
            if best is None or f1 > best["f1"]:
                best = {"type": t, "size": int(inside.sum()), "intent": list(c.intent), "extent_size": len(ext),
                        "precision": float(prec), "recall": float(rec), "f1": float(f1)}
        best["n_concepts"] = len(lattice)
        out.append(best)
    return pd.DataFrame(out)


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
    fca = fca_descriptions(binarize(raw[cols]), labels, ic["fca_attrs"])
    tree = DecisionTreeClassifier(max_depth=ic["tree_depth"], min_samples_leaf=ic["tree_min_leaf"],
                                  random_state=0).fit(raw[cols].fillna(raw[cols].median()), labels)
    rules = export_text(tree, feature_names=[LABELS.get(c, c) for c in cols], decimals=3)
    tree_acc = float(tree.score(raw[cols].fillna(raw[cols].median()), labels))
    cats = lab_categories(mo.loc[ids], ctx)
    cross = pd.DataFrame({c: cats[c].groupby(labels).mean() for c in cats}).T

    out = path(cfg, "outputs") / "interpret"
    out.mkdir(parents=True, exist_ok=True)
    prof_z.to_csv(out / "profile_z.csv")
    prof_raw.to_csv(out / "profile_median.csv")
    fca.to_json(out / "fca_descriptions.json", orient="records", force_ascii=False, indent=1)
    (out / "tree_rules.txt").write_text(f"точность дерева на обучении: {tree_acc:.3f}\n\n{rules}", encoding="utf-8")
    cross.to_csv(out / "lab_categories_by_type.csv")
    sizes = labels.value_counts().sort_index()
    for t in sizes.index:
        top = prof_z[t].sort_values()
        print(f"тип {t} ({sizes[t]} МО): выше страны — {', '.join(LABELS.get(c, c) for c in top.index[-4:][::-1])}; "
              f"ниже — {', '.join(LABELS.get(c, c) for c in top.index[:3])}")
    print(fca[["type", "size", "extent_size", "precision", "recall"]].round(2).to_string())
    print("дерево глубины", ic["tree_depth"], "— точность", round(tree_acc, 3))
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
    out = path(cfg, "outputs") / "interpret"
    res.to_csv(out / "mobility_by_type.csv")
    pd.Series({"n": n, "H": h, "p_value": p, "eta2": eta2}).to_csv(out / "mobility_test.csv")
    print(res.round(2).to_string()); print(f"n = {n}, H = {h:.1f}, p = {p:.2e}, η² = {eta2:.2f}")
    return res
