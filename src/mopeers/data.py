"""Этап 1: панель трат МО × месяц × категория и контекст МО (Росстат, СберИндекс)."""
import numpy as np
import pandas as pd

from .config import load_config, path

CATEGORIES = ["Продовольствие", "Общественное питание", "Здоровье", "Транспорт", "Маркетплейсы"]
OTHER = "Прочее"
TOTAL = "Все категории"


def municipalities(cfg):
    """Справочник действующих МО: id, название, тип, статус, регион, координаты центра."""
    d = pd.read_excel(path(cfg, "mo_dict"))
    d = d[d.year_to == 9999].copy()
    d["oktmo8"] = d.oktmo.str.replace("-", "").str[:8]
    keep = ["territory_id", "municipal_district_name_short", "municipal_district_name", "municipal_district_type",
            "municipal_district_status", "region_code", "region_name", "oktmo8",
            "municipal_district_center_lat", "municipal_district_center_lon"]
    d = d[keep].rename(columns={
        "municipal_district_name_short": "name", "municipal_district_name": "full_name",
        "municipal_district_type": "mo_type", "municipal_district_status": "status",
        "municipal_district_center_lat": "lat", "municipal_district_center_lon": "lon"})
    return d.drop_duplicates("territory_id").set_index("territory_id")


def consumption_panel(cfg):
    """Длинная таблица: territory_id, date, category, value (руб. на жителя в месяц); «Прочее» = итог − 5 категорий."""
    c = pd.read_parquet(path(cfg, "consumption"))
    w = c.pivot_table(index=["territory_id", "date"], columns="category", values="value", aggfunc="first")
    w[OTHER] = (w[TOTAL] - w[CATEGORIES].sum(axis=1)).clip(lower=0)
    w = w.dropna(subset=[TOTAL])
    return w.reset_index()


def context(cfg, mo):
    """Годовые показатели МО из пакета хакатона: население, возрастная структура, миграция, зарплата, доступность рынков."""
    ctx = pd.DataFrame(index=mo.index)

    pop = pd.read_parquet(path(cfg, "population"))
    # в исходнике есть строка «Всего» и пересекающиеся группы «65+», «70+», «80+»; одиночные годы у части МО
    # доходят только до 69. Итог — «Всего», доля 65+ — группа «65+», доля 0–15 — одиночные годы
    single = pop[pop.age.str.fullmatch(r"\d+") | (pop.age == "80+")].copy()
    single["age_n"] = pd.to_numeric(single.age.str.extract(r"(\d+)")[0])
    tot = pop[pop.age == "Всего"].groupby(["territory_id", "year"]).value.sum(min_count=1).unstack()
    last, first = tot.columns.max(), tot.columns.min()
    total = tot[last].replace(0, np.nan)
    p23 = tot[first].replace(0, np.nan)
    s24 = single[single.year == last]
    ctx["population"] = total
    g65 = pop[(pop.age == "65+") & (pop.year == last)].groupby("territory_id").value.sum(min_count=1)
    ctx["share_65plus"] = g65.reindex(ctx.index).fillna(s24[s24.age_n >= 65].groupby("territory_id").value.sum()) / total
    ctx["share_0_15"] = s24[s24.age_n <= 15].groupby("territory_id").value.sum() / total
    ctx["share_women"] = pop[(pop.age == "Всего") & (pop.year == last) & (pop.gender == "Женщины")]         .groupby("territory_id").value.sum() / total
    ctx["pop_growth"] = total / p23 - 1

    mig = pd.read_parquet(path(cfg, "migration"))
    ctx["net_migration_per_1000"] = mig.groupby("territory_id").value.sum(min_count=1) / p23 * 1000

    sal = pd.read_parquet(path(cfg, "salary"))
    # в исходнике у части МО строка «Все отрасли» продублирована пустой копией — берём непустое значение
    sal = sal[(sal.period == "январь-декабрь") & (sal.okved_letter == "0")]
    sal = sal.groupby(["territory_id", "year"]).value.max().unstack()
    ctx["salary"] = sal[sal.columns.max()]
    first = sal[sal.columns.min()]
    ctx["salary_growth"] = ctx["salary"] / first - 1

    ctx = ctx.replace([np.inf, -np.inf], np.nan)
    shares = [c for c in ctx.columns if c.startswith("share_")]
    ctx[shares] = ctx[shares].where((ctx[shares] >= 0) & (ctx[shares] <= 1))  # ошибки исходника: доля > 1
    ma = pd.read_parquet(path(cfg, "market_access")).set_index("territory_id").market_access
    ctx["market_access"] = ma
    return ctx


def build(cfg=None):
    cfg = cfg or load_config()
    out = path(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)
    mo = municipalities(cfg)
    panel = consumption_panel(cfg)
    ctx = context(cfg, mo)
    mo.to_parquet(out / "mo.parquet")
    panel.to_parquet(out / "panel.parquet")
    ctx.to_parquet(out / "context_base.parquet")
    from .rosstat import context as rosstat_context
    ctx = ctx.join(rosstat_context(cfg, mo, ctx["population"]))
    ctx.to_parquet(out / "context.parquet")
    print("МО в справочнике", len(mo), "| МО в панели трат", panel.territory_id.nunique(),
          "| месяцев", panel.date.nunique())
    print("покрытие контекста (доля МО панели):")
    ids = panel.territory_id.unique()
    print(ctx.loc[ctx.index.intersection(ids)].notna().mean().round(3).to_string())
    return mo, panel, ctx
