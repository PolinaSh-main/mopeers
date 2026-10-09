"""Малый и средний бизнес по МО — внешние данные о занятости и предпринимательстве.

Источник — «Реестр субъектов малого и среднего предпринимательства за 2016–2024 гг. с доходами» (ФНС России, обработка
и объединение — Центр научной информации РЦНИ, data.rcsi.science, набор 205, CC BY-SA 4.0; для скачивания нужна
бесплатная регистрация). Строка — субъект МСП за год: вид (1 — юрлицо, 2 — индивидуальный предприниматель), ОКТМО и
координаты населённого пункта, выручка, среднесписочная численность работников (у юрлиц).

Субъект относится к МО по координатам населённого пункта (точка в границах МО): коды ОКТМО у части МО сменились при
преобразовании районов в округа. Итог по МО за 2022 и 2023 годы — `data/processed/sme.parquet` (агрегат публикуется
в репозитории, исходный файл 16 ГБ — нет). `build()` — связь с тратами и с отношением трат к зарплатам.
"""
import geopandas as gpd
import numpy as np
import pandas as pd

from .config import load_config, path


def aggregate(cfg=None):
    import duckdb
    cfg = cfg or load_config()
    src = path(cfg, "raw") / "SME" / "panel.csv"
    q = f"""SELECT round(lat, 4) AS lat, round(lon, 4) AS lon, year,
      count(*) FILTER (WHERE kind = 1) AS orgs, count(*) FILTER (WHERE kind = 2) AS ie,
      sum(employees_count) FILTER (WHERE kind = 1) AS emp
    FROM read_csv('{src.as_posix()}', delim=';', header=true, quote='"', escape='"', sample_size=200000,
      types={{'oktmo': 'VARCHAR', 'tin': 'VARCHAR', 'reg_number': 'VARCHAR'}})
    WHERE year IN (2022, 2023) AND lat IS NOT NULL GROUP BY 1, 2, 3"""
    pts = duckdb.connect().sql(q).df()
    g = gpd.read_file(path(cfg, "mo_poly"))
    g = g[g.year_to == 9999][["territory_id", "geometry"]].astype({"territory_id": int})
    p = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.lon, pts.lat), crs="EPSG:4326").to_crs(g.crs)
    j = gpd.sjoin(p, g, predicate="within", how="left")
    print("субъектов МСП вне границ МО:", round(j[j.territory_id.isna()].orgs.sum() / j.orgs.sum(), 4))
    agg = j.dropna(subset=["territory_id"]).groupby(["territory_id", "year"])[["orgs", "ie", "emp"]].sum()
    agg = agg.unstack("year")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.index = agg.index.astype(int)
    agg.to_parquet(path(cfg, "processed") / "sme.parquet")
    return agg


def build(cfg=None):
    """Связь занятости в МСП с тратами жителя и с отношением трат к фонду оплаты труда без МСП (раздел «Невидимая
    экономика»): если траты сверх местных зарплат объясняются малым бизнесом, связь должна быть положительной."""
    from scipy.stats import spearmanr
    cfg = cfg or load_config()
    proc, out = path(cfg, "processed"), path(cfg, "outputs")
    sme = pd.read_parquet(proc / "sme.parquet") if (proc / "sme.parquet").exists() else aggregate(cfg)
    ctx = pd.read_parquet(proc / "context.parquet")
    types = pd.read_parquet(out / "typology" / "types.parquet").type
    f = pd.read_parquet(proc / "features.parquet")
    f = f[f.window_end == f.window_end.max()].set_index("territory_id")
    d = pd.DataFrame(index=types.index)
    d["type"] = types
    pop = ctx.population.reindex(d.index)
    d["sme_emp_pc"] = sme.emp_2023.reindex(d.index) / pop
    d["ie_per_1000"] = sme.ie_2023.reindex(d.index) / pop * 1000
    d["sme_emp_growth"] = sme.emp_2023.reindex(d.index) / sme.emp_2022.reindex(d.index) - 1
    d["spend"] = np.exp(f.log_spend.reindex(d.index))
    d["spend_to_payroll"] = d.spend / ctx.payroll_pc_month.reindex(d.index)
    d.to_csv(out / "interpret" / "sme.csv", lineterminator="\n")
    rows = []
    for x, y in [("sme_emp_pc", "spend_to_payroll"), ("ie_per_1000", "spend_to_payroll"),
                 ("sme_emp_pc", "spend"), ("ie_per_1000", "spend")]:
        ok = d[[x, y, "type"]].replace([np.inf, -np.inf], np.nan).dropna()
        r = spearmanr(ok[x], ok[y])
        within = np.nanmedian([spearmanr(g[x], g[y]).statistic for _, g in ok.groupby("type") if len(g) > 20])
        rows.append({"x": x, "y": y, "n": len(ok), "spearman": r.statistic, "p_value": r.pvalue,
                     "spearman_within_type_median": within})
    t = pd.DataFrame(rows)
    t.to_csv(out / "interpret" / "sme_test.csv", index=False, lineterminator="\n")
    d.groupby("type")[["sme_emp_pc", "ie_per_1000", "sme_emp_growth", "spend_to_payroll"]].median().to_csv(
        out / "interpret" / "sme_by_type.csv", lineterminator="\n")
    print(t.round(3).to_string())
    return t
