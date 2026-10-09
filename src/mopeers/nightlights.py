"""Внешняя проверка ночными огнями со спутника.

Ночная освещённость — стандартная в экономической литературе замена данных о хозяйственной активности там, где
статистики мало. Источник — годовые композиты VIIRS Nighttime Lights V2 (Earth Observation Group, Colorado School of
Mines) в пересчёте OpenGeoHub: средняя яркость за год, пиксель 500 м, шкала 0–2000 (Hengl, 2025, Zenodo 17294744,
CC BY 4.0). Для каждого МО по его границам считаются средняя яркость пикселя и суммарный свет (средняя яркость ×
площадь МО в км²). Данные в построении типов не участвуют.

`build()` — `outputs/interpret/nightlights.csv` (свет по МО) и `nightlights_test.csv` (связь с тратами, ростом трат,
типами и флагом расхождения).
"""
import json
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd

from .config import load_config, path

FILES = {y: f"nightlights.average_viirs.v21_m_500m_s_{y}0101_{y}1231_go_epsg4326_v20250904.tif" for y in (2023, 2024)}
ZENODO = "https://zenodo.org/api/records/17294744/files/{}/content"


def fetch(cfg=None):
    cfg = cfg or load_config()
    dest = path(cfg, "raw") / "nightlights"
    dest.mkdir(parents=True, exist_ok=True)
    for f in FILES.values():
        if not (dest / f).exists():
            (dest / f).write_bytes(urllib.request.urlopen(ZENODO.format(f), timeout=600).read())


def zonal(cfg):
    from rasterstats import zonal_stats
    g = gpd.read_file(path(cfg, "mo_poly"))
    g = g[g.year_to == 9999][["territory_id", "geometry"]].astype({"territory_id": int}).set_index("territory_id")
    area = g.to_crs("ESRI:102025").area / 1e6  # Albers для Азии: равновеликая проекция, км²
    out = pd.DataFrame({"area_km2": area})
    for y, f in FILES.items():
        st = zonal_stats(list(g.geometry), str(path(cfg, "raw") / "nightlights" / f), stats=["mean"], all_touched=False)
        out[f"light_mean_{y}"] = [s["mean"] for s in st]
        out[f"light_total_{y}"] = out[f"light_mean_{y}"] * out.area_km2
    out["light_growth_2024"] = out.light_mean_2024 / out.light_mean_2023 - 1
    return out


def build(cfg=None):
    from scipy.stats import mannwhitneyu, spearmanr
    cfg = cfg or load_config()
    fetch(cfg)
    out = path(cfg, "outputs")
    nl = zonal(cfg)
    ctx = pd.read_parquet(path(cfg, "processed") / "context.parquet")
    types = pd.read_parquet(out / "typology" / "types.parquet").type
    flags = pd.read_parquet(out / "dynamics" / "divergence.parquet")
    api = out / "dataset" / "api"
    rec = {t: json.loads((api / f"{t}.json").read_text(encoding="utf-8")) for t in types.index
           if (api / f"{t}.json").exists()}
    d = nl.reindex(types.index)
    d["type"] = types
    d["light_pc"] = d.light_total_2024 / ctx.population.reindex(d.index)
    d["spend_month"] = pd.Series({t: r["spend_month"] for t, r in rec.items()})
    d["growth_2024"] = pd.Series({t: r["growth_2024"] for t, r in rec.items()})
    d["divergence"] = flags.divergence.reindex(d.index).fillna(False).astype(bool)
    d.to_csv(out / "interpret" / "nightlights.csv", lineterminator="\n")

    rows = []
    ok = d.dropna(subset=["light_pc", "spend_month"])
    ok = ok[ok.light_pc > 0]
    r = spearmanr(np.log(ok.light_pc), np.log(ok.spend_month))
    rows.append({"test": "свет на жителя ~ траты жителя, все МО", "n": len(ok), "spearman": r.statistic, "p_value": r.pvalue})
    for t, x in ok.groupby("type"):
        r = spearmanr(x.light_pc, x.spend_month)
        rows.append({"test": f"то же внутри типа {t}", "n": len(x), "spearman": r.statistic, "p_value": r.pvalue})
    ok = d.dropna(subset=["light_growth_2024", "growth_2024"])
    ok = ok[np.isfinite(ok.light_growth_2024)]
    r = spearmanr(ok.light_growth_2024, ok.growth_2024)
    rows.append({"test": "рост света 2024/2023 ~ рост трат 2024/2023", "n": len(ok), "spearman": r.statistic,
                 "p_value": r.pvalue})
    a, b = ok[ok.divergence].light_growth_2024, ok[~ok.divergence].light_growth_2024
    rows.append({"test": "рост света: МО с флагом расхождения против остальных (Манна — Уитни)", "n": len(a),
                 "median_flag": a.median(), "median_rest": b.median(), "p_value": mannwhitneyu(a, b).pvalue})
    t = pd.DataFrame(rows)
    t.to_csv(out / "interpret" / "nightlights_test.csv", index=False, lineterminator="\n")
    d.groupby("type")[["light_pc", "light_growth_2024"]].median().to_csv(out / "interpret" / "nightlights_by_type.csv", lineterminator="\n")
    print(t.round(3).to_string())
    return t
