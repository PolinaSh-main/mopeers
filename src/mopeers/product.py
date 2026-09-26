"""Этап 7: сервис «Сопоставимые муниципалитеты» — датасет и статический JSON-API.

Для каждого МО по последнему окну (2024 год):
- peers      — `n_peers` экономически ближайших МО (евклидово расстояние в стандартизованных признаках),
               по всей стране и в радиусе `radius_km` по дорогам;
- potential  — место МО среди сопоставимых по ключевым показателям (1 — лучший);
- experience — сопоставимые МО с наибольшим ростом трат г/г и признаки, которыми они сильнее всего отличаются;
- divergence — флаг расхождения с типом (этап 5).
"""
import json

import numpy as np
import pandas as pd

from .cluster import feature_matrix
from .config import load_config, path
from .network import road_distances

LABELS = {
    "share_Продовольствие": "доля продовольствия", "share_Общественное питание": "доля общепита",
    "share_Здоровье": "доля здоровья", "share_Транспорт": "доля транспорта",
    "share_Маркетплейсы": "доля маркетплейсов", "share_Прочее": "доля прочих трат",
    "log_spend": "уровень трат", "growth_yoy_3m": "рост трат г/г", "season_amp": "сезонность трат",
    "volatility": "волатильность трат", "spend_to_salary": "траты к зарплате",
    "population": "население", "share_65plus": "доля 65+", "share_0_15": "доля детей",
    "pop_growth": "рост населения", "net_migration_per_1000": "миграция на 1000",
    "salary": "зарплата", "salary_growth": "рост зарплаты", "market_access": "доступность рынков",
    "urban_share": "доля горожан", "births_per_1000": "рождаемость", "deaths_per_1000": "смертность",
    "retail_pc": "розничный оборот на жителя", "catering_pc": "оборот общепита на жителя",
    "paid_services_pc": "платные услуги на жителя", "invest_pc": "инвестиции на жителя",
    "housing_m2_pc": "жильё на жителя", "agri_pc": "сельхозпродукция на жителя",
    "jobs_per_capita": "рабочих мест на жителя", "emp_mining_share": "занятость в добыче",
    "emp_manufacturing_share": "занятость в обработке", "emp_public_share": "занятость в госуправлении",
    "emp_agri_share": "занятость в сельском хозяйстве", "spend_month": "траты на жителя в месяц",
    "growth_2024": "рост трат 2024/2023",
}


def build(cfg=None):
    cfg = cfg or load_config()
    pc = cfg["product"]
    proc = path(cfg, "processed")
    feats = pd.read_parquet(proc / "features.parquet")
    ctx = pd.read_parquet(proc / ("context.parquet" if (proc / "context.parquet").exists() else "context_base.parquet"))
    mo = pd.read_parquet(proc / "mo.parquet")
    dyn = path(cfg, "outputs") / "dynamics"
    flags = pd.read_parquet(dyn / "divergence.parquet")
    end = feats.window_end.max()
    f = feats[(feats.window_end == end) & feats.territory_id.isin(mo.index)]
    ids = list(f.territory_id)
    x, cols = feature_matrix(f, ctx, ids, cfg)
    raw = pd.concat([f.set_index("territory_id").loc[ids], ctx.reindex(ids)], axis=1)
    raw["spend_month"] = np.exp(raw.log_spend)
    # рост трат за год: 2024 к 2023 по всем категориям
    panel = pd.read_parquet(proc / "panel.parquet")
    y = panel.assign(year=panel.date.str[:4]).groupby(["territory_id", "year"])["Все категории"].mean().unstack()
    raw["growth_2024"] = (y["2024"] / y["2023"] - 1).reindex(ids)

    xv = x.to_numpy()
    d = np.sqrt(((xv[:, None, :] - xv[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    road = road_distances(cfg, ids)
    n = pc["n_peers"]
    key = pc["potential_metrics"]
    records = []
    for i, tid in enumerate(ids):
        near = np.argsort(d[i])[:n]
        local = [j for j in np.argsort(d[i]) if road[i, j] <= pc["radius_km"]][:pc["n_local"]]
        grp = [i] + list(near)
        sub = raw.iloc[grp]
        potential = {}
        for m in key:
            v = sub[m].to_numpy(dtype=float)
            ok = np.isfinite(v)
            potential[m] = {"value": _num(v[0]),
                            "rank": int(sub[m].rank(ascending=False, method="min").iloc[0]) if ok[0] else None,
                            "of": int(ok.sum()),
                            "best_peer": int(ids[grp[int(np.nanargmax(v))]]) if ok.any() else None,
                            "best_value": _num(np.nanmax(v)) if ok.any() else None}
        leaders = sub.iloc[1:].sort_values("growth_2024", ascending=False).head(pc["n_experience"])
        experience = []
        for tj in leaders.index:
            j = ids.index(tj)
            diff = (x.iloc[j] - x.iloc[i])
            top = diff.abs().sort_values(ascending=False).head(3).index
            experience.append({"territory_id": int(tj), "growth_2024": _num(raw.loc[tj, "growth_2024"]),
                               "differs_by": [{"feature": LABELS.get(c, c), "direction": "выше" if diff[c] > 0 else "ниже",
                                               "z": round(float(diff[c]), 2)} for c in top]})
        records.append({
            "territory_id": int(tid), "name": mo.loc[tid, "name"], "region": mo.loc[tid, "region_name"],
            "type": flags.type.get(tid) if pd.notna(flags.type.get(tid)) else None, "divergence": bool(flags.divergence.get(tid, False)),
            "spend_month": _num(raw.loc[tid, "spend_month"]), "growth_2024": _num(raw.loc[tid, "growth_2024"]),
            "peers": [{"territory_id": int(ids[j]), "distance": round(float(d[i, j]), 3),
                       "road_km": _num(road[i, j])} for j in near],
            "local_peers": [int(ids[j]) for j in local],
            "potential": potential, "experience": experience,
        })
    out = path(cfg, "outputs") / "dataset"
    api = out / "api"
    api.mkdir(parents=True, exist_ok=True)
    for r in records:
        (api / f"{r['territory_id']}.json").write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
    flat = pd.DataFrame([{
        "territory_id": r["territory_id"], "name": r["name"], "region": r["region"], "type": r["type"],
        "divergence": r["divergence"], "peers": ";".join(str(p["territory_id"]) for p in r["peers"]),
        "local_peers": ";".join(map(str, r["local_peers"])),
        "experience_leaders": ";".join(str(e["territory_id"]) for e in r["experience"]),
        **{f"rank_{m}": r["potential"][m]["rank"] for m in key}} for r in records])
    flat.to_csv(out / "mo_peers.csv", index=False, encoding="utf-8")
    flat.to_parquet(out / "mo_peers.parquet")
    print("МО в датасете", len(flat), "| с флагом расхождения", int(flat.divergence.sum()))
    return flat


def _num(v):
    v = float(v)
    return None if not np.isfinite(v) else round(v, 4)


def _int(v):
    return None if v is None or pd.isna(v) else int(v)


def peers(name_or_id, cfg=None):
    """Сопоставимые МО для одного МО: по territory_id или по названию."""
    cfg = cfg or load_config()
    flat = pd.read_parquet(path(cfg, "outputs") / "dataset" / "mo_peers.parquet")
    row = flat[(flat.territory_id == name_or_id) | (flat.name == name_or_id)]
    if row.empty:
        raise KeyError(name_or_id)
    tid = int(row.territory_id.iloc[0])
    rec = json.loads((path(cfg, "outputs") / "dataset" / "api" / f"{tid}.json").read_text(encoding="utf-8"))
    names = flat.set_index("territory_id").name
    return pd.DataFrame([{"territory_id": p["territory_id"], "name": names.get(p["territory_id"]),
                          "distance": p["distance"], "road_km": p["road_km"]} for p in rec["peers"]])
