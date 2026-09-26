"""Этап 2: признаки МО по скользящим окнам 12 месяцев.

Окно с концом в месяце t покрывает t-11..t. Концы окон — месяцы 2024 года, чтобы у каждого окна был
рост к тем же месяцам 2023 года. Три группы признаков:
- структура трат: доли категорий в тратах за окно;
- динамика трат: уровень, рост г/г последних 3 мес., сезонная амплитуда, волатильность;
- контекст: годовые показатели Росстата и СберИндекса (одинаковы для всех окон).
"""
import numpy as np
import pandas as pd

from .config import load_config, path
from .data import CATEGORIES, OTHER, TOTAL

SHARE_COLS = [f"share_{c}" for c in CATEGORIES + [OTHER]]
DYNAMIC_COLS = ["log_spend", "growth_yoy_3m", "season_amp", "volatility"]


def window_features(panel, end, min_months):
    months = sorted(panel.date.unique())
    i = months.index(end)
    win = months[i - 11:i + 1]
    w = panel[panel.date.isin(win)]
    n = w.groupby("territory_id").size()
    ok = n[n >= min_months].index
    w = w[w.territory_id.isin(ok)]
    g = w.groupby("territory_id")
    f = pd.DataFrame(index=ok)
    tot = g[TOTAL].sum()
    for c, col in zip(CATEGORIES + [OTHER], SHARE_COLS):
        f[col] = g[c].sum() / tot
    f["log_spend"] = np.log(g[TOTAL].mean())
    f["season_amp"] = (g[TOTAL].max() - g[TOTAL].min()) / g[TOTAL].mean()
    w = w.sort_values(["territory_id", "date"])
    w["dlog"] = np.log(w[TOTAL]).groupby(w.territory_id).diff()
    f["volatility"] = w.groupby("territory_id").dlog.std()
    last3 = months[i - 2:i + 1]
    prev3 = months[i - 14:i - 11]
    a = panel[panel.date.isin(last3)].groupby("territory_id")[TOTAL].mean()
    b = panel[panel.date.isin(prev3)].groupby("territory_id")[TOTAL].mean()
    f["growth_yoy_3m"] = (a / b - 1).reindex(ok)
    f["window_end"] = end
    return f


def build(cfg=None):
    cfg = cfg or load_config()
    fc = cfg["features"]
    proc = path(cfg, "processed")
    panel = pd.read_parquet(proc / "panel.parquet")
    ctx = pd.read_parquet(proc / "context.parquet") if (proc / "context.parquet").exists() \
        else pd.read_parquet(proc / "context_base.parquet")
    ends = [m for m in sorted(panel.date.unique()) if m >= fc["first_window_end"]]
    frames = [window_features(panel, e, fc["min_months"]) for e in ends]
    feats = pd.concat(frames).rename_axis("territory_id").reset_index()
    # «Невидимая экономика»: средние траты жителя в месяц к официальной зарплате (без малого бизнеса)
    spend = np.exp(feats.log_spend)
    feats["spend_to_salary"] = spend.values / feats.territory_id.map(ctx["salary"]).values
    feats.to_parquet(proc / "features.parquet")
    print("окон", len(ends), "| строк", len(feats), "| МО в последнем окне", (feats.window_end == ends[-1]).sum())
    print(feats[feats.window_end == ends[-1]].describe().T[["mean", "50%", "min", "max"]].round(3).to_string())
    return feats
