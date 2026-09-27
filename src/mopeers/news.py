"""Этап 8: новостной слой. Разметка идёт на Kaggle (GPU), локально — подготовка входа, отправка, анализ.

`push()`   — собирает список МО (все с флагом расхождения + контрольная группа того же размера из устойчивых МО,
             случайная, с тем же распределением по типам), встраивает его в kaggle/news_kernel.py и отправляет ядро.
`pull()`   — забирает результат ядра в outputs/news/news.parquet.
`build()`  — сравнивает группы: доля МО с экономическими событиями каждой категории, точный тест Фишера.
Kaggle CLI вызывается через `uvx kaggle` (в систему не ставится); токен — ~/.kaggle/access_token.
"""
import json
import subprocess

import numpy as np
import pandas as pd

from .config import ROOT, load_config, path


def _kaggle(*args):
    return subprocess.run(["uvx", "kaggle", *args], capture_output=True, text=True, encoding="utf-8")


def mo_list(cfg):
    nc = cfg["news"]
    mo = pd.read_parquet(path(cfg, "processed") / "mo.parquet")
    flags = pd.read_parquet(path(cfg, "outputs") / "dynamics" / "divergence.parquet").dropna(subset=["type"])
    div = flags[flags.divergence]
    rng = np.random.default_rng(cfg.get("seed", 0))
    ctrl = []
    for t, n in div.type.value_counts().items():
        pool = flags[(~flags.divergence) & (flags.type == t)].index.to_numpy()
        ctrl += list(rng.choice(pool, size=min(n, len(pool)), replace=False))
    rows = [{"id": int(i), "group": "расхождение"} for i in div.index] + [{"id": int(i), "group": "контроль"} for i in ctrl]
    for r in rows:
        r["name"] = mo.loc[r["id"], "name"]
        r["region"] = mo.loc[r["id"], "region_name"]
    return rows[: nc.get("max_mo", len(rows))]


def push(cfg=None):
    cfg = cfg or load_config()
    nc = cfg["news"]
    rows = mo_list(cfg)
    src = (ROOT / "kaggle" / "news_kernel.py").read_text(encoding="utf-8")
    gold = (ROOT / "kaggle" / "gold_labels.json").read_text(encoding="utf-8")
    src = src.replace("__MO_LIST__", json.dumps(rows, ensure_ascii=False)).replace(
        "__CATEGORIES__", json.dumps(nc["categories"], ensure_ascii=False)).replace(
        "__GOLD__", json.dumps(json.loads(gold), ensure_ascii=False))
    build_dir = ROOT / "kaggle" / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "news_kernel.py").write_text(src, encoding="utf-8")
    user = nc["kaggle_user"]
    meta = {"id": f"{user}/{nc['kernel_slug']}", "title": nc["kernel_slug"], "code_file": "news_kernel.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "machine_shape": "NvidiaTeslaT4",
            "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    (build_dir / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    r = _kaggle("kernels", "push", "-p", str(build_dir))
    print("МО в запросе:", len(rows), "|", r.stdout.strip(), r.stderr.strip()[-300:])


def status(cfg=None):
    cfg = cfg or load_config()
    r = _kaggle("kernels", "status", f"{cfg['news']['kaggle_user']}/{cfg['news']['kernel_slug']}")
    return (r.stdout + r.stderr).strip()


def pull(cfg=None):
    cfg = cfg or load_config()
    out = path(cfg, "outputs") / "news"
    out.mkdir(parents=True, exist_ok=True)
    r = _kaggle("kernels", "output", f"{cfg['news']['kaggle_user']}/{cfg['news']['kernel_slug']}", "-p", str(out))
    print((r.stdout or "").strip()[-500:], (r.stderr or "").strip()[-300:])


def build(cfg=None):
    """Выбор модели разметки по ручной выборке и сравнение групп «расхождение» и «контроль».

    Ручная выборка — 120 заголовков, размеченных вручную (`kaggle/gold_labels.json`). Для каждой модели — согласие
    с ручной разметкой по признаку «экономика» (каппа Коэна). Дальше используется модель с большей каппой: доля МО
    с экономическими событиями каждой категории в двух группах и точный тест Фишера. У МО с большим числом заголовков
    экономическое событие находится чаще, поэтому различие групп проверяется и с поправкой на число заголовков:
    логистическая регрессия «есть экономическое событие» на группу и логарифм числа заголовков.
    """
    import numpy as np
    import statsmodels.formula.api as smf
    from scipy.stats import fisher_exact
    from sklearn.metrics import cohen_kappa_score
    cfg = cfg or load_config()
    out = path(cfg, "outputs") / "news"
    news = pd.read_parquet(out / "news.parquet")
    gold = pd.read_parquet(out / "gold_pred.parquet")
    models = [c.removesuffix("_economy") for c in gold.columns if c.endswith("_economy") and c != "economy"]
    agree = []
    for m in models:
        ok = gold[f"{m}_economy"].notna()
        g = gold[ok]
        agree.append({"model": m, "kappa": cohen_kappa_score(g.economy, g[f"{m}_economy"].astype(bool)),
                      "accuracy": (g.economy == g[f"{m}_economy"].astype(bool)).mean(),
                      "share_true_manual": g.economy.mean(), "share_true_model": g[f"{m}_economy"].astype(bool).mean(),
                      "n": int(ok.sum())})
    agree = pd.DataFrame(agree).sort_values("kappa", ascending=False)
    agree.to_csv(out / "labeler_agreement.csv", index=False)
    best = agree.model.iloc[0]
    groups = pd.DataFrame(mo_list(cfg)).set_index("id").group
    econ = news[news[f"{best}_economy"].fillna(False).astype(bool)]
    rows = []
    for cat in cfg["news"]["categories"] + ["любое экономическое событие"]:
        sel = econ if cat == "любое экономическое событие" else econ[econ[f"{best}_category"] == cat]
        has = groups.index.isin(sel.territory_id.unique())
        a = int((has & (groups == "расхождение")).sum())
        b = int(((~has) & (groups == "расхождение")).sum())
        c = int((has & (groups == "контроль")).sum())
        d = int(((~has) & (groups == "контроль")).sum())
        odds, p = fisher_exact([[a, b], [c, d]])
        rows.append({"category": cat, "model": best, "share_divergent": a / max(a + b, 1),
                     "share_control": c / max(c + d, 1), "odds_ratio": odds, "p_value": p})
    t = pd.DataFrame(rows)
    t.to_csv(out / "divergence_vs_control.csv", index=False)
    mo = pd.DataFrame({"group": groups, "titles": news.groupby("territory_id").size().reindex(groups.index).fillna(0)})
    mo["has"] = groups.index.isin(econ.territory_id.unique()).astype(int)
    mo["divergent"] = (mo.group == "расхождение").astype(int)
    mo["log_titles"] = np.log1p(mo.titles)
    fit = smf.logit("has ~ divergent + log_titles", mo).fit(disp=0)
    vol = pd.DataFrame({"mean_titles": mo.groupby("group").titles.mean(), "median_titles": mo.groupby("group").titles.median(),
                        "mo_with_titles": mo[mo.titles > 0].groupby("group").size()})
    vol["logit_coef_divergent"], vol["logit_p_divergent"] = fit.params["divergent"], fit.pvalues["divergent"]
    vol.to_csv(out / "divergence_adjusted.csv")
    print(vol.round(3).to_string())
    print(agree.round(3).to_string())
    print("заголовков:", len(news), "| МО:", news.territory_id.nunique(), "| экономических:", len(econ))
    print(t.round(3).to_string())
    return t
