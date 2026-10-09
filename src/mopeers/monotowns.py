"""Внешняя проверка перечнем моногородов.

Перечень монопрофильных муниципальных образований утверждён распоряжением Правительства РФ от 29.07.2014 № 1398-р.
Фонд развития моногородов делил их на три группы: с наиболее сложным социально-экономическим положением, с риском
его ухудшения и со стабильной ситуацией. Таблица с группами и градообразующими предприятиями берётся из статьи
«Список моногородов России» русской Википедии (группа закодирована цветом строки) с фиксированной версией статьи.

`fetch()` — таблица в `data/raw/monotowns.csv`; `build()` — сопоставление с МО и сравнение групп по типам,
флагу расхождения и росту трат (`outputs/interpret/monotowns*.csv`).
"""
import json
import re
import time
import urllib.parse
import urllib.request

import pandas as pd

from .config import load_config, path

GROUPS = {"#FFF0F5": "сложное положение", "#FFF8DC": "риск ухудшения", "#F0FFF0": "стабильное"}


def _clean(x):
    x = re.sub(r"<ref.*?(</ref>|/>)", "", x)
    x = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", x)
    return re.sub(r"'''?|\{\{.*?\}\}", "", x).strip()


def fetch(cfg=None):
    cfg = cfg or load_config()
    mc = cfg["monotowns"]
    q = urllib.parse.urlencode({"action": "parse", "prop": "wikitext", "format": "json", "oldid": mc["wiki_revision"]})
    req = urllib.request.Request("https://ru.wikipedia.org/w/api.php?" + q, headers={"User-Agent": "mopeers/0.1"})
    w = json.loads(urllib.request.urlopen(req, timeout=60).read())["parse"]["wikitext"]["*"]
    table = w[w.index("{|"):w.index("\n|}")]
    rows, region = [], None
    for block in table.split("\n|-")[1:]:
        if "colspan" in block:
            m = re.search(r"'''(.+?)'''", block)
            region = m.group(1).strip() if m else region
            continue
        color = re.search(r"background:(#\w+)", block)
        cells = []
        for line in block.split("\n"):
            if line.startswith("|") and not line.startswith("|-"):
                c = re.match(r'^\|\s*style="[^"]*"\s*\|(.*)$', line)
                cells.append(_clean(c.group(1) if c else line[1:]))
        if len(cells) >= 3:
            rows.append({"region": region, "town": cells[0], "population_k": cells[1], "enterprise": cells[2],
                         "group": GROUPS.get(color.group(1).upper()) if color else None})
    df = pd.DataFrame(rows)
    out = path(cfg, "raw") / "monotowns.csv"
    df.to_csv(out, index=False, lineterminator="\n")
    print("моногородов:", len(df), "| по группам:", df.group.value_counts(dropna=False).to_dict())
    return df


def match(cfg):
    """Моногород → МО того же субъекта: сначала по совпадению названия (городской округ), затем по основе названия
    (город — центр района: Онега → Онежский); при нескольких кандидатах по основе МО не назначается."""
    mono = pd.read_csv(path(cfg, "raw") / "monotowns.csv")
    mo = pd.read_parquet(path(cfg, "processed") / "mo.parquet")
    norm = lambda s: str(s).lower().replace("ё", "е").strip()  # noqa: E731
    mo["rk"], mo["nk"] = mo.region_name.map(norm), mo["name"].map(norm)
    mono["town_name"] = mono.town.str.replace(r"\s*\(.*\)$", "", regex=True)
    ids, how = [], []
    for _, r in mono.iterrows():
        reg, town = mo[mo.rk == norm(r.region)], norm(r.town_name)
        exact = reg.index[reg.nk == town]
        st = town[:max(4, len(town) - 2)]
        by_stem = reg.index[reg.nk.str.startswith(st)]
        if len(exact) == 1:
            ids.append(exact[0]); how.append("название")
        elif len(by_stem) == 1:
            ids.append(by_stem[0]); how.append("основа названия")
        else:
            ids.append(None); how.append(None)
    mono["territory_id"], mono["match"] = ids, how
    return mono


def _get(url, tries=4):
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def stock_returns(cfg=None):
    """Изменение цены акций компаний градообразующих предприятий за 2024 год (Мосбиржа, режим TQBR, цена закрытия
    первого и последнего торгового дня) → `data/raw/moex_2024.csv`."""
    cfg = cfg or load_config()
    rows = []
    for ticker in sorted(set(cfg["monotowns"]["tickers"].values())):
        url = (f"https://iss.moex.com/iss/history/engines/stock/markets/shares/boards/TQBR/securities/{ticker}.json"
               "?from=2024-01-01&till=2024-12-31&iss.meta=off&history.columns=TRADEDATE,CLOSE&limit=100&start={}")
        data, start = [], 0
        while True:
            page = json.loads(_get(url.format(start)))["history"]["data"]
            data += [d for d in page if d[1] is not None]
            if len(page) < 100:
                break
            start += 100
        # дробление акций (у Норникеля в 2024 году 1:100) — скачок цены за день больше чем в 5 раз;
        # множитель дробления переносится на цену начала года
        first = data[0][1]
        for (_, prev), (_, cur) in zip(data, data[1:]):
            if cur / prev < 0.2 or cur / prev > 5:
                first *= round(cur / prev, 2)
        rows.append({"ticker": ticker, "close_first": first, "close_last": data[-1][1],
                     "return_2024": data[-1][1] / first - 1, "days": len(data)})
    df = pd.DataFrame(rows)
    df.to_csv(path(cfg, "raw") / "moex_2024.csv", index=False, lineterminator="\n")
    return df


def build(cfg=None):
    """Моногорода по типам; группы Фонда против роста трат, места среди сопоставимых и флага расхождения."""
    from scipy.stats import chi2_contingency, kruskal
    cfg = cfg or load_config()
    raw, out = path(cfg, "raw"), path(cfg, "outputs")
    if not (raw / "monotowns.csv").exists():
        fetch(cfg)
    if not (raw / "moex_2024.csv").exists():
        stock_returns(cfg)
    m = match(cfg).dropna(subset=["territory_id"])
    m["territory_id"] = m.territory_id.astype(int)
    types = pd.read_parquet(out / "typology" / "types.parquet").type
    peers = pd.read_parquet(out / "dataset" / "mo_peers.parquet").set_index("territory_id")
    growth = pd.Series({t: json.loads((out / "dataset" / "api" / f"{t}.json").read_text(encoding="utf-8"))["growth_2024"]
                        for t in m.territory_id if (out / "dataset" / "api" / f"{t}.json").exists()})
    m = m[m.territory_id.isin(types.index)].copy()
    m["type"] = m.territory_id.map(types)
    m["divergence"] = m.territory_id.map(peers.divergence)
    m["growth_2024"] = m.territory_id.map(growth)
    m["rank_growth_2024"] = m.territory_id.map(peers.rank_growth_2024)
    m["rank_spend_month"] = m.territory_id.map(peers.rank_spend_month)
    tick = cfg["monotowns"]["tickers"]
    m["ticker"] = [next((t for pat, t in tick.items() if pat in e), None) for e in m.enterprise.fillna("")]
    ret = pd.read_csv(raw / "moex_2024.csv").set_index("ticker").return_2024
    m["stock_return_2024"] = m.ticker.map(ret)
    dest = out / "interpret"
    m.to_csv(dest / "monotowns.csv", index=False, lineterminator="\n")
    pd.crosstab(m.group, m.type, margins=True).to_csv(dest / "monotowns_by_type.csv", lineterminator="\n")
    g = m.dropna(subset=["group"])
    tests = []
    for col in ["growth_2024", "rank_growth_2024", "rank_spend_month"]:
        parts = [x[col].dropna() for _, x in g.groupby("group")]
        tests.append({"metric": col, "test": "Краскел — Уоллис", "p_value": kruskal(*parts).pvalue,
                      **{f"median: {k}": x[col].median() for k, x in g.groupby("group")}})
    ct = pd.crosstab(g.group, g.divergence.astype(bool))
    tests.append({"metric": "divergence", "test": "хи-квадрат", "p_value": chi2_contingency(ct)[1],
                  **{f"median: {k}": x.divergence.astype(bool).mean() for k, x in g.groupby("group")}})
    t = pd.DataFrame(tests)
    t.to_csv(dest / "monotowns_test.csv", index=False, lineterminator="\n")
    print("моногородов сопоставлено с МО:", len(m), "| с акциями:", m.ticker.notna().sum())
    print(t.round(3).to_string())
    return t
