"""Росстат БДМО из каталога «Точно»: каталог показателей и выборочная загрузка.

Архивы разделов весят до 3,7 ГБ, поэтому они не скачиваются целиком. `remotezip` читает
оглавление архива и отдельные файлы по HTTP Range: для каталога — только первые строки
каждого csv (там название показателя), для данных — только отобранные показатели.
"""
import time

import pandas as pd
from remotezip import RemoteZip

from .config import load_config, path


def _url(cfg, section):
    return cfg["rosstat"]["base_url"] + f"data_section{section}_{cfg['rosstat']['suffix']}.zip"


def build_catalog(cfg=None):
    """Каталог: раздел, код, название, разрез, единица, размер сжатого файла."""
    cfg = cfg or load_config()
    out = path(cfg, "rosstat_catalog")
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["section"])
    rows = done.to_dict("records")
    for section in cfg["rosstat"]["sections"]:
        if section in set(done["section"]):
            continue
        for attempt in range(5):  # хранилище изредка обрывает TLS; раздел перечитывается целиком
            try:
                rows_section = _catalog_section(cfg, section)
                break
            except Exception as e:
                print("раздел", section, "попытка", attempt + 1, type(e).__name__)
                time.sleep(5)
        else:
            continue
        rows += rows_section
        pd.DataFrame(rows).to_csv(out, index=False)
        print("раздел", section, "показателей", len(rows_section))
    return pd.DataFrame(rows)


def _catalog_section(cfg, section):
    rows = []
    z = RemoteZip(_url(cfg, section))
    for info in [i for i in z.infolist() if i.filename.endswith(".csv")]:
        for attempt in range(4):  # обрыв на одном файле: архив открывается заново, чтение продолжается с него
            try:
                with z.open(info) as f:  # заголовок и первая строка данных
                    lines = f.read(4096).decode("utf-8", "replace").split("\n")
                break
            except Exception:
                time.sleep(3)
                z = RemoteZip(_url(cfg, section))
        else:
            continue
        vals = dict(zip(lines[0].split(";"), lines[1].split(";")))
        rows.append({
            "section": section,
            "section_name": vals.get("indicator_section"),
            "code": vals.get("indicator_code"),
            "name": vals.get("indicator_name"),
            "unit": vals.get("indicator_unit"),
            "period": vals.get("indicator_period"),
            "file": info.filename,
            "compressed_mb": round(info.compress_size / 1e6, 2),
        })
    return rows


def fetch_indicators(codes, cfg=None):
    """Скачать выбранные показатели; из каждого оставить годы из конфига. Уже скачанные пропускаются."""
    cfg = cfg or load_config()
    cat = pd.read_csv(path(cfg, "rosstat_catalog"))
    dest = path(cfg, "rosstat_dir")
    dest.mkdir(parents=True, exist_ok=True)
    years = set(cfg["rosstat"]["years"])
    for code in codes:
        target = dest / f"{code}.parquet"
        if target.exists():
            continue
        files = cat[cat.code == code]
        # у крупных показателей есть части по годам (_parts/..._yearYYYY_...): берём только нужные годы
        parts = files[files.file.str.contains(r"_year(\d{4})_", regex=True)]
        parts = parts[parts.file.str.extract(r"_year(\d{4})_")[0].astype(int).isin(years)]
        chosen = parts if len(parts) else files[~files.file.str.contains("_parts/")].head(1)
        frames = []
        for attempt in range(4):
            try:
                with RemoteZip(_url(cfg, int(files.section.iloc[0]))) as z:
                    for fname in chosen.file:
                        with z.open(fname) as f:
                            frames.append(pd.read_csv(f, sep=";", dtype=str))
                break
            except Exception as e:
                frames = []
                print(code, "попытка", attempt + 1, type(e).__name__)
                time.sleep(5)
        if not frames:
            continue
        df = pd.concat(frames)
        df["year"] = pd.to_numeric(df.year, errors="coerce")
        df["indicator_value"] = pd.to_numeric(df.indicator_value.str.replace(",", "."), errors="coerce")
        df = df[df.year.isin(years)]
        df.to_parquet(target)
        print(code, len(df), "строк,", round(chosen.compressed_mb.sum(), 1), "МБ")


def _load(cfg, code, filters=None, period=None, year=None):
    """Значения показателя по МО верхнего уровня: Series oktmo → value и год.

    Год — последний, в котором покрыто ≥ 70% МО максимального года покрытия (или заданный).
    Дубли строк одного МО (разные версии ОКТМО) не суммируются: берётся максимум.
    """
    d = pd.read_parquet(path(cfg, "rosstat_dir") / f"{code}.parquet")
    d = d[d.mun_level == "Муниципальное образование верхнего уровня"]
    for col, val in (filters or {}).items():
        d = d[d[col].str.contains(val, regex=False, na=False)]
    if period:
        d = d[d.indicator_period == period]
    d = d.dropna(subset=["indicator_value"])
    if year is None:
        cov = d.groupby("year").oktmo.nunique()
        year = int(cov[cov >= 0.7 * cov.max()].index.max())
    d = d[d.year == year]
    return d.groupby("oktmo").indicator_value.max(), int(year)


def context(cfg, mo, population):
    """Показатели Росстата на МО: доля горожан, рождаемость, торговля, услуги, инвестиции, жильё, отраслевая занятость.

    На жителя — по населению из пакета хакатона (`population`). Доля с флагом `zero_if_missing` при известном итоге
    и отсутствующей части считается нулём: Росстат не публикует строку «городское население» у района без городов
    и раздел занятости без работников.
    """
    specs = cfg["rosstat"]["context"]
    key = mo.oktmo8
    out = pd.DataFrame(index=mo.index)
    years = {}
    for name, s in specs.items():
        v, years[name] = _load(cfg, s["code"], s.get("filters"), s.get("period"))
        v = key.map(v)
        if s.get("per") == "capita":
            v = v / population.reindex(mo.index) * s.get("scale", 1)
        elif s.get("per") == "ratio":
            base, _ = _load(cfg, s["base_code"], s.get("base_filters"), s.get("period"), year=years[name])
            b = key.map(base)
            v = v / b
            if s.get("zero_if_missing"):
                v = v.where(v.notna() | b.isna(), 0.0)
        out[name] = v
    out = out.replace([float("inf"), float("-inf")], float("nan"))
    print("годы показателей Росстата:", years)
    return out
