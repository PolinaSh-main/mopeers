"""Kaggle-ядро: новости 2024 года по муниципалитетам и их разметка открытыми языковыми моделями.

Вход — список МО и ручная разметка 120 заголовков (встраиваются в файл при отправке, см. mopeers.news).
Для каждого МО из RSS-поиска Google News берутся заголовки за каждый месяц 2024 года (поиск отдаёт не больше
100 заголовков на запрос). Остаются заголовки, где названо само МО, а не одноимённый регион.
Две открытые модели — Qwen3.5-35B-A3B (Apache 2.0, 4 бита) и GigaChat 3.1 10B-A1.8B (MIT, 8 бит) — присваивают
каждому заголовку категорию события и признак «событие затрагивает экономику МО». Модели работают в Ollama
(версия зафиксирована; модель делится между двумя GPU T4), ответ ограничен JSON-схемой. Те же модели размечают
ручную выборку, по которой выбирается модель для вывода.
Выход — /kaggle/working/news.parquet и /kaggle/working/gold_pred.parquet.
"""
import calendar
import json
import os
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

MO_LIST = json.loads(r'''__MO_LIST__''')
GOLD = json.loads(r'''__GOLD__''')
CATEGORIES = __CATEGORIES__
MODELS = {"qwen35": "hf.co/unsloth/Qwen3.5-35B-A3B-GGUF:Q4_K_M",
          "gigachat": "hf.co/ai-sage/GigaChat3.1-10B-A1.8B-GGUF:Q8_0"}
OLLAMA_VERSION = "0.34.4"
BATCH = 15
PARALLEL = 4
REGION_WORDS = r"\w*\s+(област|кра[йяею]|республик|автономн)"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas", "pyarrow"], check=True)
# Ollama ставится и запускается в фоне, пока собираются новости; модели — в /tmp (в /kaggle/working только 20 ГБ)
env = {**os.environ, "OLLAMA_MODELS": "/tmp/ollama", "OLLAMA_NUM_PARALLEL": str(PARALLEL),
       "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_VERSION": OLLAMA_VERSION}
subprocess.run("(apt-get -qq update && apt-get -qq install -y zstd) > /dev/null && "
               f"curl -fsSL https://github.com/ollama/ollama/releases/download/v{OLLAMA_VERSION}/ollama-linux-amd64.tar.zst"
               " | zstd -d | tar x -C /usr/local && ollama --version", shell=True, env=env, check=True)
server = subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=open("/tmp/ollama.log", "w"))
for _ in range(120):  # сервер поднимается несколько секунд; скачивание моделей — только после этого
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=5)
        break
    except OSError:
        time.sleep(2)
else:
    raise RuntimeError(open("/tmp/ollama.log").read()[-3000:])
pulls = subprocess.Popen(" && ".join(f"ollama pull {m}" for m in MODELS.values()), shell=True, env=env,
                         stdout=open("/tmp/pull.log", "w"), stderr=subprocess.STDOUT)
import pandas as pd  # noqa: E402


def fetch(name, region, month):
    last = calendar.monthrange(2024, month)[1]
    q = f'"{name}" {region} after:2024-{month:02d}-01 before:2024-{month:02d}-{last:02d}'
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": q, "hl": "ru", "gl": "RU", "ceid": "RU:ru"})
    for attempt in range(4):
        try:
            root = ET.fromstring(urllib.request.urlopen(url, timeout=30).read())
            out = []
            for it in root.iter("item"):
                title = it.findtext("title") or ""
                src = it.findtext("source") or ""
                title = re.sub(r"\s+-\s+" + re.escape(src) + r"$", "", title) if src else title
                out.append({"title": title, "date": it.findtext("pubDate"), "source": src, "link": it.findtext("link")})
            return out
        except Exception:
            time.sleep(10 * (attempt + 1))
    return []


def stem(name):
    """Основа названия для поиска в заголовке: без последних двух букв у длинных слов (Братск → Брат, Братского)."""
    w = name.split()[0].replace("ё", "е")
    return (w[:-2] if len(w) > 6 else w).lower()


def about_mo(title, st):
    """Названо само МО: основа есть в заголовке, и хотя бы одно её вхождение — не название региона."""
    t = title.lower().replace("ё", "е")
    hits = [m.start() for m in re.finditer(re.escape(st), t)]
    return any(not re.match(REGION_WORDS, t[h + len(st):]) for h in hits)


rows = []
for i, mo in enumerate(MO_LIST):
    st = stem(mo["name"])
    for month in range(1, 13):
        for n in fetch(mo["name"], mo["region"], month):
            if about_mo(n["title"], st):
                rows.append({**n, "territory_id": mo["id"], "group": mo["group"]})
        time.sleep(1.5)
    if i % 25 == 0:
        print("новости", i, len(rows), flush=True)
news = pd.DataFrame(rows)
news["date"] = pd.to_datetime(news.date, errors="coerce", utc=True)
news = news[(news.date >= "2024-01-01") & (news.date < "2025-01-01")].drop_duplicates(["territory_id", "title"])
news = news.reset_index(drop=True)
print("заголовков", len(news), "МО", news.territory_id.nunique(), flush=True)

cat_list = "\n".join(f"- {c}" for c in CATEGORIES)
schema = {"type": "object", "properties": {"labels": {"type": "array", "items": {"type": "object", "properties": {
    "i": {"type": "integer"}, "category": {"type": "string", "enum": CATEGORIES}, "economy": {"type": "boolean"}},
    "required": ["i", "category", "economy"]}}}, "required": ["labels"]}
RULES = ("economy = true, только если событие меняет занятость, доходы жителей, торговлю, производство или "
         "инвестиции в этом муниципалитете: открытие, расширение, закрытие или продажа предприятия, сокращения, "
         "зарплаты и выплаты жителям, крупная стройка или инвестиционный проект, тарифы, банкротства. "
         "economy = false для ремонта дорог и зданий, благоустройства, спорта, культуры, происшествий, "
         "криминала, назначений и выборов, погоды, здравоохранения и образования, если в них нет перечисленного.")


def label(model, name, region, titles):
    prompt = (f"Муниципалитет: {name} ({region}).\nДля каждого заголовка новости выбери одну категорию события "
              f"из списка и укажи признак economy.\n{RULES}\nКатегории:\n{cat_list}\n\nЗаголовки:\n" +
              "\n".join(f"{j}. {t}" for j, t in enumerate(titles)))
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "format": schema, "stream": False,
            "think": False, "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 100 * len(titles) + 50}}
    try:
        req = urllib.request.Request("http://127.0.0.1:11434/api/chat", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req, timeout=900).read())
        ans = json.loads(r["message"]["content"])
        ans = ans.get("labels", []) if isinstance(ans, dict) else ans
    except Exception as e:
        print("ошибка разметки", name, repr(e)[:200], flush=True)
        return {}
    return {a["i"]: (a.get("category"), bool(a.get("economy"))) for a in ans
            if isinstance(a, dict) and isinstance(a.get("i"), int) and 0 <= a["i"] < len(titles)}


if pulls.wait() != 0:
    raise RuntimeError(open("/tmp/pull.log").read()[-3000:])
gold = pd.DataFrame(GOLD)
mo_by_id = {m["id"]: m for m in MO_LIST}
jobs = [("gold", g.index, name, region, g.title.tolist()) for (name, region), g in gold.groupby(["name", "region"])]
for mo_id, g in news.groupby("territory_id"):
    for st in range(0, len(g), BATCH):
        jobs.append(("news", g.index[st:st + BATCH], mo_by_id[mo_id]["name"], mo_by_id[mo_id]["region"],
                     g.title.iloc[st:st + BATCH].tolist()))
for key, model in MODELS.items():
    t0 = time.time()
    with ThreadPoolExecutor(PARALLEL) as pool:
        results = list(pool.map(lambda j: label(model, j[2], j[3], j[4]), jobs))
    pred, cats, econ = {}, {}, {}
    for (kind, idx, _, _, _), res in zip(jobs, results):
        for j, i in enumerate(idx):
            c, e = res.get(j, (None, None))
            if kind == "gold":
                pred[i] = e
            else:
                cats[i], econ[i] = c, e
    gold[f"{key}_economy"] = pd.Series(pred)
    news[f"{key}_category"] = pd.Series(cats)
    news[f"{key}_economy"] = pd.Series(econ)
    print(key, "размечено", news[f"{key}_category"].notna().sum(), "из", len(news),
          "за", round((time.time() - t0) / 60), "мин", flush=True)
    subprocess.run(["ollama", "stop", model], env=env)
print("\n".join(x for x in open("/tmp/ollama.log").read().splitlines() if "CUDA" in x or "offload" in x)[-2000:])
news["date"] = news.date.astype(str)
news.to_parquet("/kaggle/working/news.parquet")
gold.to_parquet("/kaggle/working/gold_pred.parquet")
server.terminate()
print("готово")
