"""Kaggle-ядро: новости 2024 года по муниципалитетам и их разметка моделью GigaChat 3.1 (открытые веса, MIT).

Вход — список МО (встраивается в файл при отправке, см. mopeers.news). Для каждого МО из RSS-поиска Google News
берутся заголовки за 2024 год. Модель ai-sage/GigaChat3.1-10B-A1.8B (GGUF q8_0) на GPU присваивает каждому заголовку
категорию события и признак «событие затрагивает экономику МО». Выход — /kaggle/working/news.parquet.
"""
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

MO_LIST = json.loads(r'''__MO_LIST__''')
CATEGORIES = __CATEGORIES__
BATCH = 15

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python",
                "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu124"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "pandas", "pyarrow"], check=True)
import pandas as pd  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from llama_cpp import Llama  # noqa: E402


def fetch(name, region):
    q = f'"{name}" {region} after:2024-01-01 before:2025-01-01'
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": q, "hl": "ru", "gl": "RU", "ceid": "RU:ru"})
    for attempt in range(3):
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
            time.sleep(5 * (attempt + 1))
    return []


def stem(name):
    """Основа названия для поиска в заголовке: без последних двух букв у длинных слов (Братск → Брат, Братского)."""
    w = name.split()[0].replace("ё", "е")
    return (w[:-2] if len(w) > 6 else w).lower()


rows = []
for i, mo in enumerate(MO_LIST):
    st = stem(mo["name"])
    for n in fetch(mo["name"], mo["region"]):
        # только заголовки, где названо само МО: запрос с регионом приносит и новости региона в целом
        if st in n["title"].lower().replace("ё", "е"):
            rows.append({**n, "territory_id": mo["id"], "group": mo["group"]})
    time.sleep(1.0)
    if i % 50 == 0:
        print("новости", i, len(rows), flush=True)
news = pd.DataFrame(rows)
news["date"] = pd.to_datetime(news.date, errors="coerce", utc=True)
news = news[(news.date >= "2024-01-01") & (news.date < "2025-01-01")].drop_duplicates(["territory_id", "title"])
print("заголовков", len(news), flush=True)

path = hf_hub_download("ai-sage/GigaChat3.1-10B-A1.8B-GGUF", "GigaChat3.1-10B-A1.8B-q8_0.gguf")
llm = Llama(model_path=path, n_gpu_layers=-1, n_ctx=4096, verbose=False)
cat_list = "\n".join(f"- {c}" for c in CATEGORIES)
schema = {"type": "array", "items": {"type": "object", "properties": {
    "i": {"type": "integer"}, "category": {"type": "string", "enum": CATEGORIES}, "economy": {"type": "boolean"}},
    "required": ["i", "category", "economy"]}}
labels = []
for mo_id, g in news.groupby("territory_id"):
    mo = next(m for m in MO_LIST if m["id"] == mo_id)
    titles = g.title.tolist()
    for s in range(0, len(titles), BATCH):
        chunk = titles[s:s + BATCH]
        prompt = (f"Муниципалитет: {mo['name']} ({mo['region']}).\n"
                  f"Для каждого заголовка новости выбери одну категорию события из списка и укажи, затрагивает ли "
                  f"событие экономику этого муниципалитета (занятость, доходы, торговлю, производство, инвестиции).\n"
                  f"Категории:\n{cat_list}\n\nЗаголовки:\n" + "\n".join(f"{j}. {t}" for j, t in enumerate(chunk)))
        try:
            r = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}], temperature=0,
                                           max_tokens=60 * len(chunk),
                                           response_format={"type": "json_object", "schema": schema})
            ans = json.loads(r["choices"][0]["message"]["content"])
        except Exception as e:
            print("ошибка разметки", mo_id, e, flush=True)
            ans = []
        for a in ans:
            if isinstance(a, dict) and 0 <= a.get("i", -1) < len(chunk):
                labels.append({"territory_id": mo_id, "title": chunk[a["i"]], "category": a.get("category"),
                               "economy": bool(a.get("economy"))})
    print("размечено МО", mo_id, len(labels), flush=True)
out = news.merge(pd.DataFrame(labels), on=["territory_id", "title"], how="left")
out["date"] = out.date.astype(str)
out.to_parquet("/kaggle/working/news.parquet")
print("готово", len(out), out.category.notna().sum())
