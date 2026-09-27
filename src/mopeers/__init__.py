"""mopeers — сопоставимые муниципалитеты по данным СберИндекса и Росстата.

Запуск: `uv run mopeers <шаг>`; `uv run mopeers all` выполняет все шаги по порядку.
"""
import importlib
import sys

STEPS = [
    ("download", "download", "build", "исходные данные СберИндекса и Росстата с проверкой SHA-256"),
    ("data", "data", "build", "справочник МО, панель трат, контекст (скачивает показатели Росстата)"),
    ("features", "features", "build", "признаки по 12 окнам 2024 г."),
    ("network", "network", "compare", "сравнение 6 правил ребра"),
    ("null", "synth", "null_structure", "нулевая модель: есть ли кластерная структура"),
    ("synth", "synth", "run", "выбор K и методы на синтетических сетях с известным ответом"),
    ("synth_weak", "synth", "run_weak", "то же при слабых признаках"),
    ("grid", "cluster", "experiment", "6 методов × правила × K, ICVI"),
    ("xi", "cluster", "xi_sweep", "вес сети ξ в KEFRiN"),
    ("seeds", "cluster", "methods_seeds", "методы при итоговом K по 10 сидам"),
    ("typology", "typology", "build", "типология KEFRiN и выбор числа типов"),
    ("stability", "typology", "stability", "устойчивость типологии на подвыборках"),
    ("dynamics", "dynamics", "build", "типы по окнам, переходы, флаг расхождения"),
    ("interpret", "interpret", "build", "профили, интервальные описания типов, дерево, сверка с категориями"),
    ("mobility", "interpret", "external_mobility", "внешняя проверка индексом мобильности"),
    ("product", "product", "build", "сопоставимые МО, потенциал, опыт; JSON-API"),
    ("thesis", "product", "thesis_check", "проверка тезиса: сопоставимые против своего региона"),
    ("geo", "geo", "build", "границы МО для карты (нужен Node.js для npx mapshaper)"),
    ("site", "site", "build", "лендинг site/index.html"),
]


def _run(module, func):
    getattr(importlib.import_module(f"mopeers.{module}"), func)()


def main():
    for stream in (sys.stdout, sys.stderr):  # консоль Windows по умолчанию не в UTF-8
        stream.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    names = [s[0] for s in STEPS]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        for name, _, _, desc in STEPS:
            print(f"  {name:<10} {desc}")
        return
    todo = names if args[0] == "all" else args
    for name in todo:
        step = next(s for s in STEPS if s[0] == name)
        print(f"== {name}: {step[3]}")
        _run(step[1], step[2])
