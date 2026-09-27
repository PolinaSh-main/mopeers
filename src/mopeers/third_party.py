"""Сторонний код — KEFRiN с генератором синтетических сетей (Shalileh, Mirkin) и метрики библиотеки Pattern —
скачивается с зафиксированных коммитов репозиториев авторов на GitHub.

В репозиторий он не копируется: Pattern распространяется под GPL-3.0, у KEFRiN лицензия не указана.
Файлы используются без изменений.
"""
import importlib.util
import sys
import urllib.request

from .config import load_config, path


def fetch(cfg=None):
    cfg = cfg or load_config()
    base = path(cfg, "third_party")
    for name, spec in cfg["third_party"].items():
        for rel in spec["files"]:
            target = base / name / rel.split("/")[-1]
            if target.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['commit']}/{rel}"
            target.write_bytes(urllib.request.urlopen(url, timeout=60).read())
            print("скачан", url)


def load(name, module):
    """Импорт модуля `module` из third_party/<name>; при отсутствии файлов — скачивание."""
    cfg = load_config()
    folder = path(cfg, "third_party") / name
    if not (folder / f"{module}.py").exists():
        fetch(cfg)
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))  # KEFRiN импортирует processing_tools как модуль верхнего уровня
    spec = importlib.util.spec_from_file_location(module, folder / f"{module}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
