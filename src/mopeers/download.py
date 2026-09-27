"""Скачивание исходных данных из официальных источников с проверкой SHA-256.

- пакет конкурса СберИндекса 2026 (`hackathonlicence.zip`: траты, доступность рынков, расстояния, лицензия);
- пакет хакатона СберИндекса Data Sense 2025 (Яндекс.Диск: население, миграция, зарплата);
- справочник и границы МО СберИндекса (`t_dict_municipal.rar`);
- индекс мобильности (API СберИндекса).

Сайты sberbank.com и sberindex.ru используют сертификат Минцифры, которого нет в стандартных хранилищах,
поэтому для них проверка сертификата отключена; целостность пакета Data Sense проверяется по SHA-256 из API диска.

Снимок всех исходных файлов на 27.09.2026 (включая выгрузки показателей Росстата) опубликован в релизе репозитория
(`download.snapshot_url`, SHA-256 — `download.snapshot_sha256`). Он используется, если официальный источник
недоступен, или по требованию: `MOPEERS_SNAPSHOT=1 uv run mopeers download`.
"""
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import urllib.parse
import urllib.request
import zipfile

from .config import load_config, path

CONTEST_ZIP = "https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip"
BORDERS = "https://www.sberbank.com/common/files/t_dict_municipal.rar"
MOBILITY = "https://sberindex.ru/api/dataset/v1/download/indeks-mobilnosti/parquet"
DISK = "https://disk.yandex.ru/d/WH8yJOogD4UrOg"
DISK_FILES = ["2_bdmo_population.parquet", "3_bdmo_migration.parquet", "4_bdmo_salary.parquet",
              "Описание_данных_для_Хакатона_2025_6_6.docx"]
_NOVERIFY = ssl._create_unverified_context()


def _get(url, verify=True):
    return urllib.request.urlopen(url, timeout=300, context=None if verify else _NOVERIFY).read()


def _unpack_contest(contest):
    zipfile.ZipFile(contest / "hackathonlicence.zip").extractall(contest)


def _unpack_borders(raw):
    borders = raw / "borders"
    borders.mkdir(exist_ok=True)
    # RAR читает tar на libarchive: bsdtar в Linux и macOS, системный tar.exe в Windows 10+
    win_tar = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "tar.exe")
    tar = shutil.which("bsdtar") or (win_tar if os.path.exists(win_tar) else shutil.which("tar"))
    subprocess.run([tar, "-xf", str(raw / "t_dict_municipal.rar"), "-C", str(borders)], check=True)


def snapshot(cfg=None):
    """Все исходные файлы из снимка в релизе репозитория, с проверкой SHA-256."""
    cfg = cfg or load_config()
    raw = path(cfg, "raw")
    raw.mkdir(parents=True, exist_ok=True)
    data = _get(cfg["download"]["snapshot_url"])
    if hashlib.sha256(data).hexdigest() != cfg["download"]["snapshot_sha256"]:
        raise ValueError("SHA-256 снимка данных не совпал")
    (raw / "snapshot.zip").write_bytes(data)
    zipfile.ZipFile(raw / "snapshot.zip").extractall(raw)
    _unpack_contest(raw / "contest2026")
    print("снимок данных:", len(zipfile.ZipFile(raw / "snapshot.zip").namelist()), "файлов")


def build(cfg=None):
    cfg = cfg or load_config()
    if os.environ.get("MOPEERS_SNAPSHOT") == "1":
        return snapshot(cfg)
    try:
        _official(cfg)
    except Exception as e:  # noqa: BLE001 — любой сбой источника: берём снимок
        print("официальный источник недоступен:", e, "— используется снимок данных")
        snapshot(cfg)


def _official(cfg):
    raw = path(cfg, "raw")
    raw.mkdir(parents=True, exist_ok=True)
    contest = raw / "contest2026"
    if not (contest / "hackathonlicence" / "consumption.parquet").exists():
        contest.mkdir(exist_ok=True)
        (contest / "hackathonlicence.zip").write_bytes(_get(CONTEST_ZIP, verify=False))
        _unpack_contest(contest)
        print("пакет конкурса 2026")
    listing = json.loads(_get("https://cloud-api.yandex.net/v1/disk/public/resources?" +
                              urllib.parse.urlencode({"public_key": DISK, "limit": 50})))
    for item in listing["_embedded"]["items"]:
        if item["name"] not in DISK_FILES or (raw / item["name"]).exists():
            continue
        data = _get(item["file"])
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"SHA-256 не совпал: {item['name']}")
        (raw / item["name"]).write_bytes(data)
        print("Data Sense:", item["name"])
    if not (raw / "borders" / "t_dict_municipal_districts.xlsx").exists():
        (raw / "t_dict_municipal.rar").write_bytes(_get(BORDERS, verify=False))
        _unpack_borders(raw)
        print("справочник и границы МО")
    mob = path(cfg, "mobility")
    if not mob.exists():
        mob.parent.mkdir(parents=True, exist_ok=True)
        mob.write_bytes(_get(MOBILITY, verify=False))
        print("индекс мобильности")
