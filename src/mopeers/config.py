from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(name="base"):
    with open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(cfg, key):
    return ROOT / cfg["paths"][key]
