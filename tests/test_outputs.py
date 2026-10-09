"""Проверки опубликованных результатов в outputs/: продукт, типология и контрольные суммы."""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
pytestmark = pytest.mark.skipif(not (OUT / "dataset" / "mo_peers.parquet").exists(), reason="нет outputs/")


@pytest.fixture(scope="module")
def peers():
    return pd.read_parquet(OUT / "dataset" / "mo_peers.parquet").set_index("territory_id")


def ids(s):
    return [int(x) for x in s.split(";")] if isinstance(s, str) and s else []


def test_types_match_config(peers):
    types = pd.read_parquet(OUT / "typology" / "types.parquet").type
    names = yaml.safe_load((ROOT / "configs" / "types.yaml").read_text(encoding="utf-8"))
    assert set(types.unique()) == set(names)
    assert set(types.index) == set(peers.index)
    assert (peers.type == types.reindex(peers.index)).all()


def test_peers_count_and_no_self(peers):
    cfg = yaml.safe_load((ROOT / "configs" / "base.yaml").read_text(encoding="utf-8"))["product"]
    for i, r in peers.iterrows():
        p = ids(r.peers)
        assert len(p) == cfg["n_peers"] and len(set(p)) == len(p) and i not in p
        assert set(p) <= set(peers.index)
        assert set(ids(r.experience_leaders)) <= set(p)  # у кого учиться — из числа сопоставимых
        assert i not in ids(r.local_peers)


def test_ranks_within_peer_group(peers):
    # место среди МО и его 15 сопоставимых: от 1 до 16
    r = peers.rank_spend_month
    assert r.min() >= 1 and r.max() <= 16


def test_api_card_for_every_mo(peers):
    api = OUT / "dataset" / "api"
    assert len(list(api.glob("*.json"))) == len(peers)
    card = json.loads((api / f"{peers.index[0]}.json").read_text(encoding="utf-8"))
    assert [p["territory_id"] for p in card["peers"]] == ids(peers.peers.iloc[0])


def test_checksums():
    sums = OUT / "SHA256SUMS"
    bad = []
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            bad.append(name)
    assert not bad, f"не совпали контрольные суммы: {bad[:10]}"
