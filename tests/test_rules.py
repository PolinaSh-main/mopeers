"""Проверки правил конвейера на маленьких примерах с известным ответом."""
import numpy as np
import pandas as pd
import pytest

from mopeers import rosstat
from mopeers.cluster import feature_matrix
from mopeers.network import knn_graph
from mopeers.synth import choose


def table(rows):
    """Таблица индексов по K: SW, CH, MQ — больше лучше."""
    return pd.DataFrame(rows, columns=["K", "SW", "CH", "MQ"]).set_index("K")


def test_borda_blocks_tie_broken_by_mq():
    # K = 3: ранги SW, CH, MQ = 1, 1, 3 → сумма блоков 1 + 3 = 4; K = 5: 3, 3, 1 → 3 + 1 = 4; K = 4: 2, 2, 2 → 4.
    # Все три K в ничьей; лучший ранг MQ — у K = 5
    t = table([[3, 0.30, 900, 0.40], [4, 0.20, 800, 0.45], [5, 0.10, 700, 0.50]])
    assert choose(t, "borda_blocks") == 3
    assert choose(t, "borda_blocks_large") == 5
    assert choose(t, "borda_blocks_net") == 5
    assert choose(t, "borda_blocks_feat") == 3


def test_borda_blocks_without_tie():
    # K = 4 лучший по всем трём индексам — его выбирает любое правило
    t = table([[3, 0.20, 800, 0.40], [4, 0.30, 900, 0.50], [5, 0.10, 700, 0.45]])
    for rule in ["plurality", "borda_3", "borda_blocks", "borda_blocks_net"]:
        assert choose(t, rule) == 4


def test_knn_graph_symmetric_without_self_loops():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    a = knn_graph(x @ x.T, k=5)
    assert (a != a.T).nnz == 0
    assert a.diagonal().sum() == 0
    assert ((a > 0).sum(axis=1) >= 5).all()
    assert a.data.min() > 0 and a.data.max() <= 1.001


def cluster_cfg(impute_k=5, clip_z=5):
    return {"cluster": {"window_features": ["f"], "context_features": ["c"], "log_features": [],
                        "impute_k": impute_k, "clip_z": clip_z}}


def test_missing_value_taken_from_similar_mo():
    # Две группы по 15 МО: признаки около 0 и около 10. У одного МО второй группы пропущен «c» —
    # он должен получить значение своей группы (z > 0), а не середину страны (z ≈ 0)
    rng = np.random.default_rng(0)
    ids = list(range(30))
    f = np.r_[rng.normal(0, 1, 15), rng.normal(10, 1, 15)]
    c = f + rng.normal(0, 0.5, 30)
    c[29] = np.nan
    feats = pd.DataFrame({"territory_id": ids, "f": f})
    ctx = pd.DataFrame({"c": c}, index=ids)
    z, cols = feature_matrix(feats, ctx, ids, cluster_cfg())
    assert cols == ["f", "c"]
    assert not z.isna().any().any()
    assert z.loc[29, "c"] > 0.8


def test_standardized_features_clipped():
    ids = list(range(20))
    f = np.r_[np.zeros(19), 1000.0]
    feats = pd.DataFrame({"territory_id": ids, "f": f})
    ctx = pd.DataFrame({"c": np.arange(20.0)}, index=ids)
    z, _ = feature_matrix(feats, ctx, ids, cluster_cfg(clip_z=2))
    assert z.abs().max().max() <= 2


@pytest.mark.parametrize("part, total, expected", [
    (30.0, 100.0, 0.3),        # обе величины известны — доля
    (np.nan, 100.0, 0.0),      # итог известен, части нет (район без городов) — ноль
    (np.nan, np.nan, np.nan),  # нет и итога — пропуск
])
def test_zero_if_missing(monkeypatch, part, total, expected):
    mo = pd.DataFrame({"oktmo8": ["01"]}, index=[1])
    series = {"PART": pd.Series({"01": part}), "TOTAL": pd.Series({"01": total})}
    monkeypatch.setattr(rosstat, "_load", lambda cfg, code, *a, **k: (series[code], 2024))
    cfg = {"rosstat": {"context": {"urban_share": {"code": "PART", "per": "ratio", "base_code": "TOTAL",
                                                   "zero_if_missing": True}}}}
    v = rosstat.context(cfg, mo, pd.Series({1: 1000.0})).loc[1, "urban_share"]
    assert (np.isnan(v) and np.isnan(expected)) or v == pytest.approx(expected)
