"""Проверка правила выбора K на синтетических атрибутированных сетях с известным числом кластеров.

Генератор — `synthetic_data_generator.py` из репозитория KEFRiN (Shalileh, Mirkin), без изменений.
Идея — как в докладе Шалилеха и Антонова на AI Journey 2025: при контролируемых параметрах сети
смотреть, какие ICVI указывают на истинное K. Для каждого набора (K_истинное, pr_v, повтор) KEFRiN
запускается при K из k_range, и для каждого индекса фиксируется K с лучшим значением.
"""
import sys
import types

import numpy as np
import pandas as pd
from scipy import sparse

from .cluster import HIGHER_BETTER, icvi, rank_score, run_method
from .config import load_config, path
from .third_party import load


def _generator():
    sys.modules.setdefault("evaluation", types.ModuleType("evaluation"))  # импортируется, но не используется
    return load("kefrin", "synthetic_data_generator")


def run(cfg=None):
    cfg = cfg or load_config()
    sc, k_range = cfg["synthetic"], cfg["cluster"]["k_range"]
    gen = _generator()
    rows = []
    for k_true in sc["k_true"]:
        for pr_v in sc["pr_v"]:
            for rep in range(sc["repeats"]):
                np.random.seed(1000 * k_true + rep)
                n = sc["n_nodes"]
                card = [n // k_true] * (k_true - 1) + [n - (n // k_true) * (k_true - 1)]
                _, y = gen.generate_Y(n, sc["n_features"], k_true, pr_v, card, "Q", sc["noise_features"])
                p = gen.generate_P(n, card, sc["p_within"], sc["p_between"])
                y = (y - y.mean(0)) / y.std(0)
                a = sparse.csr_matrix(p)
                res = []
                for k in k_range:
                    lab = run_method("kefrin", k, y, a, seed=rep)
                    res.append({"k": k, **icvi(y, a, lab)})
                t = pd.DataFrame(res)
                t["mean_rank"] = rank_score(t)
                picks = {m: int(t.loc[t[m].idxmax() if up else t[m].idxmin(), "k"]) for m, up in HIGHER_BETTER.items()}
                picks["mean_rank"] = int(t.loc[t.mean_rank.idxmin(), "k"])
                rows.append({"k_true": k_true, "pr_v": pr_v, "rep": rep, **picks})
                print(rows[-1])
    out = pd.DataFrame(rows)
    d = path(cfg, "outputs") / "cluster"
    d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "synthetic_k_choice.csv", index=False)
    hit = {m: float((out[m] == out.k_true).mean()) for m in list(HIGHER_BETTER) + ["mean_rank"]}
    pd.Series(hit, name="share_true_k").to_csv(d / "synthetic_k_accuracy.csv")
    print("доля наборов, где индекс указал истинное K:", hit)
    return out
