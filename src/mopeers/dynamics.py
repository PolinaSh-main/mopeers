"""Этап 5: как меняются типы МО по 12 окнам 2024 года.

Типология строится один раз по опорному окну (`mopeers.typology`, окно 2024-01…2024-12). Центры типов — средние
признаков МО каждого типа в опорном окне. В каждом окне признаки стандартизуются внутри окна: общий для страны
сдвиг (например, рост доли маркетплейсов у всех) вычитается, и остаётся положение МО относительно страны в этом
окне. МО относится к типу с ближайшим центром; это же правило применяется и к опорному окну, поэтому переходы
сравнивают одинаково полученные метки. Доля совпадения правила с типами KEFRiN на опорном окне сохраняется
(`agreement_reference.csv`): расходятся МО, которых в тип KEFRiN отнесла сеть. Изменение видно двумя способами:
- переходы МО между типами от окна к окну (матрица переходов);
- смещение центров типов — центры пересчитываются по МО, отнесённым к типу в каждом окне.

Флаг расхождения: МО в последнем окне ближе к центру другого типа, чем к своему опорному, либо расстояние до центра
своего типа растёт быстрее, чем у доли `drift_quantile` МО (наклон по окнам).
"""
import numpy as np
import pandas as pd

from .cluster import feature_matrix
from .config import load_config, path


def build(cfg=None):
    cfg = cfg or load_config()
    dc = cfg["dynamics"]
    proc = path(cfg, "processed")
    feats = pd.read_parquet(proc / "features.parquet")
    ctx = pd.read_parquet(proc / "context.parquet")
    mo = pd.read_parquet(proc / "mo.parquet")
    types = pd.read_parquet(path(cfg, "outputs") / "typology" / "types.parquet").type
    feats = feats[feats.territory_id.isin(mo.index)]
    ends = sorted(feats.window_end.unique())
    parts = {}
    for e in ends:  # стандартизация внутри окна
        fe = feats[feats.window_end == e]
        parts[e] = feature_matrix(fe, ctx, list(fe.territory_id), cfg)[0]
    allx = pd.concat(parts)
    ref = allx.loc[ends[-1]]
    ref = ref.loc[ref.index.intersection(types.index)]
    names = sorted(types.unique())
    centers_ref = ref.groupby(types.loc[ref.index]).mean().loc[names]

    labels = pd.DataFrame(index=sorted(feats.territory_id.unique()), columns=ends, dtype=object)
    own_dist = pd.DataFrame(index=labels.index, columns=ends, dtype=float)
    center_track = []
    for e in ends:
        x = allx.loc[e]
        d = np.linalg.norm(x.to_numpy()[:, None, :] - centers_ref.to_numpy()[None, :, :], axis=2)
        lab = pd.Series(np.array(names)[d.argmin(1)], index=x.index)
        labels.loc[x.index, e] = lab
        own = types.reindex(x.index)
        known = own.notna().to_numpy()
        col = pd.Series(np.nan, index=x.index)
        col[known] = d[known, [names.index(t) for t in own[known]]]
        own_dist.loc[x.index, e] = col
        c = x.groupby(lab).mean()
        c["window_end"] = e
        center_track.append(c.rename_axis("type").reset_index())
    out = path(cfg, "outputs") / "dynamics"
    out.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(out / "labels_by_window.parquet")
    pd.concat(center_track).to_parquet(out / "centers_by_window.parquet")

    steps = []
    for a_, b_ in zip(ends[:-1], ends[1:]):
        m = labels[[a_, b_]].dropna()
        steps.append(pd.crosstab(m[a_], m[b_]).reindex(index=names, columns=names, fill_value=0))
    step_total = sum(steps)
    step_total.to_csv(out / "transitions_monthly.csv")
    fl = labels[[ends[0], ends[-1]]].dropna()
    pd.crosstab(fl[ends[0]], fl[ends[-1]]).reindex(index=names, columns=names, fill_value=0) \
        .to_csv(out / "transitions_first_last.csv")

    t = np.arange(len(ends))
    slope = own_dist[ends].apply(
        lambda r: np.polyfit(t[r.notna().to_numpy()], r.dropna(), 1)[0] if r.notna().sum() >= 6 else np.nan, axis=1)
    last = labels[ends[-1]]
    flags = pd.DataFrame({"type": types.reindex(labels.index), "type_nearest_last": last,
                          "type_first": labels[ends[0]], "drift_slope": slope})
    flags["moved"] = flags.type_first.notna() & flags.type_nearest_last.notna() &         (flags.type_first != flags.type_nearest_last)
    agree = flags.dropna(subset=["type", "type_nearest_last"])
    pd.Series({"agreement": float((agree.type == agree.type_nearest_last).mean()), "n": len(agree)})         .to_csv(out / "agreement_reference.csv")
    flags["drifting"] = flags.drift_slope > flags.drift_slope.quantile(dc["drift_quantile"])
    flags["divergence"] = flags.moved | flags.drifting
    flags.to_parquet(out / "divergence.parquet")
    stay = float(np.trace(step_total.to_numpy()) / step_total.to_numpy().sum())
    print("совпадение правила с типами KEFRiN на опорном окне:", round(float((agree.type == agree.type_nearest_last).mean()), 3))
    print("доля МО, сохраняющих тип от окна к окну:", round(stay, 3),
          "| сменили тип за год:", int(flags.moved.sum()), "| флаг расхождения:", int(flags.divergence.sum()))
    return labels, flags
