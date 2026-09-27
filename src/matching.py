"""Matching model (LightGBM, MIT licence) + decision rules tuned for macro F0.5.

Training: GroupKFold by Source-1 entity, so all candidates of one entity are in the same fold.
The out-of-fold (OOF) probabilities are used to choose the decision rule; test probabilities are
the average of the fold models.

Decision rules compared on OOF macro F0.5 (the exact leaderboard metric, singletons included,
matches lost in blocking counted as misses):
  * exclusivity on/off: a Source-2/3 record may be assigned only to the S1 entity for which it has
    the highest probability (Source 1 is deduplicated). Checked against training labels.
  * "thr": a global probability threshold;  "thr_src": separate thresholds for S2 and S3;
  * "expf": per S1 entity, choose the top-k set (k = 0..n) that maximises the expected F0.5 given
    the probabilities (k = 0 has value P(no match) = prod(1 - p)), optionally with a floor.
"""
import time

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from data_io import macro_f05_from_arrays


# --------------------------------------------------------------------------- training
def train_cv(X, y, groups, cfg, log=print):
    params = dict(cfg.lgb_params, seed=cfg.seed, num_threads=cfg.n_jobs)
    n_folds = int(min(cfg.n_folds, len(np.unique(groups))))
    oof = np.zeros(len(y), dtype=np.float64)
    models = []
    for f, (tr, va) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        t0 = time.time()
        dtr = lgb.Dataset(X.iloc[tr], label=y[tr], free_raw_data=True)
        dva = lgb.Dataset(X.iloc[va], label=y[va], reference=dtr, free_raw_data=True)
        m = lgb.train(params, dtr, num_boost_round=cfg.num_boost_round, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        auc = roc_auc_score(y[va], oof[va]) if 0 < y[va].sum() < len(va) else float("nan")
        log(f"  fold {f + 1}/{n_folds}: best_iter={m.best_iteration} auc={auc:.5f} ({time.time() - t0:.0f}s)")
        models.append(m)
    return models, oof


def predict(models, X):
    return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)


def feature_importance(models, names, top=25):
    imp = np.mean([m.feature_importance("gain") for m in models], axis=0)
    order = np.argsort(-imp)[:top]
    return [(names[k], float(imp[k])) for k in order]


# --------------------------------------------------------------------------- decision rules
def exclusive_mask(cand, p):
    """True for the highest-probability S1 entity of each candidate record."""
    order = np.lexsort((-p, cand))
    c_sorted = cand[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = c_sorted[1:] != c_sorted[:-1]
    m = np.zeros(len(p), dtype=bool)
    m[order[first]] = True
    return m


def expf_mask(s1_code, p, floor):
    """Per S1 entity, select the top-k candidates maximising expected F0.5."""
    n = len(p)
    sel = np.zeros(n, dtype=bool)
    if n == 0:
        return sel
    order = np.lexsort((-p, s1_code))
    g = s1_code[order]
    ps = p[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    grp = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, n]))
    cs = np.cumsum(ps)
    base = np.r_[0.0, cs][starts]
    tp = cs - base[grp]                                   # expected TP when taking top-k
    total = np.add.reduceat(ps, starts)[grp]
    k = np.arange(n) - starts[grp] + 1
    e = 1.25 * tp / (1.25 * tp + 0.25 * (total - tp) + (k - tp) + 1e-12)
    e[ps < floor] = -1.0
    e0 = np.exp(np.add.reduceat(np.log(np.clip(1.0 - ps, 1e-9, 1.0)), starts))   # P(no match)
    best_idx = np.zeros(len(starts), dtype=np.int64)
    best_val = e0.copy()
    # per-group argmax over k (vectorised via sort on (group, -e))
    o2 = np.lexsort((-e, grp))
    first = np.r_[True, grp[o2][1:] != grp[o2][:-1]]
    gmax_pos = o2[first]
    gmax_val = e[gmax_pos]
    take = gmax_val > best_val
    best_idx[take] = k[gmax_pos[take]]
    chosen_k = best_idx[grp]
    sel[order[k <= chosen_k]] = True
    return sel


def _grid():
    return np.round(np.arange(0.05, 0.96, 0.01), 2)


def choose_decision(s1_code, cand, src, p, y, ntrue, n_s1, allow_exclusive=True, log=print):
    """Grid-search the decision rule on OOF predictions; returns (best_rule, table)."""
    results = []
    excl_options = [False, True] if allow_exclusive else [False]
    for excl in excl_options:
        base = exclusive_mask(cand, p) if excl else np.ones(len(p), dtype=bool)
        # global threshold
        best_t, best_f = 0.5, -1.0
        for t in _grid():
            f = macro_f05_from_arrays(s1_code, base & (p >= t), y, ntrue, n_s1)
            if f > best_f:
                best_t, best_f = float(t), f
        results.append((best_f, dict(method="thr", exclusive=excl, t=best_t)))
        # per-source thresholds (coordinate ascent from the global optimum)
        t2 = t3 = best_t
        fcur = best_f
        for _ in range(2):
            for which in (2, 3):
                for t in _grid():
                    a, b = (t, t3) if which == 2 else (t2, t)
                    thr = np.where(src == 2, a, b)
                    f = macro_f05_from_arrays(s1_code, base & (p >= thr), y, ntrue, n_s1)
                    if f > fcur + 1e-9:
                        fcur, t2, t3 = f, a, b
        results.append((fcur, dict(method="thr_src", exclusive=excl, t2=float(t2), t3=float(t3))))
        # expected-F0.5 set selection
        for floor in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4):
            idx = np.flatnonzero(base)
            m = np.zeros(len(p), dtype=bool)
            m[idx[expf_mask(s1_code[idx], p[idx], floor)]] = True
            f = macro_f05_from_arrays(s1_code, m, y, ntrue, n_s1)
            results.append((f, dict(method="expf", exclusive=excl, floor=floor)))
    results.sort(key=lambda r: -r[0])
    for f, r in results[:8]:
        log(f"    OOF macro-F0.5 {f:.5f}  {r}")
    return results[0][1], results


def apply_decision(rule, s1_code, cand, src, p):
    base = exclusive_mask(cand, p) if rule["exclusive"] else np.ones(len(p), dtype=bool)
    if rule["method"] == "thr":
        return base & (p >= rule["t"])
    if rule["method"] == "thr_src":
        return base & (p >= np.where(src == 2, rule["t2"], rule["t3"]))
    idx = np.flatnonzero(base)
    m = np.zeros(len(p), dtype=bool)
    m[idx[expf_mask(s1_code[idx], p[idx], rule["floor"])]] = True
    return m
