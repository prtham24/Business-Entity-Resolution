"""Pairwise features for every (Source-1, candidate) pair.

All features are country-agnostic string/set similarities (no country one-hot), so the model
transfers to countries unseen in training (France).

Groups
1. TF-IDF cosines: name char 2-4-grams, name word 1-2-grams, address char 3-grams, address words.
2. rapidfuzz string similarities on core names (ratio, partial, token sort/set, Jaro-Winkler),
   no-space ratio, transliteration-skeleton ratio, best DBA-variant match, acronym match.
3. Legal-suffix signature agreement.
4. Address: fuzzy ratios, city/area tail match, number-set Jaccard, postal-code and house-number
   agreement, landmark flag, missing-address flags (NaN = missing; LightGBM handles NaN natively).
5. Blocking context: view cosines, forward/reverse ranks, union neighbourhood sizes.
6. Group-relative features: margin of this pair vs the best competing candidate of the same S1
   entity and vs the best competing S1 entity for the same candidate (label-free, cheap, and
   very informative because Source 1 is deduplicated).
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

from blocking import rowwise_dot
from data_io import prepare_details

STRING_FEATURES = [
    "name_ratio", "name_partial", "name_tsort", "name_tset", "name_jw", "name_norm_tset",
    "name_nospace_ratio", "name_skel_ratio", "name_var_max", "name_acronym", "name_first_eq",
    "name_tok_jacc", "name_contain", "name_len_diff", "name_char_ratio",
    "legal_both", "legal_same",
    "addr_ratio", "addr_tset", "addr_tsort", "addr_partial", "addr_tail_tset", "addr_skel_ratio",
    "num_jacc", "postal_eq", "firstnum_eq", "addr_missing", "landmark_any", "addr_len_ratio",
]
_REC_COLS = ["name_core", "name_norm", "name_skel", "name_var", "name_legal", "name_acr",
             "addr_norm", "addr_tail", "addr_skel", "addr_nums", "addr_postal", "addr_first", "addr_landmark"]
NAN = float("nan")


def _string_feats_chunk(a: dict, b: dict) -> np.ndarray:
    ratio, pratio, tsort, tset = fuzz.ratio, fuzz.partial_ratio, fuzz.token_sort_ratio, fuzz.token_set_ratio
    jw = JaroWinkler.normalized_similarity
    rows = []
    for k in range(len(a["name_core"])):
        c1, c2 = a["name_core"][k], b["name_core"][k]
        t1, t2 = c1.split(), c2.split()
        s1, s2 = set(t1), set(t2)
        uni = s1 | s2
        ns1, ns2 = c1.replace(" ", ""), c2.replace(" ", "")
        f_tset = tset(c1, c2)
        v1 = a["name_var"][k].split("|") if a["name_var"][k] else [c1]
        v2 = b["name_var"][k].split("|") if b["name_var"][k] else [c2]
        var_max = max(tset(x, y) for x in v1 for y in v2) if (len(v1) > 1 or len(v2) > 1) else f_tset
        var_max = max(var_max, f_tset)
        ac1, ac2 = a["name_acr"][k], b["name_acr"][k]
        acr = 1.0 if ((ac1 and (ac1 == ns2 or ac1 in s2)) or (ac2 and (ac2 == ns1 or ac2 in s1))) else 0.0
        lg1, lg2 = a["name_legal"][k], b["name_legal"][k]

        d1, d2 = a["addr_norm"][k], b["addr_norm"][k]
        if d1 and d2:
            ad = (ratio(d1, d2), tset(d1, d2), tsort(d1, d2), pratio(d1, d2))
            ta, tb = a["addr_tail"][k], b["addr_tail"][k]
            tail = tset(ta, tb) if (ta and tb) else NAN
            askel = ratio(a["addr_skel"][k], b["addr_skel"][k])
            alen = min(len(d1), len(d2)) / max(len(d1), len(d2))
            amiss = 0.0
        else:
            ad, tail, askel, alen = (NAN, NAN, NAN, NAN), NAN, NAN, NAN
            amiss = 1.0 if (d1 or d2) else 2.0
        n1 = set(a["addr_nums"][k].split()); n2 = set(b["addr_nums"][k].split())
        num_jacc = len(n1 & n2) / len(n1 | n2) if (n1 and n2) else NAN
        p1 = set(a["addr_postal"][k].split()); p2 = set(b["addr_postal"][k].split())
        postal = (1.0 if p1 & p2 else 0.0) if (p1 and p2) else NAN
        f1, f2 = a["addr_first"][k], b["addr_first"][k]
        firstnum = (1.0 if f1 == f2 else 0.0) if (f1 and f2) else NAN

        rows.append((
            ratio(c1, c2), pratio(c1, c2), tsort(c1, c2), f_tset, jw(c1, c2),
            tset(a["name_norm"][k], b["name_norm"][k]),
            ratio(ns1, ns2), ratio(a["name_skel"][k], b["name_skel"][k]), var_max, acr,
            1.0 if (t1 and t2 and t1[0] == t2[0]) else 0.0,
            len(s1 & s2) / len(uni) if uni else 0.0,
            1.0 if (ns1 and ns2 and (ns1 in ns2 or ns2 in ns1)) else 0.0,
            abs(len(t1) - len(t2)),
            min(len(c1), len(c2)) / max(len(c1), len(c2), 1),
            1.0 if (lg1 and lg2) else 0.0,
            1.0 if (lg1 and lg1 == lg2) else 0.0,
            *ad, tail, askel, num_jacc, postal, firstnum, amiss,
            float(max(a["addr_landmark"][k], b["addr_landmark"][k])), alen,
        ))
    return np.asarray(rows, dtype=np.float32).reshape(-1, len(STRING_FEATURES))


def _tfidf_views(D):
    kw = dict(sublinear_tf=True, dtype=np.float32)
    return {
        "cos_name_char": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), **kw).fit_transform(D["name_core"].values).tocsr(),
        "cos_name_word": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b", **kw).fit_transform(D["name_core"].values).tocsr(),
        "cos_addr_char": TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), **kw).fit_transform(D["addr_norm"].values).tocsr(),
        "cos_addr_word": TfidfVectorizer(analyzer="word", token_pattern=r"(?u)\b\w+\b", **kw).fit_transform(D["addr_norm"].values).tocsr(),
    }


def _add_group_relative(X: pd.DataFrame, gi: np.ndarray, gj: np.ndarray, cols):
    """margin_fwd: value minus best OTHER candidate of the same (S1, source);
       margin_rev: value minus best OTHER S1 entity competing for the same candidate."""
    for c in cols:
        v = X[c].fillna(-1.0).values.astype(np.float32)
        for tag, g in (("fwd", gi), ("rev", gj)):
            s = pd.Series(v)
            top1 = s.groupby(g).transform("max").values
            n_top = (pd.Series((v >= top1).astype(np.int32)).groupby(g).transform("sum")).values
            below = np.where(v < top1, v, -np.inf)
            top2 = pd.Series(below).groupby(g).transform("max").values
            top2 = np.where(np.isfinite(top2), top2, -1.0)
            other_best = np.where(v >= top1, np.where(n_top > 1, top1, top2), top1)
            X[f"{c}_margin_{tag}"] = (v - other_best).astype(np.float32)
            X[f"{c}_is_best_{tag}"] = (v >= top1).astype(np.float32)


def build_features(R: pd.DataFrame, P: pd.DataFrame, cfg, log=print) -> pd.DataFrame:
    """R: all records of the split; P: pruned candidate pairs (global row indices i, j)."""
    t0 = time.time()
    involved = np.unique(np.concatenate([P["i"].values, P["j"].values]))
    D = prepare_details(R.iloc[involved].reset_index(drop=True), cfg.n_jobs)
    i = np.searchsorted(involved, P["i"].values)          # local indices into D
    j = np.searchsorted(involved, P["j"].values)
    log(f"  normalised details for {len(D):,} involved records ({time.time() - t0:.1f}s)")
    X = pd.DataFrame(index=np.arange(len(P)))

    for name, M in _tfidf_views(D).items():
        X[name] = rowwise_dot(M, i, j)
    log(f"  tfidf cosines done ({time.time() - t0:.1f}s)")

    rec = {c: D[c].values for c in _REC_COLS}
    chunks = [(s, min(s + cfg.feature_chunk, len(P))) for s in range(0, len(P), cfg.feature_chunk)]

    def payload(s, e):
        ii, jj = i[s:e], j[s:e]
        return ({c: rec[c][ii].tolist() for c in _REC_COLS}, {c: rec[c][jj].tolist() for c in _REC_COLS})

    if len(chunks) > 1 and cfg.n_jobs > 1:
        parts = Parallel(n_jobs=cfg.n_jobs, backend="loky")(
            delayed(_string_feats_chunk)(*payload(s, e)) for s, e in chunks)
    else:
        parts = [_string_feats_chunk(*payload(s, e)) for s, e in chunks]
    S = np.vstack(parts) if parts else np.zeros((0, len(STRING_FEATURES)), np.float32)
    for k, name in enumerate(STRING_FEATURES):
        X[name] = S[:, k]
    log(f"  string features done ({time.time() - t0:.1f}s)")

    X["is_s3"] = (P["src_j"].values == 3).astype(np.float32)
    for c in ("s_name", "s_full", "s_blk", "rank_fwd", "rank_rev", "n_fwd_union", "n_rev_union"):
        X[c] = P[c].values.astype(np.float32)
    gi = P["i"].values.astype(np.int64) * 4 + P["src_j"].values
    gj = P["j"].values
    X["n_cand_fwd"] = pd.Series(gi).groupby(gi).transform("size").values.astype(np.float32)
    X["n_cand_rev"] = pd.Series(gj).groupby(gj).transform("size").values.astype(np.float32)
    _add_group_relative(X, gi, gj, ["cos_name_char", "name_tset", "addr_tset", "cos_addr_word", "s_blk"])
    log(f"  {X.shape[1]} features for {len(X):,} pairs ({time.time() - t0:.1f}s)")
    return X.astype(np.float32)


# --------------------------------------------------------------------------- per-country processing
def _country_chunks(R: pd.DataFrame, P: pd.DataFrame):
    """Pair indices grouped by the Source-1 entity's country. Blocking never crosses countries,
    so every candidate's competing S1 entities are in the same chunk: group features stay exact.
    Processing one chunk at a time bounds peak memory on large test sets."""
    ck = R["country_key"].values[P["i"].values]
    return [np.flatnonzero(ck == c) for c in pd.unique(ck)]


def build_features_by_country(R: pd.DataFrame, P: pd.DataFrame, cfg, log=print) -> pd.DataFrame:
    chunks = _country_chunks(R, P)
    if len(chunks) <= 1:
        return build_features(R, P, cfg, log)
    arr, cols = None, None
    for idx in chunks:
        log(f"  feature chunk: {len(idx):,} pairs")
        Xc = build_features(R, P.iloc[idx].reset_index(drop=True), cfg, log)
        if arr is None:
            cols = list(Xc.columns)
            arr = np.empty((len(P), len(cols)), dtype=np.float32)
        arr[idx] = Xc[cols].values
        del Xc
    return pd.DataFrame(arr, columns=cols)


def predict_by_country(R: pd.DataFrame, P: pd.DataFrame, cfg, models, feat_names, predict_fn, log=print):
    import gc
    p = np.zeros(len(P), dtype=np.float64)
    for idx in _country_chunks(R, P):
        log(f"  feature chunk: {len(idx):,} pairs")
        Xc = build_features(R, P.iloc[idx].reset_index(drop=True), cfg, log)[feat_names]
        log("  predicting chunk ...")
        p[idx] = predict_fn(models, Xc)
        del Xc
        gc.collect()
    return p
