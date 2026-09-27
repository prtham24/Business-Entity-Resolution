"""Blocking / candidate generation (scales to millions of records; never all-pairs).

Stage A - retrieval:
  Two TF-IDF views, each L2-normalised so dot product == cosine:
    * "name": character 3-grams of the core business name (typos, abbreviations, transliteration)
    * "full": word tokens of core name + normalised address (street names, PIN/ZIP codes, city)
  The exact cost of each sparse product is sum_f df_A(f)*df_B(f); if it exceeds block_cost_budget
  the most common features (e.g. "ent", "rd") are skipped for RETRIEVAL only (little identity
  signal, but they dominate the cost). Similarity scores always use all features.
  For every Source-1 entity we retrieve the top-k Source-2 and, separately, top-k Source-3 records
  per view with a multi-threaded sparse top-n product (sparse_dot_topn), within the same country
  label when training shows matches never cross countries (country = open set of labels; an unseen
  label such as France simply forms its own block).

Stage B - pruning (this is what candidate_pairs.tsv contains):
  For every retrieved pair: both view cosines, a combined score, the forward rank (rank of the
  candidate among the S1 entity's candidates from that source) and the reverse rank (rank of the S1
  entity among all S1 entities that retrieved that candidate; Source 1 is deduplicated, so a record
  much closer to another S1 entity is rarely a match). The rule
      score >= tau AND fwd_rank <= K AND rev_rank <= R
  is tuned on training labels to keep >= block_target_keep of the matches stage A found, with the
  fewest candidates per Source-1 entity.
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

_BACKEND_LOGGED = False


# --------------------------------------------------------------------------- vectorisation
def _tfidf(texts, **kw):
    for min_df in (2, 1):
        try:
            return TfidfVectorizer(sublinear_tf=True, dtype=np.float32, min_df=min_df,
                                   **kw).fit_transform(texts).tocsr()
        except ValueError:
            continue
    raise ValueError("could not vectorise blocking texts")


def build_blocking_matrices(R: pd.DataFrame) -> dict:
    name_txt = R["name_core"].values
    full_txt = (R["name_core"] + " " + R["addr_norm"]).values
    return {
        "name": _tfidf(name_txt, analyzer="char_wb", ngram_range=(3, 3)),
        "full": _tfidf(full_txt, analyzer="word", token_pattern=r"(?u)\b\w+\b"),
    }


def cap_cost(A, B, budget):
    """The exact cost of the sparse product A @ B.T is sum_f df_A(f) * df_B(f). If it exceeds
    `budget`, drop the most expensive (most common) features for RETRIEVAL only.
    Returns (A', B', n_dropped)."""
    V = A.shape[1]
    ca = np.bincount(A.indices, minlength=V).astype(np.float64)
    cb = np.bincount(B.indices, minlength=V).astype(np.float64)
    cost = ca * cb
    total = cost.sum()
    if total <= budget:
        return A, B, 0
    order = np.argsort(-cost)
    remaining = total - np.cumsum(cost[order])
    n_drop = int(np.argmax(remaining <= budget)) + 1
    D = sp.diags((~np.isin(np.arange(V), order[:n_drop])).astype(np.float32))
    A2, B2 = (A @ D).tocsr(), (B @ D).tocsr()
    A2.eliminate_zeros()          # make sure dropped features cost nothing downstream
    B2.eliminate_zeros()
    return A2, B2, n_drop


# --------------------------------------------------------------------------- sparse top-k
def topk_rows(A, B, k, threshold, n_threads, log=print):
    """For each row of A: top-k rows of B by dot product (> threshold). Returns (rows, cols)."""
    global _BACKEND_LOGGED
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    k = int(min(k, B.shape[0]))
    BT = B.T.tocsr()
    try:
        from sparse_dot_topn import sp_matmul_topn
        if not _BACKEND_LOGGED:
            log("  blocking backend: sparse_dot_topn (fast)")
            _BACKEND_LOGGED = True
        C = sp_matmul_topn(A, BT, top_n=k, threshold=threshold, sort=False, n_threads=n_threads).tocoo()
        return C.row.astype(np.int64), C.col.astype(np.int64)
    except ImportError:
        if not _BACKEND_LOGGED:
            log("  WARNING: sparse_dot_topn not installed -> slow scipy fallback. "
                "Run: pip install sparse-dot-topn")
            _BACKEND_LOGGED = True
    # fallback: chunked sparse product, top-k per row on the sparse result (adaptive chunk size)
    rows, cols = [], []
    s, chunk = 0, 1000
    while s < A.shape[0]:
        S = (A[s:s + chunk] @ BT).tocsr()
        n_rows = S.shape[0]
        r = np.repeat(np.arange(n_rows), np.diff(S.indptr))
        v, c = S.data, S.indices
        m = v > threshold
        r, v, c = r[m], v[m], c[m]
        if len(v):
            rank, _ = group_rank_desc(r, v)
            keep = rank <= k
            rows.append(r[keep] + s); cols.append(c[keep].astype(np.int64))
        s += n_rows
        chunk = int(np.clip(chunk * 3e7 / max(S.nnz, 1), 50, 20000))
    if not rows:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate(rows).astype(np.int64), np.concatenate(cols)


def rowwise_dot(M, i, j, chunk=500_000):
    """Exact cosine for arbitrary pairs (rows of L2-normalised sparse matrix M)."""
    out = np.empty(len(i), dtype=np.float32)
    for s in range(0, len(i), chunk):
        out[s:s + chunk] = np.asarray(M[i[s:s + chunk]].multiply(M[j[s:s + chunk]]).sum(axis=1)).ravel()
    return out


def group_rank_desc(group, score):
    """Rank (1 = best) of each row within its group by descending score, and the group size."""
    n = len(group)
    if n == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    order = np.lexsort((-score, group))
    g = group[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    new[1:] = g[1:] != g[:-1]
    starts = np.flatnonzero(new)
    sizes = np.diff(np.r_[starts, n])
    rank = np.empty(n, dtype=np.int32)
    rank[order] = np.arange(n) - np.repeat(starts, sizes) + 1
    size = np.empty(n, dtype=np.int32)
    size[order] = np.repeat(sizes, sizes)
    return rank, size


def _country_groups(s1, tgt, ck, by_country):
    if not by_country:
        return [(s1, tgt)]
    groups = []
    tgt_ck = ck[tgt]
    for c in pd.unique(ck[s1]):
        a = s1[ck[s1] == c]
        if c == "":
            b = tgt
        else:
            b = tgt[(tgt_ck == c) | (tgt_ck == "")]
            if len(b) == 0:          # label never seen in this source: fall back to global search
                b = tgt
        groups.append((a, b))
    return groups


# --------------------------------------------------------------------------- stage A
def generate_union(R: pd.DataFrame, cfg, by_country: bool, log=print) -> pd.DataFrame:
    t0 = time.time()
    mats = build_blocking_matrices(R)
    src = R["src"].values.astype(np.int8)
    empty_name = float((np.diff(mats["name"].indptr) == 0)[src == 1].mean())
    log(f"  blocking matrices built ({time.time() - t0:.0f}s); "
        f"S1 with empty name vector: {empty_name:.2%}")
    ck = R["country_key"].values
    N = len(R)
    s1 = np.flatnonzero(src == 1)
    keys = []
    for t in (2, 3):
        tgt = np.flatnonzero(src == t)
        if len(tgt) == 0:
            continue
        for a, b in _country_groups(s1, tgt, ck, by_country):
            for name, M in mats.items():
                t1 = time.time()
                A, B, n_drop = cap_cost(M[a], M[b], cfg.block_cost_budget)
                if n_drop:
                    log(f"    S{t} view={name}: skipping {n_drop} most common features for retrieval (cost cap)")
                r, c = topk_rows(A, B, cfg.block_k_per_view, cfg.block_min_sim, cfg.n_jobs, log)
                del A, B
                if len(r):
                    keys.append(a[r].astype(np.int64) * N + b[c].astype(np.int64))
                if time.time() - t1 > 20:
                    log(f"    S{t} view={name} {len(a):,}x{len(b):,} done in {time.time() - t1:.0f}s")
    keys = np.unique(np.concatenate(keys)) if keys else np.empty(0, np.int64)
    i = (keys // N).astype(np.int64)
    j = (keys % N).astype(np.int64)
    del keys
    s_name = rowwise_dot(mats["name"], i, j)
    s_full = rowwise_dot(mats["full"], i, j)
    del mats
    src_j = src[j]
    cols = {"i": i, "j": j, "src_j": src_j, "s_name": s_name, "s_full": s_full,
            "s_max": np.maximum(s_name, s_full), "s_mean": 0.5 * (s_name + s_full)}
    g_fwd = i * 4 + src_j
    for sc in ("s_max", "s_mean"):
        cols[f"rf_{sc}"], n_fwd = group_rank_desc(g_fwd, cols[sc])
        cols[f"rr_{sc}"], n_rev = group_rank_desc(j, cols[sc])
    cols["n_fwd_union"], cols["n_rev_union"] = n_fwd, n_rev
    P = pd.DataFrame(cols)
    log(f"  stage-A union: {len(P):,} pairs ({len(P) / max(len(s1), 1):.1f} per S1) "
        f"in {time.time() - t0:.0f}s")
    return P


# --------------------------------------------------------------------------- stage B
def tune_pruning(P: pd.DataFrame, y: np.ndarray, n_s1: int, cfg, log=print) -> dict:
    """Choose the pruning rule (score, tau, K, R).

    1) Preferred: the smallest candidate set keeping >= block_target_keep of the matches in the union.
    2) If that needs more than block_max_avg_cands candidates per S1 entity (dense data), instead keep
       the MOST matches possible within that budget (also requiring K <= budget, so the per-entity
       worst case stays bounded on a denser test set). Bounds memory/time at test time.
    """
    union_hits = max(int(y.sum()), 1)
    target = cfg.block_target_keep * union_hits
    max_avg = float(getattr(cfg, "block_max_avg_cands", 10.0) or 0)
    budget = max_avg * n_s1 if max_avg > 0 else float("inf")
    best, best_budget = None, None
    ks = [k for k in cfg.block_ks if k <= 2 * cfg.block_k_per_view] or [max(cfg.block_ks)]
    for sc in ("s_max", "s_mean"):
        order = np.argsort(-P[sc].values, kind="stable")      # sort ONCE per score
        neg_sorted = -P[sc].values[order]
        ys = y[order].astype(np.int64)
        rf = P[f"rf_{sc}"].values[order]
        rr = P[f"rr_{sc}"].values[order]
        cut = [int(np.searchsorted(neg_sorted, -tau, side="right")) for tau in cfg.block_taus]
        for K in ks:
            mK = rf <= K
            for Rr in cfg.block_rs:
                m = mK & (rr <= Rr)
                cnt_cum = np.cumsum(m)
                hit_cum = np.cumsum(ys * m)
                for tau, c in zip(cfg.block_taus, cut):
                    cnt = int(cnt_cum[c - 1]) if c > 0 else 0
                    hits = int(hit_cum[c - 1]) if c > 0 else 0
                    rule = dict(score=sc, tau=float(tau), K=int(K), R=int(Rr))
                    if hits >= target and (best is None or (cnt, -hits) < best[0]):
                        best = ((cnt, -hits), rule)
                    if cnt <= budget and K <= max(max_avg, 1) and (
                            best_budget is None or (-hits, cnt) < best_budget[0]):
                        best_budget = ((-hits, cnt), rule)
    if best is not None and best[0][0] <= budget:
        params, cnt, hits = best[1], best[0][0], -best[0][1]
    elif best_budget is not None:
        params, cnt, hits = best_budget[1], best_budget[0][1], -best_budget[0][0]
        log(f"  keeping {cfg.block_target_keep:.1%} of matches would need too many candidates; "
            f"using the best rule within {max_avg:g} candidates/S1")
    else:
        params, cnt, hits = dict(score="s_max", tau=0.0, K=max(ks), R=max(cfg.block_rs)), len(P), union_hits
    log(f"  pruning rule: {params}  (keeps {hits / union_hits:.2%} of union matches, "
        f"{cnt / max(n_s1, 1):.2f} candidates/S1 on train)")
    return params


def apply_pruning(P: pd.DataFrame, params: dict) -> pd.DataFrame:
    sc = params["score"]
    m = ((P[sc].values >= params["tau"]) & (P[f"rf_{sc}"].values <= params["K"])
         & (P[f"rr_{sc}"].values <= params["R"]))
    Q = P.loc[m, ["i", "j", "src_j", "s_name", "s_full", "n_fwd_union", "n_rev_union"]].copy()
    Q["s_blk"] = P.loc[m, sc].values
    Q["rank_fwd"] = P.loc[m, f"rf_{sc}"].values
    Q["rank_rev"] = P.loc[m, f"rr_{sc}"].values
    return Q.reset_index(drop=True)
