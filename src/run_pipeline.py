"""End-to-end pipeline: data -> blocking -> features -> LightGBM -> decision -> output TSVs.

Usage (from code/business_entity_resolution/):
    python src/run_pipeline.py --data-dir /path/to/student_resource/dataset --out-dir ../../output

Writes:
    <out-dir>/matching_results.tsv   (upload this to the leaderboard)
    <out-dir>/candidate_pairs.tsv    (exact set the model scored)
    <artifacts-dir>/report.json, models, blocking + decision parameters
"""
import argparse
import gc
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocking  # noqa: E402
import features  # noqa: E402
import matching  # noqa: E402
from config import Config  # noqa: E402
from data_io import (count_lines, load_split, load_train_sampled,  # noqa: E402
                     macro_f05_from_arrays, prepare_records, self_validate, write_id_lists)

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def pair_keys(i, j, N):
    return i.astype(np.int64) * N + j.astype(np.int64)


def gt_structures(R, gt):
    """Integer GT pair keys + n_true per S1 (only S1 present in R)."""
    N = len(R)
    idx = {}
    for k, e in enumerate(R["entity_id"].values):
        idx.setdefault(e, k)
    s1_rows = np.flatnonzero(R["src"].values == 1)
    s1_code = np.full(N, -1, dtype=np.int64)
    s1_code[s1_rows] = np.arange(len(s1_rows))
    ntrue = np.zeros(len(s1_rows), dtype=np.int64)
    keys, unknown = [], 0
    for s1, lst in gt.items():
        if s1 not in idx:
            continue
        a = idx[s1]
        ntrue[s1_code[a]] = len(lst)
        for m in lst:
            if m in idx:
                keys.append(a * N + idx[m])
            else:
                unknown += 1
    if unknown:
        log(f"  WARNING: {unknown} ground-truth ids not found among records")
    return np.array(sorted(set(keys)), dtype=np.int64), ntrue, s1_code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder containing train/ and test/")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--quick", action="store_true",
                    help="small train sample (~40k S1 entities) + 3 folds: fast first submission")
    ap.add_argument("--train-frac", type=float, default=None,
                    help="override the train sampling fraction (default: auto = test size / train size)")
    ap.add_argument("--max-train-records", type=int, default=None, help="cap on sampled train records")
    ap.add_argument("--block-budget", type=float, default=None,
                    help="max multiply-adds per blocking product before common n-grams are skipped (default 5e10)")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--k-per-view", type=int, default=None)
    ap.add_argument("--target-keep", type=float, default=None,
                    help="fraction of reachable matches the pruned candidate set must keep (default 0.995)")
    ap.add_argument("--max-avg-cands", type=float, default=None,
                    help="cap on average candidates per S1 entity chosen by pruning (default 10)")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--loco", action="store_true",
                    help="diagnostic: leave-one-country-out CV to estimate transfer to unseen countries")
    args = ap.parse_args()

    cfg = Config()
    if args.n_jobs:
        cfg.n_jobs = args.n_jobs
    if args.k_per_view:
        cfg.block_k_per_view = args.k_per_view
    if args.target_keep:
        cfg.block_target_keep = args.target_keep
    if args.folds:
        cfg.n_folds = args.folds
    if args.max_avg_cands is not None:
        cfg.block_max_avg_cands = args.max_avg_cands
    if args.max_train_records:
        cfg.max_train_records = args.max_train_records
    if args.block_budget:
        cfg.block_cost_budget = args.block_budget
    if args.quick:
        cfg.n_folds = min(cfg.n_folds, 3)
        cfg.lgb_params["learning_rate"] = 0.1
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    report = {"config": {k: v for k, v in vars(cfg).items() if k not in ("block_taus", "block_ks", "block_rs")}}

    # ================================================================ TRAIN
    # --- size the training sample from file sizes (cheap line counts)
    tr = lambda k: os.path.join(args.data_dir, "train", f"train_source{k}.tsv")  # noqa: E731
    te = lambda k: os.path.join(args.data_dir, "test", f"test_source{k}.tsv")  # noqa: E731
    n_tr = {k: count_lines(tr(k)) - 1 for k in (1, 2, 3)}
    n_te = {k: count_lines(te(k)) - 1 for k in (1, 2, 3)}
    log(f"File sizes  train: {n_tr}  test: {n_te}")
    if args.train_frac:
        frac = args.train_frac
    else:
        frac = min(1.0, n_te[1] / max(n_tr[1], 1))                         # match test crowding
        frac = min(frac, cfg.max_train_records / max(sum(n_tr.values()), 1))  # RAM/time guard
        if args.quick:
            frac = min(frac, cfg.quick_train_s1 / max(n_tr[1], 1))
    log(f"Train sampling fraction f = {frac:.4f} (~{int(frac * n_tr[1]):,} S1 entities)")
    report["train_sampling_fraction"] = frac

    log("Loading train ...")
    R, gt = load_train_sampled(args.data_dir, frac, cfg.seed, log)
    log(f"  records per source: {R['src'].value_counts().sort_index().to_dict()}")
    R = prepare_records(R, cfg.n_jobs, log)
    N = len(R)
    gt_keys, ntrue, s1_code_of_row = gt_structures(R, gt)
    n_s1 = len(ntrue)
    log(f"  S1 entities: {n_s1:,} | GT pairs: {len(gt_keys):,} | singletons: {(ntrue == 0).mean():.1%} "
        f"| mean matches/S1: {ntrue.mean():.2f}")

    # --- data diagnostics that drive design decisions
    gi, gj = gt_keys // N, gt_keys % N
    same_country = float((R["country_key"].values[gi] == R["country_key"].values[gj]).mean()) if len(gi) else 1.0
    by_country = {"yes": True, "no": False}.get(cfg.block_by_country, same_country >= 0.995)
    cnt = pd.Series(gj).value_counts()
    excl_frac = float((cnt == 1).mean()) if len(cnt) else 1.0
    log(f"  GT pairs with same country label: {same_country:.4f} -> block by country: {by_country}")
    log(f"  S2/S3 records matched to exactly one S1: {excl_frac:.4f} (exclusivity assumption)")
    report["train_data"] = dict(n_s1=n_s1, gt_pairs=int(len(gt_keys)), singleton_rate=float((ntrue == 0).mean()),
                                same_country=same_country, exclusive_frac=excl_frac)

    # --- blocking
    log("Blocking (train) ...")
    U = blocking.generate_union(R, cfg, by_country, log)
    yU = np.isin(pair_keys(U["i"].values, U["j"].values, N), gt_keys).astype(np.int8)
    union_recall = yU.sum() / max(len(gt_keys), 1)
    log(f"  union recall: {union_recall:.4f} ({yU.sum():,}/{len(gt_keys):,})")
    prune = blocking.tune_pruning(U, yU, n_s1, cfg, log)
    P = blocking.apply_pruning(U, prune)
    del U, yU
    gc.collect()
    y = np.isin(pair_keys(P["i"].values, P["j"].values, N), gt_keys).astype(np.int8)
    per_s1 = np.bincount(s1_code_of_row[P["i"].values], minlength=n_s1)
    block_stats = dict(pairs=int(len(P)), avg_candidates_per_s1=float(per_s1.mean()),
                       median=float(np.median(per_s1)), p95=float(np.percentile(per_s1, 95)),
                       max=int(per_s1.max()) if n_s1 else 0, recall_ceiling=float(y.sum() / max(len(gt_keys), 1)),
                       union_recall=float(union_recall), rule=prune)
    log(f"  candidates: {len(P):,} | avg/S1 {block_stats['avg_candidates_per_s1']:.2f} "
        f"| p95 {block_stats['p95']:.0f} | max {block_stats['max']} | recall ceiling {block_stats['recall_ceiling']:.4f}")
    report["blocking_train"] = block_stats

    # --- features + model
    log("Features (train) ...")
    X = features.build_features_by_country(R, P, cfg, log)
    feat_names = list(X.columns)
    groups = P["i"].values
    log(f"Training LightGBM ({cfg.n_folds}-fold GroupKFold, positives {y.mean():.3%}) ...")
    models, oof = matching.train_cv(X, y, groups, cfg, log)
    s1c = s1_code_of_row[P["i"].values]
    cand = P["j"].values
    src = P["src_j"].values
    log("Choosing decision rule on OOF predictions ...")
    rule, table = matching.choose_decision(s1c, cand, src, oof, y, ntrue, n_s1,
                                           allow_exclusive=excl_frac >= 0.98, log=log)
    oof_f = table[0][0]
    log(f"  chosen rule: {rule}  ->  OOF macro F0.5 = {oof_f:.5f}")
    # per-country OOF breakdown (useful for the write-up)
    ck_s1 = R["country_key"].values[np.flatnonzero(R["src"].values == 1)]
    m_final = matching.apply_decision(rule, s1c, cand, src, oof)
    per_country = {}
    for c in pd.unique(ck_s1):
        sel_s1 = np.flatnonzero(ck_s1 == c)
        remap = np.full(n_s1, -1); remap[sel_s1] = np.arange(len(sel_s1))
        rows = remap[s1c] >= 0
        per_country[str(c)] = macro_f05_from_arrays(remap[s1c[rows]], m_final[rows], y[rows],
                                                    ntrue[sel_s1], len(sel_s1))
    log(f"  OOF macro F0.5 by country: {per_country}")
    report["model"] = dict(oof_macro_f05=oof_f, rule=rule, oof_by_country=per_country,
                           n_features=len(feat_names),
                           best_iterations=[int(m.best_iteration) for m in models],
                           top_features=matching.feature_importance(models, feat_names))
    for k, m in enumerate(models):
        m.save_model(os.path.join(args.artifacts_dir, f"lgb_fold{k}.txt"))

    if args.loco:
        cty = R["country_key"].values[P["i"].values]
        if len(pd.unique(cty)) >= 2:
            log("LOCO diagnostic (train on other countries, predict held-out country) ...")
            import copy
            c2 = copy.deepcopy(cfg); c2.n_folds = len(pd.unique(cty))
            _, oof_loco = matching.train_cv(X, y, cty, c2, log)
            f_loco = macro_f05_from_arrays(s1c, matching.apply_decision(rule, s1c, cand, src, oof_loco),
                                           y, ntrue, n_s1)
            log(f"  LOCO macro F0.5 (pessimistic proxy for France) = {f_loco:.5f}")
            report["model"]["loco_macro_f05"] = f_loco
    del X, P, R, oof
    gc.collect()

    # ================================================================ TEST
    log("Loading test ...")
    Rt = load_split(args.data_dir, "test")
    log(f"  records per source: {Rt['src'].value_counts().sort_index().to_dict()}")
    Rt = prepare_records(Rt, cfg.n_jobs, log)
    Nt = len(Rt)
    log("Blocking (test) ...")
    Ut = blocking.generate_union(Rt, cfg, by_country, log)
    Pt = blocking.apply_pruning(Ut, prune)
    del Ut
    gc.collect()
    s1_rows_t = np.flatnonzero(Rt["src"].values == 1)
    s1_code_t = np.full(Nt, -1, dtype=np.int64); s1_code_t[s1_rows_t] = np.arange(len(s1_rows_t))
    per_s1_t = np.bincount(s1_code_t[Pt["i"].values], minlength=len(s1_rows_t))
    log(f"  test candidates: {len(Pt):,} | avg/S1 {per_s1_t.mean():.2f} | max {per_s1_t.max() if len(per_s1_t) else 0}")
    log("Features (test) ...")
    if len(Pt):
        pt = features.predict_by_country(Rt, Pt, cfg, models, feat_names, matching.predict, log)
    else:
        pt = np.zeros(0)
    mt = matching.apply_decision(rule, s1_code_t[Pt["i"].values], Pt["j"].values, Pt["src_j"].values, pt)

    ids = Rt["entity_id"].values
    s1_order = ids[s1_rows_t].tolist()
    o = np.lexsort((-Pt["s_blk"].values, Pt["i"].values))
    cand_path = os.path.join(args.out_dir, "candidate_pairs.tsv")
    write_id_lists(cand_path, s1_order, ids[Pt["i"].values[o]], ids[Pt["j"].values[o]], "candidate_entity_ids")
    sel = np.flatnonzero(mt)
    o = sel[np.lexsort((-pt[sel], Pt["i"].values[sel]))]
    match_path = os.path.join(args.out_dir, "matching_results.tsv")
    write_id_lists(match_path, s1_order, ids[Pt["i"].values[o]], ids[Pt["j"].values[o]], "matched_entity_ids")

    n_pred = np.bincount(s1_code_t[Pt["i"].values[sel]], minlength=len(s1_rows_t))
    ck_t = Rt["country_key"].values[s1_rows_t]
    by_c = {str(c): dict(n_s1=int((ck_t == c).sum()),
                         avg_candidates=float(per_s1_t[ck_t == c].mean()),
                         predicted_empty=float((n_pred[ck_t == c] == 0).mean()))
            for c in pd.unique(ck_t)}
    log(f"  test per-country: {by_c}")
    report["test"] = dict(n_s1=len(s1_rows_t), candidate_pairs=int(len(Pt)),
                          avg_candidates_per_s1=float(per_s1_t.mean()), predicted_pairs=int(mt.sum()),
                          predicted_empty_rate=float((n_pred == 0).mean()), by_country=by_c)

    problems = self_validate(match_path, cand_path, Rt)
    if problems:
        log("SELF-VALIDATION PROBLEMS:\n  " + "\n  ".join(problems))
    else:
        log("Self-validation PASSED (still run utils/validate_submission.py before uploading)")
    report["self_validation"] = problems or "PASS"
    with open(os.path.join(args.artifacts_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"Done. Outputs in {os.path.abspath(args.out_dir)} ; report in {args.artifacts_dir}/report.json")


if __name__ == "__main__":
    main()
