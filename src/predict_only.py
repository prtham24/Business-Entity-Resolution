"""Test-only inference with already-trained fold models + decision-threshold variants.

Re-runs exactly the test stage of run_pipeline.py (same blocking, pruning rule and features), loads
the saved LightGBM fold models, saves the match probabilities, and writes one output folder per
threshold scale, with label-free diagnostics (predicted-empty rate, predicted matches per entity)
to compare with the known training statistics (5.7% singletons, 3.45 matches per S1 entity).

Usage:
    python src/predict_only.py --data-dir <dataset> --models-dir <folder with lgb_fold*.txt> \
        --out-dir variants --scales 1.0,0.9,0.8,0.7
"""
import argparse
import glob
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import blocking  # noqa: E402
import features  # noqa: E402
import matching  # noqa: E402
from config import Config  # noqa: E402
from data_io import load_split, prepare_records, self_validate, write_id_lists  # noqa: E402

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def scaled_rule(rule, scale):
    r = dict(rule)
    for k in ("t", "t2", "t3"):
        if k in r:
            r[k] = float(min(0.99, r[k] * scale))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--out-dir", default="variants")
    ap.add_argument("--k-per-view", type=int, default=12)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--prune", default='{"score": "s_mean", "tau": 0.0, "K": 20, "R": 8}')
    ap.add_argument("--rule", default='{"method": "thr_src", "exclusive": true, "t2": 0.82, "t3": 0.77}')
    ap.add_argument("--scales", default="1.0,0.9,0.8,0.7")
    ap.add_argument("--block-budget", type=float, default=None,
                    help="blocking cost cap per sparse product (final test run used 1e10)")
    args = ap.parse_args()

    cfg = Config()
    cfg.block_k_per_view = args.k_per_view
    if args.n_jobs:
        cfg.n_jobs = args.n_jobs
    if args.block_budget:
        cfg.block_cost_budget = args.block_budget
    prune = json.loads(args.prune)
    rule = json.loads(args.rule)
    scales = [float(x) for x in args.scales.split(",")]
    files = sorted(glob.glob(os.path.join(args.models_dir, "**", "lgb_fold*.txt"), recursive=True))
    if not files:
        raise SystemExit(f"no lgb_fold*.txt found under {args.models_dir}")
    models = [lgb.Booster(model_file=f) for f in files]
    feat_names = models[0].feature_name()
    log(f"loaded {len(models)} models ({len(feat_names)} features); prune={prune}; rule={rule}")
    os.makedirs(args.out_dir, exist_ok=True)

    Rt = load_split(args.data_dir, "test")
    log(f"  records per source: {Rt['src'].value_counts().sort_index().to_dict()}")
    Rt = prepare_records(Rt, cfg.n_jobs, log)
    Nt = len(Rt)
    log("Blocking (test) ...")
    Ut = blocking.generate_union(Rt, cfg, True, log)
    Pt = blocking.apply_pruning(Ut, prune)
    del Ut
    s1_rows = np.flatnonzero(Rt["src"].values == 1)
    s1_code = np.full(Nt, -1, dtype=np.int64)
    s1_code[s1_rows] = np.arange(len(s1_rows))
    n_s1 = len(s1_rows)
    ci = s1_code[Pt["i"].values]
    per_s1 = np.bincount(ci, minlength=n_s1)
    log(f"  test candidates: {len(Pt):,} | avg/S1 {per_s1.mean():.2f} | max {per_s1.max()}")
    pt = features.predict_by_country(Rt, Pt, cfg, models, feat_names, lambda ms, X: np.mean(
        [m.predict(X) for m in ms], axis=0), log)

    ids = Rt["entity_id"].values
    np.savez(os.path.join(args.out_dir, "test_probs.npz"), s1=ids[Pt["i"].values].astype(str),
             cand=ids[Pt["j"].values].astype(str), src=Pt["src_j"].values, p=pt.astype(np.float32),
             s_blk=Pt["s_blk"].values)
    log("  saved probabilities to test_probs.npz")

    s1_order = ids[s1_rows].tolist()
    o = np.lexsort((-Pt["s_blk"].values, Pt["i"].values))
    ck = Rt["country_key"].values[s1_rows]
    summary = []
    for sc in scales:
        r = scaled_rule(rule, sc)
        m = matching.apply_decision(r, ci, Pt["j"].values, Pt["src_j"].values, pt)
        d = os.path.join(args.out_dir, f"scale_{sc:.2f}")
        os.makedirs(d, exist_ok=True)
        write_id_lists(os.path.join(d, "candidate_pairs.tsv"), s1_order, ids[Pt["i"].values[o]],
                       ids[Pt["j"].values[o]], "candidate_entity_ids")
        sel = np.flatnonzero(m)
        so = sel[np.lexsort((-pt[sel], Pt["i"].values[sel]))]
        write_id_lists(os.path.join(d, "matching_results.tsv"), s1_order, ids[Pt["i"].values[so]],
                       ids[Pt["j"].values[so]], "matched_entity_ids")
        npred = np.bincount(ci[sel], minlength=n_s1)
        row = dict(scale=sc, rule=r, predicted_empty=float((npred == 0).mean()),
                   matches_per_s1=float(npred.mean()),
                   by_country={str(c): dict(empty=round(float((npred[ck == c] == 0).mean()), 4),
                                            per_s1=round(float(npred[ck == c].mean()), 3))
                               for c in pd.unique(ck)})
        summary.append(row)
        log(f"  scale {sc:.2f}: empty={row['predicted_empty']:.2%}  matches/S1={row['matches_per_s1']:.3f}  "
            f"{row['by_country']}")
    problems = self_validate(os.path.join(args.out_dir, f"scale_{scales[0]:.2f}", "matching_results.tsv"),
                             os.path.join(args.out_dir, f"scale_{scales[0]:.2f}", "candidate_pairs.tsv"), Rt)
    log("Self-validation " + ("PASSED" if not problems else "PROBLEMS: " + "; ".join(problems)))
    with open(os.path.join(args.out_dir, "variants_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log("Reference (train): 5.7% singletons, 3.45 true matches per S1 entity. Done.")


if __name__ == "__main__":
    main()
