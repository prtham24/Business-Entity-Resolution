"""Score a matching_results.tsv against a ground-truth file with the challenge metric:
macro F0.5 over Source-1 entities (singletons included: empty prediction for an entity with no
true matches scores 1.0, any prediction for it scores 0.0).

Usage: python scripts/score.py --pred output/matching_results.tsv --gt data_synth/test/test_ground_truth.tsv
       [--cand output/candidate_pairs.tsv]   (also reports blocking recall and candidate-set size)
"""
import argparse
import csv


def load(path):
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row:
                out[row[0]] = {x for x in (row[1] if len(row) > 1 else "").split(",") if x}
    return out


def f05(true, pred):
    if not true and not pred:
        return 1.0
    tp = len(true & pred)
    return 0.0 if tp == 0 else 1.25 * tp / (0.25 * len(true) + len(pred))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--cand", default=None)
    a = ap.parse_args()
    gt, pred = load(a.gt), load(a.pred)
    scores = [f05(t, pred.get(k, set())) for k, t in gt.items()]
    print(f"macro F0.5 = {sum(scores) / len(scores):.5f} over {len(scores):,} Source-1 entities")
    if a.cand:
        cand = load(a.cand)
        n_true = sum(len(t) for t in gt.values())
        hit = sum(len(t & cand.get(k, set())) for k, t in gt.items())
        print(f"blocking recall = {hit / max(n_true, 1):.4f} | avg candidates per entity = "
              f"{sum(len(v) for v in cand.values()) / max(len(cand), 1):.2f}")
