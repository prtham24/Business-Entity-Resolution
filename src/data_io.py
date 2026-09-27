"""Data loading, train sampling, record preparation, output writing, macro-F0.5 metric, self-checks.

Scale notes (train has ~2M Source-1 entities and ~10M Source-2/3 records):
- Training uses an entity-cluster sample: a random fraction f of Source-1 entities, ALL of their
  true matches, and the same fraction f of Source-2/3 records that match nobody. This is an exact
  f-scaled replica of the training distribution, so candidate "crowding" (how many look-alike
  records compete) is the same as in a test set of that size. f defaults to test/train size ratio.
- Normalisation runs in parallel on unique strings. Only the fields needed for blocking are kept
  for every record; the full set of fields is computed lazily for records that survive blocking.
"""
import csv
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from normalize import country_key, process_address, process_name, skeleton

COLS = ["entity_id", "business_name", "business_address", "country"]


# --------------------------------------------------------------------------- reading
def count_lines(path: str) -> int:
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            n += block.count(b"\n")
    return n


def read_tsv(path: str) -> pd.DataFrame:
    """Read a challenge TSV as strings. Falls back to QUOTE_NONE if quote characters inside
    names made pandas merge lines (detected by comparing with the file's line count)."""
    kw = dict(sep="\t", dtype=str, keep_default_na=False, na_filter=False, encoding="utf-8")
    df = pd.read_csv(path, **kw)
    expected = count_lines(path) - 1
    if abs(len(df) - expected) > 1:
        df2 = pd.read_csv(path, quoting=csv.QUOTE_NONE, **kw)
        if abs(len(df2) - expected) < abs(len(df) - expected):
            df = df2
    df.columns = [c.strip() for c in df.columns]
    return df


def load_source(path: str, k: int) -> pd.DataFrame:
    df = read_tsv(path)
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; found {list(df.columns)}")
    df = df[COLS].copy()
    df["entity_id"] = df["entity_id"].str.strip()
    df["src"] = np.int8(k)
    return df


def load_split(data_dir: str, split: str) -> pd.DataFrame:
    """Unified record table (S1, S2, S3 stacked) with a `src` column (1/2/3)."""
    frames = [load_source(os.path.join(data_dir, split, f"{split}_source{k}.tsv"), k) for k in (1, 2, 3)]
    return pd.concat(frames, ignore_index=True)


def load_ground_truth(data_dir: str) -> dict:
    gt = read_tsv(os.path.join(data_dir, "train", "train_ground_truth.tsv"))
    out = {}
    for s1, m in zip(gt["source1_entity_id"].values, gt["matched_entity_ids"].values):
        out[s1.strip()] = [x.strip() for x in m.split(",") if x.strip()] if m else []
    return out


def load_train_sampled(data_dir: str, frac: float, seed: int, log=print):
    """Entity-cluster sample of the training data (see module docstring). Returns (R, gt)."""
    d = os.path.join(data_dir, "train")
    gt = load_ground_truth(data_dir)
    S1 = load_source(os.path.join(d, "train_source1.tsv"), 1)
    rng = np.random.default_rng(seed)
    if frac < 1:
        S1 = S1[rng.random(len(S1)) < frac].reset_index(drop=True)
    keep_s1 = set(S1["entity_id"].values)
    matched_all, matched_keep = set(), set()
    for s1, lst in gt.items():
        matched_all.update(lst)
        if s1 in keep_s1:
            matched_keep.update(lst)
    gt = {k: v for k, v in gt.items() if k in keep_s1}
    parts = [S1]
    for k in (2, 3):
        S = load_source(os.path.join(d, f"train_source{k}.tsv"), k)
        if frac < 1:
            ids = S["entity_id"]
            keep = ids.isin(matched_keep).values | (~ids.isin(matched_all).values & (rng.random(len(S)) < frac))
            S = S[keep].reset_index(drop=True)
        parts.append(S)
        log(f"  loaded train source{k}: kept {len(S):,} records")
    del matched_all
    return pd.concat(parts, ignore_index=True), gt


# --------------------------------------------------------------------------- normalisation
def _pmap(func, values, n_jobs, chunk=100_000):
    values = list(values)
    if n_jobs <= 1 or len(values) <= 2 * chunk:
        return func(values)
    parts = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(func)(values[s:s + chunk]) for s in range(0, len(values), chunk))
    return [x for p in parts for x in p]


def _name_core_chunk(vals):
    return [process_name(v)[1] for v in vals]


def _addr_norm_chunk(vals):
    return [process_address(v)[0] for v in vals]


def _name_full_chunk(vals):
    out = []
    for v in vals:
        norm, core, var, legal, acr = process_name(v)
        out.append((norm, core, var, legal, acr, skeleton(core)))
    return out


def _addr_full_chunk(vals):
    out = []
    for v in vals:
        norm, nums, postal, first, tail, lm = process_address(v)
        out.append((norm, nums, postal, first, tail, lm, skeleton(norm)))
    return out


def _mapped(values, func, n_jobs):
    codes, uniq = pd.factorize(values)
    res = _pmap(func, uniq, n_jobs)
    return codes, res


def prepare_records(R: pd.DataFrame, n_jobs: int = 1, log=print) -> pd.DataFrame:
    """Blocking fields for every record: name_core, addr_norm, country_key."""
    codes, res = _mapped(R["business_name"].values, _name_core_chunk, n_jobs)
    R["name_core"] = np.asarray(res, dtype=object)[codes]
    codes, res = _mapped(R["business_address"].values, _addr_norm_chunk, n_jobs)
    R["addr_norm"] = np.asarray(res, dtype=object)[codes]
    codes, res = _mapped(R["country"].values, lambda v: [country_key(x) for x in v], 1)
    R["country_key"] = np.asarray(res, dtype=object)[codes]
    log(f"  prepared {len(R):,} records; countries: "
        f"{R.groupby('src')['country_key'].value_counts().to_dict()}")
    return R


NAME_DETAIL = ["name_norm", "name_core", "name_var", "name_legal", "name_acr", "name_skel"]
ADDR_DETAIL = ["addr_norm", "addr_nums", "addr_postal", "addr_first", "addr_tail", "addr_landmark", "addr_skel"]


def prepare_details(sub: pd.DataFrame, n_jobs: int = 1) -> pd.DataFrame:
    """All normalised fields, for the (few) records that survived blocking."""
    D = pd.DataFrame(index=np.arange(len(sub)))
    codes, res = _mapped(sub["business_name"].values, _name_full_chunk, n_jobs)
    arr = np.empty((len(res), len(NAME_DETAIL)), dtype=object)
    for r, t in enumerate(res):
        arr[r] = t
    for k, c in enumerate(NAME_DETAIL):
        D[c] = arr[codes, k] if len(res) else []
    codes, res = _mapped(sub["business_address"].values, _addr_full_chunk, n_jobs)
    arr = np.empty((len(res), len(ADDR_DETAIL)), dtype=object)
    for r, t in enumerate(res):
        arr[r] = t
    for k, c in enumerate(ADDR_DETAIL):
        D[c] = arr[codes, k] if len(res) else []
    D["addr_landmark"] = D["addr_landmark"].astype(np.int8)
    return D


# --------------------------------------------------------------------------- metric
def macro_f05_from_arrays(s1_code, pred_mask, y, ntrue, n_s1) -> float:
    """Macro F0.5 over ALL Source-1 entities.
    Per entity: F0.5 = 1.25*TP / (0.25*n_true + n_pred); empty/empty = 1; else 0 when TP = 0.
    ntrue comes from the full ground truth, so matches lost in blocking count as misses."""
    tp = np.bincount(s1_code[pred_mask], weights=y[pred_mask], minlength=n_s1)
    npred = np.bincount(s1_code[pred_mask], minlength=n_s1)
    f = np.zeros(n_s1, dtype=np.float64)
    f[(ntrue == 0) & (npred == 0)] = 1.0
    pos = tp > 0
    f[pos] = 1.25 * tp[pos] / (0.25 * ntrue[pos] + npred[pos])
    return float(f.mean())


# --------------------------------------------------------------------------- outputs
def write_id_lists(path, s1_ids_in_order, pair_s1, pair_c, header2):
    """One row per S1 entity (in file order); comma-separated IDs; empty when none."""
    groups = defaultdict(list)
    seen = set()
    for a, b in zip(pair_s1, pair_c):
        if (a, b) not in seen:
            seen.add((a, b))
            groups[a].append(b)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header2}\n")
        for s1 in s1_ids_in_order:
            f.write(f"{s1}\t{','.join(groups.get(s1, []))}\n")


def self_validate(matching_path, candidate_path, R_test) -> list:
    """Mirror of the official rules; returns a list of problems (empty == OK)."""
    problems = []
    s1_ids = R_test.loc[R_test["src"] == 1, "entity_id"].tolist()
    valid = set(R_test.loc[R_test["src"] != 1, "entity_id"])
    parsed = {}
    for path, col in ((matching_path, "matched_entity_ids"), (candidate_path, "candidate_entity_ids")):
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False)
        if list(df.columns) != ["source1_entity_id", col]:
            problems.append(f"{path}: bad header {list(df.columns)}")
        if df["source1_entity_id"].duplicated().any():
            problems.append(f"{path}: duplicate source1_entity_id rows")
        if set(df["source1_entity_id"]) != set(s1_ids):
            problems.append(f"{path}: S1 ids do not match the test set exactly")
        d = {}
        for s1, lst in zip(df["source1_entity_id"], df[col]):
            ids = [x for x in lst.split(",") if x] if lst else []
            if len(ids) != len(set(ids)):
                problems.append(f"{path}: duplicate ids in list of {s1}")
            bad = [x for x in ids if x not in valid]
            if bad:
                problems.append(f"{path}: {s1} has invalid/non-S2/S3 ids e.g. {bad[:3]}")
            d[s1] = set(ids)
        parsed[col] = d
    m, c = parsed["matched_entity_ids"], parsed["candidate_entity_ids"]
    not_subset = [s for s in m if not m[s] <= c.get(s, set())]
    if not_subset:
        problems.append(f"{len(not_subset)} S1 rows have matches that are not in their candidates")
    return problems[:20]
