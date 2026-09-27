"""Central configuration for the Business Entity Resolution pipeline.

Every value here can be left as-is; the most useful ones are also exposed as CLI flags
in run_pipeline.py.
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    seed: int = 42
    n_jobs: int = max(1, (os.cpu_count() or 2) - 1)

    # ---------------- training sample ----------------
    # Training uses an entity-cluster sample (see data_io.load_train_sampled). By default its
    # fraction f = (#test S1)/(#train S1) so candidate crowding matches the test set, capped so the
    # sampled train has at most max_train_records records (RAM/time guard).
    max_train_records: int = 4_000_000
    quick_train_s1: int = 40_000          # --quick: ~this many train S1 entities

    # ---------------- blocking / candidate generation ----------------
    # Neighbours retrieved per Source-1 entity, per blocking view, per target source (S2 / S3),
    # BEFORE pruning. The final candidate set is much smaller (see block_target_keep).
    block_k_per_view: int = 10
    block_min_sim: float = 0.05          # ignore neighbours below this TF-IDF cosine
    # Max multiply-adds per sparse retrieval product. Above it, the most common n-grams are skipped
    # for RETRIEVAL only (scores still use everything). 5e10 ~ seconds to a minute on a laptop.
    block_cost_budget: float = 5e10
    # After building the union of neighbours, prune it with a (score, tau, K, R) rule tuned on train
    # so that we keep >= this fraction of the true matches the unpruned union contained, using the
    # smallest possible number of candidates per Source-1 entity.
    block_target_keep: float = 0.995
    # ...unless that needs more than this many candidates per S1 entity (dense data): then keep as
    # many matches as possible within this budget. Bounds test-time memory and candidate-set size.
    block_max_avg_cands: float = 10.0
    # "auto": block within country only if (almost) all training matches share the country label.
    block_by_country: str = "auto"
    block_taus: tuple = tuple(round(x * 0.05, 2) for x in range(0, 13))      # 0.00 .. 0.60
    block_ks: tuple = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
    block_rs: tuple = (1, 2, 3, 5, 8, 1_000_000)

    # ---------------- features ----------------
    feature_chunk: int = 50_000          # pairs per parallel job for string features

    # ---------------- model ----------------
    n_folds: int = 5
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100
    lgb_params: dict = field(default_factory=lambda: dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=40,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        verbose=-1,
    ))
