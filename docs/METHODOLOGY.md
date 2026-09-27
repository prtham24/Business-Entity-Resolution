# Business Entity Resolution — Methodology


## 1. Overview

We treat the task as **Source-1-anchored pairwise classification** with three stages:

1. **Normalisation** of names and addresses using country-agnostic, hand-written rules.
2. **Blocking**: sparse TF-IDF nearest-neighbour retrieval, followed by a pruning rule tuned on training labels to minimise candidates per Source-1 entity at a fixed recall.
3. **Matching**: a LightGBM classifier over 63 similarity features, followed by a decision rule chosen to directly maximise the leaderboard metric (macro F0.5 including singletons) on out-of-fold predictions.

No external data, APIs, geocoders, or pretrained models are used. The only learned model is LightGBM (MIT licence, a few MB, far below the 8B-parameter limit).

## 2. Normalisation

- Unicode NFKD accent stripping (French "Société" → "societe") and lower-casing. Dotted acronyms are collapsed ("P.V.T." → "pvt"), "&" becomes "and", "d/b/a" and "t/a" become "dba", and the Indian "M/s." prefix is dropped.
- **Names**: multi-word legal forms are mapped to one canonical token ("private limited" → "pvt ltd", "société à responsabilité limitée" → "sarl"). Common business words are abbreviated canonically (Corporation/Corp → corp, Enterprises → ent, …). Legal-suffix tokens are removed to form a **core name** and kept separately as a **legal signature**. DBA/AKA names are split into **variants**, and an **acronym** of the core name is computed.
- **Addresses**: canonical street-type and direction abbreviations (Road/Rd → rd, Boulevard/Bd → blvd, Rue, Chemin, …) and landmark words (Near/Opp → nr/opp). US and Indian state names are mapped to codes, and common old/new city spellings are unified (Bangalore/Bengaluru, Gurgaon/Gurugram). Split PIN codes are joined ("226 010" → "226010"). From each address we extract the number set, postal code (5–6 digits), first house/plot number, and the area/city tail.
- **Transliteration skeleton**: digraph folding (sh→s, ph→f, w→v, …), collapsing of repeated letters, and removal of non-initial vowels, so "Laxmi" and "Lakshmi" share the key "lksm".
- **Country** is treated as an open set of string labels. It is never one-hot encoded or hard-coded, and it is used only to restrict blocking (see below).

## 3. Candidate generation / blocking

**Stage A — retrieval (never all-pairs).** Every record's text is embedded in two sparse, L2-normalised TF-IDF views: (i) character 3-grams of the core name, and (ii) word tokens of core name plus normalised address. For each Source-1 entity we retrieve the top-10 Source-2 records and, separately, the top-10 Source-3 records per view. Retrieval uses multi-threaded sparse top-n matrix multiplication (`sparse_dot_topn`), which never materialises the full similarity matrix. The exact cost of each product is Σ_f df_A(f)·df_B(f). If it exceeds a budget (5·10¹⁰ multiply-adds), the few most common n-grams (e.g. "ent", "rd") are skipped for retrieval only; they carry little identity signal but dominate the cost. Similarity scores always use all n-grams. On training data, 100% of true matches share the country label, so search is restricted to records with the same country label. France therefore simply forms its own block at test time, and an unseen label falls back to global search.

**Stage B — pruning (this is `candidate_pairs.tsv`).** For each retrieved pair we compute both view cosines, a combined score, the **forward rank** (candidate's rank among the entity's candidates from that source), and the **reverse rank** (the entity's rank among all Source-1 entities that retrieved that candidate). Because Source 1 is deduplicated, a record that is much closer to a different Source-1 entity is rarely a true match. A rule `score ≥ τ ∧ fwd_rank ≤ K ∧ rev_rank ≤ R` is grid-searched on training labels. We pick the rule with the fewest candidates that still keeps ≥ 99.5% of the true matches reachable after Stage A. In dense data that can require very large sets. In that case, a **candidate budget** of 24 per Source-1 entity applies: we pick the rule that keeps the most true matches within the budget, with K ≤ 24, so the worst case per entity is bounded. This bounds test-time memory and keeps candidate sets small. The pairs dropped this way are the lowest-ranked, lowest-similarity ones, which a precision-oriented matcher would almost never accept.

| Blocking metric (train) | Value |
|---|---|
| Stage-A union recall | 0.9792 (744,710 / 760,567 true pairs) |
| Chosen rule | score = mean of view cosines, τ = 0, K = 20, R = 8 (keeps 99.46% of union matches) |
| Final recall ceiling | 0.9739 |
| Avg / p95 / max candidates per Source-1 entity | 19.98 / 33 / 40 |
| Test: avg / max candidates per Source-1 entity | 16.21 / 40 (28.1M pairs, from a 72.6M-pair stage-A union) |

Complexity is O(n·k) in memory and near-linear in time. It scales to billions of records by sharding on the country or blocking key and using an ANN index for Stage A.

**Training sample at scale.** The training data has ~2.2M Source-1 entities and ~10M Source-2/3 records, so we train on an entity-cluster sample. We take a random fraction *f* of Source-1 entities, all of their true matches, and the same fraction *f* of Source-2/3 records that match no entity. This is an exact *f*-scaled replica of the training distribution. We set *f* = |test S1| / |train S1| (capped by memory and time), so the number of look-alike competitors per entity during training matches the test set. That keeps rank-based blocking rules and decision thresholds valid on test. Sampling fraction used: *f* = 0.10, i.e. 220,140 Source-1 entities, 502,446 Source-2 and 526,087 Source-3 records, and 760,567 true pairs (5.7% singletons, 3.45 matches per entity on average).

## 4. Model architecture and feature engineering

**Features (63, all country-agnostic):**

- *TF-IDF cosines*: name char 2–4-grams, name word 1–2-grams, address char 3-grams, address words.
- *Name similarities (rapidfuzz)*: ratio, partial ratio, token-sort, token-set, Jaro-Winkler, token-set on the full normalised name, no-space ratio, skeleton ratio, best DBA-variant match, acronym match, first-token match, token Jaccard, containment, and length differences.
- *Legal signature*: both present, identical.
- *Address*: ratio, token-set, token-sort, partial, city/area-tail match, skeleton ratio, number-set Jaccard, postal-code equality, house-number equality, landmark flag, missing-address indicator, and length ratio. Missing values are NaN, which LightGBM handles natively.
- *Blocking context*: view cosines, forward/reverse ranks, neighbourhood sizes, and a source flag (S2/S3).
- *Group-relative features*: for key similarities, the margin between this pair and the **best competing candidate** of the same Source-1 entity, and the **best competing Source-1 entity** for the same candidate, plus "is best" flags. These are label-free and encode the one-to-many structure of the problem.

**Model:** LightGBM binary classifier (63 leaves, learning rate 0.05, feature/bagging fraction 0.8, early stopping). It is trained with **3-fold GroupKFold grouped by Source-1 entity**, so no entity's candidates leak across folds. Test probabilities are the average of the 3 fold models. Features and predictions are computed one country at a time. This is exact, because blocking never crosses countries, and it bounds peak memory.

**Decision rule, optimised for the metric.** On out-of-fold probabilities we compute the exact leaderboard metric: macro F0.5 per Source-1 entity, with singletons included and matches lost in blocking counted as misses. We then compare several rules: a global threshold, per-source thresholds, and per-entity expected-F0.5-optimal set selection (choose k ∈ {0..n} maximising expected F0.5, where k = 0 is worth P(no match) = Π(1−p)). Each rule is tried with and without an **exclusivity constraint** (a Source-2/3 record is assigned only to its most probable Source-1 entity; training labels show that 100% of matched records belong to exactly one entity). The best rule is used on test.

| Result | Value |
|---|---|
| OOF macro F0.5 (train, 3-fold) | 0.9774 (AUC 0.9999) |
| OOF macro F0.5 — US / India | 0.9872 / 0.9626 |
| Chosen decision rule | per-source thresholds (S2 ≥ 0.82, S3 ≥ 0.77) with exclusivity |
| Public leaderboard F0.5 | 0.880 |

## 5. Other relevant information

- **Unseen country (France)**: no country-specific features are used. Normalisation includes French legal forms and address words, accents are stripped, and postal-code and number features are country-neutral. 
- **Precision focus**: F0.5 and the 1.0 credit for correct singletons are optimised directly through the decision-rule search rather than through a proxy pairwise metric.
- **Fair play**: only the provided training and test files are used. Normalisation maps are generic hand-written abbreviation rules; there are no lookups, gazetteers, or geocoding.
- **Reproducibility**: `python src/run_pipeline.py --data-dir <dataset> --out-dir output --train-frac 0.1 --k-per-view 12 --max-avg-cands 24 --folds 3` regenerates both output files (we ran it on a 4-CPU / 31 GB machine in about 2 h). Seeds are fixed and versions are pinned in `requirements.txt`.
- **Compute**: CPU only, with no GPU and no pretrained models. Training uses 10% of Source-1 entities (about 220k), which is enough for LightGBM to converge.

## 6. Experiments and lessons

| Run | Train sample | Candidates / S1 (train) | Recall ceiling | OOF macro F0.5 |
|---|---|---|---|---|
| Quick run | 40k S1 (1.8%) | 8.35 | 0.9840 | 0.9862 |
| Final run | 220k S1 (10%) | 19.98 | 0.9739 | 0.9774 |

- **Crowding matters.** A thinner training sample has fewer look-alike competitors per entity, so it looks easier: a higher OOF and smaller candidate sets. The denser final sample gives a lower but more honest estimate, and pruning rules tuned on it transfer better to the crowded test set. This is why the sampling fraction is tied to the test/train size ratio.
- **Candidate budget.** Keeping 99.5% of reachable matches on the dense sample needed about 37 candidates per entity. A budget of 10 lost 7.7% of matches (recall ceiling 0.904). A budget of 24 keeps 99.46% of them with about 20 candidates, which is the trade-off we used.
- **Scale engineering.** We use a cost-capped sparse top-k search, numpy group ranking, lazy per-record normalisation, and per-country feature construction. Together these let the full 11.7M-record test set run on a 4-CPU / 31 GB machine.
- **Where errors remain.** India (0.963) is harder than the US (0.987) because of transliteration variants and landmark-style addresses, so richer transliteration features are the most promising next step.
- **Gap between validation and leaderboard (0.977 OOF vs 0.880 public).** We believe the main cause is *crowding shift*. For compute reasons the model was trained on a 10% entity-cluster sample, while the test set is about 8× more crowded with look-alike businesses. Denser neighbourhoods produce more near-duplicate candidates with high similarity, which increases false merges, the error F0.5 penalises most. The lower retrieval cost cap used at test time (1e10, needed to finish in time) also skipped more common n-grams during search. The predicted statistics (5.5% empty entities, 3.30 matches per entity, against a true 5.7% and 3.45) look well calibrated in *quantity*, which points to precision on look-alikes, not recall, as the main loss.
- **What we would do next.** Train on a sample whose crowding matches the test set (*f* ≈ 0.8, which needs more compute). Add features that directly compare a pair against its nearest competing look-alike. Tune decision thresholds under test-like density. Keep the full retrieval budget.
