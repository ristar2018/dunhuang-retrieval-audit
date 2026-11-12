# Reproduction package

For *Answer-Key Provenance and Margin-Unit Conventions in Cultural Heritage
Image Retrieval: A Dunhuang Re-evaluation*.

The package is maintained at https://github.com/ristar2018/dunhuang-retrieval-audit.

This package contains the annotation tables, the splits, the derived features,
every run configuration, the aggregate result tables and the analysis code that
produce the numbers in the manuscript. It does **not** contain the source images
or the trained checkpoints (see *What is not shipped*).

The point of shipping the tables and the configs, rather than a prose
description of them, is that the two findings of the paper are claims about
*relations between files*: which label set defines relevance at evaluation
time, and what numeric convention the margin argument followed. Neither is
visible from a method section; both are visible from the data.

## One command

```
python checks/verify_numbers.py
```

Re-derives 62 quantities the manuscript states in prose or in a table from
the files in this package, prints `PASS`/`FAIL` per quantity, and exits 1 if any
disagree. Expected values are the manuscript's claims; the files are the
evidence; a mismatch is a defect in one of the two.

## What is not shipped

| Item | Why | Consequence |
|---|---|---|
| 32,415 source images | third-party cultural-heritage assets; redistribution is not ours to grant | identifiers, source, provenance and derived features ship; the images can be re-fetched from the sources named in `data/annotations/all_labels.csv` |
| the trained checkpoints of all 628 runs (~12 GB, dominated by the `.pt` files) | size; and they are re-derivable from the configs | every config that produced them is in `runs/RUN_MANIFEST.csv`; the aggregate tables they produced are in `results/` |
| `dinov3_features.npy`, `dinov2_features.npy` (196.9 MB) | recomputable in minutes from the images and the pinned backbone | `data/features/feature_index.json` keeps the row order and the backbone identifier |

## Layout

```
data/annotations/   all_labels.csv (32,415 rows), seed_labels.csv (1,602 rows), label_statistics.json
data/splits/        train.csv, query.csv, gallery.csv, cross_source_test.csv, split_statistics.json
data/gradients/     S0..S3_indices.json, S0_prime..S3_prime_indices.json, gradient_metadata.json
data/features/      feature_index.json (row order only)
runs/               RUN_MANIFEST.csv, one row per run, 628 rows
results/            13 aggregate CSVs; every table in the paper is generated from these
code/pipeline/      feature extraction, label propagation, deduplication, splitting, model, training, evaluation
code/audit/         dataset audit, split verification, overlap checks, bootstrap, statistics
code/stage0/        the stage-0 experiment drivers (E0 grid, E1 grid, dose response, error sweep, folds)
code/build/         manuscript and figure builders
checks/             verify_numbers.py
environment/        exact versions
```

``SHA256SUMS.txt`` covers every file in the package except itself and ``MANIFEST.json``, which are written after the digest pass; ``MANIFEST.json`` lists the same 77 files with sizes.

## Path conventions

The scripts in `code/` do not carry absolute paths. Four placeholders stand in for
the directories of the machine the runs were made on:

| Placeholder | Replaces |
|---|---|
| `<PROJECT_ROOT>` | the project root holding `08_阶段0复核/` |
| `<DATA_ROOT>` | the root of the retrieval data tree (`retrieval/`, `data/`, `runs/`) |
| `<DRIVE_ROOT>` | the working directory that held the paper sources |
| `<USER_HOME>` | the account's home directory |

Point them at your own locations before running anything; `environment/environment.md`
records the layout they are expected to resolve to. The CSVs under `data/` are shipped
byte for byte as produced, so their `image_path` column still contains the original
absolute paths. Treat those as provenance identifiers rather than runnable paths: they
are what ties a row to a source image, and they are what the SHA-256 sums certify.

## Label semantics: the one thing to read first

`data/annotations/all_labels.csv` has one row per image and these columns:

`global_idx, path, source, label, label_cn, confidence, label_type, vote_details`

`label_type` is the column that matters. It records how the label came to
exist, and the two evaluation keys of the paper differ only in which values of
it are allowed to define relevance:

- `seed` — 1,602 images labelled by a person. This is the only human evidence
  in the corpus: 339 gallery images, 337 query images, and the rest used for
  training.
- `seed_knn`, `expanded_knn` — propagated labels, adopted when the k-NN vote
  agreed with itself.
- `uncertain` — propagation declined to assign a label.

The published answer key treats every gallery image that carries a `label` as a
correct answer, including the 18,000 candidates flagged in the distractor column
of `data/splits/gallery.csv`. That is what makes 21,703 gallery candidates with
only 339 human labels, i.e. a 1.56% human share. Restricting relevance to
`label_type == "seed"` is the other key, and it is the one the paper's clean
protocol uses.

`confidence` is the vote margin of the propagation, not a calibrated
correctness estimate; `vote_details` keeps the per-class vote mass so the
margin can be recomputed.

## Splits

| file | rows | role |
|---|---:|---|
| `train.csv` | 9,036 | training pool; `S*_indices.json` select subsets of it |
| `query.csv` | 1,076 | queries (337 of them human-labelled) |
| `gallery.csv` | 19,235 | same-source candidates (339 human-labelled) |
| `cross_source_test.csv` | 2,468 | cross-source candidates; append to the gallery for the published 21,703 |

The four files are disjoint by `global_idx`; the verification script asserts it,
along with the tier arithmetic: `L0=1,602`, `L1=1,396`, `L2=4,562`, `L3=1,476`,
so `S1=2,998`, `S2=7,560`, `S3=9,036`, and `S0'..S3'` are all size-matched to
`S0` at 1,602.

## Run manifest

One row per training run, 628 rows: 129 published, 49 stage-0 grid, 450
cross-fitted grid. 627 enter the statistics — `e02_ArcFace_S0_seed42` is a
smoke run that trained for a few steps and is marked `in_statistics=no`.

| column | meaning |
|---|---|
| `run_id` | directory name; `phase` is `published`, `E0` or `E1` |
| `method` | loss head; `SubCenterArcFace` carries `subcenter_K` |
| `training_set` | `S0`..`S3`, or `S0_prime`..`S3_prime` for the size-matched controls |
| `sampling_seed` | seed of the pseudo-label sampling that built the training set; empty for `S*`, where the tier is fixed |
| `training_seed` | seed of weight init and batch order |
| `scale`, `margin_arg`, `margin_unit`, `margin_rad_effective` | the head's angular parameters. `margin_arg` is the value handed to the loss, in whatever unit the loss expects; `margin_unit` says which; `margin_rad_effective` is the same margin in radians, computed here rather than copied. `margin_unit` is `none` for GCE, SCE and the two pair-based baselines. |
| `subcenter_K` | number of sub-centers; `1` for everything that is not `SubCenterArcFace` |
| `num_training_images`, `num_update_steps` | the two denominators of any per-image or per-step comparison. `num_update_steps` is blank for published runs, which did not log step counts — see *Known gaps*. |
| `checkpoint_rule` | how the evaluated checkpoint was chosen |
| `param_source` | where the numbers came from: the published runs' heads are hard-coded in `code/pipeline/train_metric.py`, so their parameters were read out of that file, not out of `config.json` |

Read `margin_arg` together with `margin_unit`, not alone. The published runs
passed the margin in degrees to a library that converts degrees internally; the
stage-0 runs pass radians deliberately, with the unit recorded. A row with
`margin_arg = 0.5` and `margin_unit = degree` is a 0.0087 rad margin; a row with
`margin_arg = 28.6` in the same unit is 0.4992 rad. That factor of 57 is the
subject of Section 7 of the paper and it is legible here without reading code.

## Which script produces which table

| result file | produced by |
|---|---|
| `e0_1_reeval.csv` | `code/stage0/e0_1_reeval.py` |
| `e0_1b_clean_gallery.csv` | `code/stage0/e0_1b_clean_gallery.py` |
| `e0_2_eval.csv` | `code/stage0/e0_2_eval.py` |
| `e1_eval.csv` | `code/stage0/e1_eval.py` |
| `e1_2_dose.csv` | `code/stage0/e1_2_dose.py`, one file per pseudo-label pool, then merged |
| `e1_2b_error_sensitivity.csv`, `e1_2b_vote_margin.csv` | `code/stage0/e1_2b_error_sensitivity.py` |
| `e1_1_main_tests.csv`, `e1_1_fold_summary.csv` | `code/stage0/analyze_e1_1.py` |
| `e1_3_perclass.csv`, `e1_3_summary.csv` | `code/stage0/analyze_e1_3.py` |
| `e1_perclass.csv` | `code/stage0/e1_eval.py` |
| `e1_4_tests.csv` | `code/stage0/e1_4_tests.py` |

All five figures and both manuscript tables are generated from these files by
`code/build/make_figs.py` and `code/build/build_main.py`; no number in the
manuscript was transcribed by hand.

## Rebuilding from images

The images are not shipped, so this is the path for someone who has them.
Order matters: the evaluation key must be built before any label-dependent
quantity is computed, and the tiers before any split.

```
code/pipeline/01_extract_features_dinov3.py    # frozen backbone -> feature matrix
code/pipeline/02_label_propagation.py          # k-NN propagation -> all_labels.csv
code/pipeline/03b_deduplicate.py               # near-duplicate removal
code/pipeline/03c_clean_cross.py               # cross-source deduplication
code/pipeline/03d_fix_gallery_dupes.py         # gallery-side duplicates
code/pipeline/03_dataset_split.py              # train/query/gallery/cross_source
code/pipeline/build_label_gradient.py          # confidence tiers -> S*_indices.json
code/stage0/e1_make_splits.py                  # five folds over the human labels
code/stage0/e1_run_grid.py                     # 450 runs
code/stage0/e1_eval.py                         # evaluate, then the analysis scripts above
```

Training entry point is `code/stage0/train_v2.py`; the published runs used
`code/pipeline/train_metric.py`, whose head definitions are the ones discussed
in Section 7. Both select the checkpoint by minimum training loss — see
*Known gaps*.

## Known gaps

Recorded here rather than argued away.

1. **No second annotator.** The 1,602 human labels have no inter-annotator
   measurement, so no error rate is reported anywhere in the package, and the
   paper's conclusion is a statement about what the evidence cannot decide, not
   about the true error rate.
2. **Published runs did not log step counts.** `num_update_steps` is therefore
   blank for the 129 published runs, and the training-budget comparison in
   Section 6.3 uses the planned budget, not an observed one.
3. **Published configs do not record scale or margin.** `config.json` in the
   published runs contains neither. Their head parameters were read from the
   hard-coded table in `code/pipeline/train_metric.py`, which is why
   `param_source` exists as a column.
4. **No validation split.** Checkpoints are selected by minimum training loss,
   in the published runs and in the stage-0 runs alike. The paper keeps the same
   rule so that the comparison against the published checkpoints stays
   one-to-one, and says so. The sensitivity of the reported ordering to that
   rule is in the supplementary technical report, Section 8.
5. **One collection, one backbone, sixteen classes.** No claim in the paper
   extends to another collection, another backbone, or a finer taxonomy.

## Licence and contact

Code is released for verification and reuse with attribution. The label tables
and derived features are released under the same terms as the manuscript's data
availability statement. The images remain under the terms of their sources.