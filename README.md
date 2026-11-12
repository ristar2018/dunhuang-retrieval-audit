# Reproduction package

For *Auditing Relevance-Label Provenance and Angular-Margin Units in Dunhuang
Image Retrieval*.

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

## Data provenance and licences

The 32,415 rows resolve to five released collections through seven `source`
values. No image is redistributed in this package; only annotation tables,
derived features and code are.

| `source` | rows | collection and release | licence |
|---|---:|---|---|
| `muraldh_512`, `muraldh_sr`, `muraldh_seg` | 5,096 / 2,560 / 1,126 | **MuralDH** — Xu and Chen, China University of Mining and Technology; *Scientific Data* 11:955 (2024), doi:10.1038/s41597-024-03785-0; data at Dryad, doi:10.5061/dryad.bnzs7h4jd; code at github.com/tearsheaven/MuralDH | CC0 1.0 |
| `dh_v1` | 15,594 | **DH_Mural** — kaggle.com/datasets/mrzzzz22/dh-version1, an aggregate of MuralDH, Dunhuang_Grottoes_Painting, Dunhuang_Faces and DhMurals1714, resized or cropped to 512 x 512 | no licence declared; the release page requires the upstream licences to be respected and prohibits commercial use |
| `deepflying` | 4,561 | **DunHuang frescos** — kaggle.com/datasets/jchrysanthemum/deepflying, COCO-style annotations | CC BY-NC-SA 4.0 |
| `dunhuang_grottoes` | 1,200 | **Dunhuang Grottoes Painting Dataset and Benchmark** — kaggle.com/datasets/xuhangc/dunhuang-grottoes-painting-dataset-and-benchmark; Dunhuang Academy and Microsoft Research Asia (Yu et al., 2019, arXiv:1907.04589) | CC0 1.0 |
| `hf_dunhuang` | 2,278 | **b1ackrnoon/dunhuang** on the Hugging Face Hub — huggingface.co/datasets/b1ackrnoon/dunhuang; 2,352 images in 16 classes, the only source of human labels | no licence declared on the dataset card |

Two upstream collections are reachable only through `dh_v1`: DhMurals1714
(MuralNet, arXiv:2211.06649, doi:10.1145/3737703, released at
github.com/qinnzou/mural-image-inpainting without an explicit licence) and
Dunhuang_Faces (doi:10.18280/ts.420148), whose images originate from the Digital
Dunhuang project (e-dunhuang.com). Neither carries a `source` value of its own,
so the five collections above, not seven, are what the rows resolve to.

All use was non-commercial. Where a licence forbids commercial use or imposes
share-alike terms, the constraint applies to the images, which stay with their
providers; what ships here is the annotation table produced by this work and the
code that produced it.

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

The table is the hyperparameter record for every run. The per-run `config.json`
files are **not** shipped, because every field they carry is expanded into a
column here; `provenance_note` says which file the published heads were read
from instead. Every run is evaluated at its lowest-training-loss checkpoint; the
published runs never logged that rule, which `provenance_note` records.

| column | values | meaning |
|---|---|---|
| `run_id` | unique | run directory name |
| `phase` | `published` (129), `E0` (49), `E1` (450) | which experiment produced the row |
| `method` | 8 | loss head; `SubCenterArcFace` is the only one that carries `subcenter_K` |
| `training_set` | `S0`..`S3`, `S0_prime`..`S3_prime` | training tier, or its size-matched control |
| `training_seed` | 42, 123, 456 | weight initialisation and batch order |
| `scale` | blank, 10, 30, 64 | angular scale; blank only for `CircleLoss`, which has none |
| `margin_arg` | blank, 0.0, 0.5, 28.6, 28.6479 | the margin **as handed to the loss**, in the unit the loss expects |
| `margin_unit` | `degree` (385), `none` (243) | unit of `margin_arg`; `none` for GCE, SCE, `CircleLoss` and `ProxyNCA`, which have no angular margin |
| `margin_rad_effective` | blank, 0.0, 0.008727, 0.499164, 0.5 | the same margin in radians, computed here from the two columns above; it agrees with them in all 628 rows, so it is a convenience column, not an independent measurement |
| `subcenter_K` | blank (511), 1 (21), 3 (84), 5 (12) | sub-centre count; **blank** where the loss has no sub-centres. `1` is a real setting of `SubCenterArcFace`, not a stand-in for "no sub-centres" |
| `num_training_images` | 20 | images in the training tier |
| `num_update_steps` | blank, 9 values | optimiser steps; blank for the 129 published runs, which did not log step counts (see *Known gaps*) |
| `epochs` | 40 (625), 189 (3) | training length |
| `batch_size`, `lr`, `weight_decay`, `embed_dim`, `num_classes` | 32, 0.0005, 0.01, 512, 16 | **constant in all 628 rows**; repeated per row so the table is self-contained |
| `torch_version` | blank (129), `2.11.0+cu128` (499) | blank for the published runs, whose environment was not recorded |
| `param_source` | `config.json` (499), `code default` (129) | whether the row was read out of a config or out of the training code |
| `provenance_note` | blank, prose | what a numeric column cannot hold: the file the published heads were read from, and the default standing in for an unlogged value |
| `in_statistics` | `yes` (627), `no` (1) | the smoke run is excluded from every aggregate |

Three columns were removed when this table was normalised, and the reason is
worth keeping: `config_path` pointed at the `config.json` on the machine the runs
were made on — a path that does not exist in this package and that spelled out a
Chinese-named project directory; `sampling_seed` was empty in all 628 rows,
because the four size-matched control sets ship as explicit index lists
(`data/gradients/S0_prime..S3_prime_indices.json`) rather than being sampled; and
`checkpoint_rule` held `loss` in all 628 rows, so it is stated once above instead
of repeated per row.

Read `margin_arg` together with `margin_unit`, never alone. Every run that has a
margin hands the loss a value in degrees; the library converts it internally, so
the effective margin is the radian column. The published `ArcFace` row reads
`margin_arg = 0.5` with `margin_unit = degree`, i.e. 0.0087 rad; the stage-0 rows
set the margin they want in radians (0.5) and hand over its degree equivalent
(28.6479), i.e. 0.4992 rad. That factor of 57 is the subject of Section 7 of the
paper, and it is legible here without reading any code.


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
6. **165 of the 31,815 files are binary segmentation masks, not photographs.**
   They sit under the Kaggle DH `image_mask/{train,val}/masks/` tree and entered
   the corpus as ordinary images, so they carry propagated labels like any other
   file: 0 in `S0` and `S0'`, 22 in `S1`, 102 in `S2` and `S3`, and 9-21 in each
   size-matched `S'` control (0.6-1.4% of those pools). No mask is a query, and
   none appears in the E1-1 fold query or gallery sets, which are drawn from the
   2,278 human-labelled images. 62 masks sit in the `A_current` gallery (0.29% of
   21,703); recomputing that arm with them removed moves MAP by at most +0.0013
   and takes the S2-S0 contrast from +0.0283 to +0.0276. The pools were left as
   ingested rather than rebuilt, and the paper reports no number that depends on
   them.

## Licence and contact

Code is released for verification and reuse with attribution. The label tables
and derived features are released under the same terms as the manuscript's data
availability statement. The images remain under the terms of their sources.