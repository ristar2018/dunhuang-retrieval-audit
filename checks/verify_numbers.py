# -*- coding: utf-8 -*-
"""Re-derive the manuscript's headline numbers from the shipped artifacts.

Every check reads a file inside this package and recomputes a quantity the
manuscript states in prose or in a table. The expected values are the claims;
the files are the evidence; a mismatch is a defect in one of the two.

Run from the package root:   python checks/verify_numbers.py
Exit: 0 all pass, 1 any failure.
"""
import csv
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
DATA = os.path.join(PKG, "data")
RES = os.path.join(PKG, "results")
RUNS = os.path.join(PKG, "runs")

results = []


def check(name, got, want, tol=0.0):
    ok = abs(got - want) <= tol if isinstance(want, (int, float)) else got == want
    shown = ("%.4f vs %.4f (tol %.4f)" % (got, want, tol)
             if isinstance(want, (int, float)) else "%r vs %r" % (got, want))
    results.append((ok, name, shown))


def rows(path):
    with io.open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fnum(row, key):
    return float(row[key]) if row.get(key) not in ("", None) else float("nan")


def mean(rowset, key="MAP"):
    v = [fnum(r, key) for r in rowset]
    return sum(v) / len(v)


# ------------------------------------------------------------ confidence tiers
meta = json.load(io.open(os.path.join(DATA, "gradients", "gradient_metadata.json"),
                         encoding="utf-8"))
layers = meta["layers"]


def load_set(tier):
    j = json.load(io.open(os.path.join(DATA, "gradients", "%s_indices.json" % tier),
                          encoding="utf-8"))
    return j if isinstance(j, list) else j[tier]


check("L0 (expert-verified) size", layers["L0"], 1602)
check("L1 size", layers["L1"], 1396)
check("L2 size", layers["L2"], 4562)
check("L3 size", layers["L3"], 1476)
check("L0+L1+L2+L3 = total_train", sum(layers.values()), meta["total_train"])
sizes = {k: len(load_set(k)) for k in
         ("S0", "S1", "S2", "S3", "S0_prime", "S1_prime", "S2_prime", "S3_prime")}
check("S0 train images", sizes["S0"], 1602)
check("S1 train images", sizes["S1"], 2998)
check("S2 train images", sizes["S2"], 7560)
check("S3 train images", sizes["S3"], 9036)
check("S1 = L0+L1", sizes["S1"], layers["L0"] + layers["L1"])
check("S2 = L0+L1+L2", sizes["S2"], layers["L0"] + layers["L1"] + layers["L2"])
check("S3 = L0+L1+L2+L3", sizes["S3"], meta["total_train"])
for t in ("S0_prime", "S1_prime", "S2_prime", "S3_prime"):
    check("%s is size-matched to S0" % t, sizes[t], sizes["S0"])

# ------------------------------------------------------------------ splits
train = rows(os.path.join(DATA, "splits", "train.csv"))
query = rows(os.path.join(DATA, "splits", "query.csv"))
gallery = rows(os.path.join(DATA, "splits", "gallery.csv"))
cross = rows(os.path.join(DATA, "splits", "cross_source_test.csv"))
check("train rows", len(train), 9036)
check("query rows (all sources)", len(query), 1076)
check("gallery rows (same-source candidates)", len(gallery), 19235)
check("cross-source candidates", len(cross), 2468)
check("gallery as scored in the published key", len(gallery) + len(cross), 21703)

idx = lambda rs: set(int(r["global_idx"]) for r in rs)
t, q, g = idx(train), idx(query), idx(gallery)
check("train and query disjoint", len(t & q), 0)
check("train and gallery disjoint", len(t & g), 0)
check("query and gallery disjoint", len(q & g), 0)
check("no duplicate global_idx inside gallery", len(gallery) - len(g), 0)
check("splits do not overlap", len(t | q | g), len(t) + len(q) + len(g))

strict_query = [r for r in query if r["label_type"] == "seed"]
strict_gallery = [r for r in gallery if r["label_type"] == "seed"]
check("query images with a human label", len(strict_query), 337)
check("gallery images with a human label", len(strict_gallery), 339)
check("human share of the published answer key",
      round(len(strict_gallery) / (len(gallery) + len(cross)), 4), 0.0156, 0.0001)
check("human share of the same-source key",
      round(len(strict_gallery) / len(gallery), 4), 0.0176, 0.0001)
check("gallery candidates flagged as distractors",
      sum(1 for r in gallery if r["is_distractor"] == "True"), 18000)

# --------------------------------------------------------------- run manifest
manifest = rows(os.path.join(RUNS, "RUN_MANIFEST.csv"))
check("run manifest rows", len(manifest), 628)
by_phase = {}
for r in manifest:
    by_phase[r["phase"]] = by_phase.get(r["phase"], 0) + 1
check("published runs", by_phase.get("published", 0), 129)
check("E0 runs", by_phase.get("E0", 0), 49)
check("E1 runs", by_phase.get("E1", 0), 450)
check("runs entering the statistics",
      sum(1 for r in manifest if r["in_statistics"] == "yes"), 627)
check("rows missing num_training_images",
      sum(1 for r in manifest if r["num_training_images"] == ""), 0)
pair_loss = [r for r in manifest if r["method"] in ("CircleLoss",)]
check("blank scale only where the loss has none",
      sum(1 for r in manifest if r["scale"] == ""), len(pair_loss))

# ------------------------------------------- answer-key dependence (§5.1)
e02 = [r for r in rows(os.path.join(RES, "e0_2_eval.csv"))
       if r["ckpt"] == "best" and r["tag"] == "orig"]
for variant, method, want in (("A_current", "ArcFace", 0.0283),
                              ("B_seed_only", "ArcFace", -0.0076),
                              ("A_current", "GCE", 0.0427),
                              ("B_seed_only", "GCE", 0.0011),
                              ("A_current", "SCE", 0.0386),
                              ("B_seed_only", "SCE", -0.0010)):
    sel = [r for r in e02 if r["variant"] == variant and r["method"] == method]
    s2 = [r for r in sel if r["dataset"] == "S2"]
    s0 = [r for r in sel if r["dataset"] == "S0"]
    check("%s %s Δ(S2−S0)" % (variant, method),
          round(mean(s2) - mean(s0), 4), want, 0.0002)

# ------------------------------------------------------ dose response (§5.3)
dose = rows(os.path.join(RES, "e1_2_dose.csv"))
piv = {}
for r in dose:
    key = (r["method"], r["seed"], r["arm"], int(r["n_p"]))
    piv.setdefault(key, {})[r["dataset"]] = fnum(r, "MAP")
agg = {}
for (method, seed, arm, n_p), d in piv.items():
    if "S2" in d and "S0" in d:
        agg.setdefault((method, arm, n_p), []).append(d["S2"] - d["S0"])
dd = {k: sum(v) / len(v) for k, v in agg.items()}
check("G_knn ArcFace at 17,000 admitted images",
      round(dd[("ArcFace", "G_knn", 17000)], 4), 0.0284, 0.0002)
check("G_knn GCE at 17,000 admitted images",
      round(dd[("GCE", "G_knn", 17000)], 4), 0.0420, 0.0002)
check("G_null ArcFace at 17,000 admitted images",
      round(dd[("ArcFace", "G_null", 17000)], 4), -0.1187, 0.0002)
check("G_null GCE at 17,000 admitted images",
      round(dd[("GCE", "G_null", 17000)], 4), -0.1543, 0.0002)
check("G_null is negative from the first dose",
      dd[("ArcFace", "G_null", 500)] < 0, True)
check("the two arms differ by 0.147–0.254 at the largest dose",
      round(dd[("ArcFace", "G_knn", 17000)] - dd[("ArcFace", "G_null", 17000)], 3),
      0.147, 0.001)

# --------------------------------------------------- cross-fitted grid (§6)
e1 = rows(os.path.join(RES, "e1_eval.csv"))
check("E1 evaluation rows (450 runs × best/final)", len(e1), 900)
groups = {}
for r in e1:
    groups[(r["method"], r["tier"])] = groups.get((r["method"], r["tier"]), 0) + 1
check("method × tier groups", len(groups), 30)
check("every group holds 5 folds × 3 seeds × 2 checkpoints",
      sorted(set(groups.values())), [30])
fold_n = {}
for r in e1:
    fold_n.setdefault(r["fold"], set()).add(int(r["n_query"]))
check("query counts per fold", sorted(v for s in fold_n.values() for v in s),
      [450, 454, 454, 458, 462])

tests = rows(os.path.join(RES, "e1_1_main_tests.csv"))
check("pre-declared contrasts in the main family", len(tests), 10)
check("Holm-adjusted p-values (3 dp)",
      [round(float(r["p_holm"]), 3) for r in tests],
      [0.243, 0.665, 0.669, 0.669, 0.359, 0.038, 0.004, 0.004, 0.008, 0.004])
order = sorted(range(len(tests)), key=lambda i: float(tests[i]["p_raw"]))
seq = [float(tests[i]["p_holm"]) for i in order]
check("Holm values are monotone along the raw p-order",
      all(seq[k] <= seq[k + 1] + 1e-9 for k in range(len(seq) - 1)), True)
check("Holm never falls below the raw p-value",
      all(float(r["p_holm"]) >= float(r["p_raw"]) - 1e-12 for r in tests), True)

# --------------------------------------------------------- per-class (§6.4)
per = [r for r in rows(os.path.join(RES, "e1_3_perclass.csv"))
       if r["method"] == "ArcFace"]
check("per-class rows for ArcFace", len(per), 16)
up = sum(1 for r in per if fnum(r, "delta_S2_S0") > 0)
check("classes improving at S2 vs S0", up, 12)
check("classes declining at S2 vs S0", len(per) - up, 4)
check("AP range spans 0.057–0.986",
      (round(min(fnum(r, "AP_S0") for r in per), 3),
       round(max(fnum(r, "AP_S0") for r in per), 3)), (0.057, 0.986))

# ------------------------------------------------------------------- report
failed = [r for r in results if not r[0]]
width = max(len(r[1]) for r in results)
for ok, name, shown in results:
    print("%s  %-*s  %s" % ("PASS" if ok else "FAIL", width, name, shown))
print("\n%d checks, %d failed" % (len(results), len(failed)))
sys.exit(1 if failed else 0)