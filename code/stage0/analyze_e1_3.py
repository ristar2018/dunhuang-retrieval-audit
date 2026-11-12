# -*- coding: utf-8 -*-
r"""
E1-3 class-level breakdown: are the gains and losses of the enlarged training set shared evenly by large and small classes

where the doubt comes from
--------
the manuscript reports micro-MAP only (mean AP over all queries), and the large class (wall_mural, 103 queries)
dominates the result, while the practical value of cultural-heritage retrieval tends to lie in the rare fine-grained classes
(landscape / dance / other hold only seven queries each). If enlargement lifts the large classes and depresses the small ones,
then "enlargement helps" misleads anyone who cares about real use

what is done
----
Using e1_perclass.csv (AP per run x 16 classes), average the seeds within each fold for each method,
then averaged across folds, which gives that class's AP per tier and its delta(S2-S0); then:
  1. list the per-class AP and the delta;
  2. Spearman-correlate the class size (mean query count per fold) with the delta;
  3. report the gap between macro-MAP (equal weight) and micro-MAP (query-weighted),
     i.e. whether large and small classes tell the same story
output: e1_3_perclass.csv, e1_3_summary.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).parent
METHODS = ["ArcFace", "GCE", "SCE", "SubCenterArcFace_K3"]


def main():
    pc = pd.read_csv(OUT / "e1_perclass.csv", encoding="utf-8-sig")
    pc = pc[pc.ckpt == "best"]
    print(f"[data] {len(pc)} 行；类={pc.label.nunique()}；折={sorted(pc.fold.unique())}")

    # average the seeds within a fold, then across folds
    per_fold = pc.groupby(["method", "tier", "fold", "label"])["AP"].mean().reset_index()
    per_class = per_fold.groupby(["method", "tier", "label"])["AP"].agg(["mean", "std"]).reset_index()

    # class size: the stratified size of the clean query (averaged over folds)
    qsize = (pc[pc.tier == "S0"].groupby("label")["n_query"].mean().round(2)
             if "n_query" in pc.columns else None)
    if qsize is None:
        print("[warn] 无 n_query 列，类规模改用折内 query 计数推断")
    else:
        print("\n=== 每折平均 query 数（按类）===")
        print(qsize.sort_values(ascending=False).to_string())

    rows = []
    for meth in METHODS:
        for label in sorted(per_class.label.unique()):
            g = per_class[(per_class.method == meth) & (per_class.label == label)]
            d = dict(zip(g.tier, g["mean"]))
            if not {"S0", "S2"} <= set(d):
                continue
            rows.append(dict(method=meth, label=label,
                             size=float(qsize[label]) if qsize is not None else np.nan,
                             AP_S0=d["S0"], AP_S1=d.get("S1", np.nan),
                             AP_S2=d["S2"], AP_S3=d.get("S3", np.nan),
                             delta_S2_S0=d["S2"] - d["S0"],
                             delta_S3_S0=d.get("S3", np.nan) - d["S0"]))
    detail = pd.DataFrame(rows)
    detail.to_csv(OUT / "e1_3_perclass.csv", index=False, encoding="utf-8-sig")

    print("\n=== 逐类 AP 与 Δ(S2−S0)（按类规模降序）===")
    for meth in METHODS:
        sub = detail[detail.method == meth].sort_values("size", ascending=False)
        if sub.empty:
            continue
        print(f"\n--- {meth} ---")
        print(sub.assign(**{c: sub[c].round(3) for c in
                            ["size", "AP_S0", "AP_S2", "AP_S3", "delta_S2_S0"]})
              [["label", "size", "AP_S0", "AP_S2", "AP_S3", "delta_S2_S0"]]
              .to_string(index=False))

    summ = []
    for meth in METHODS:
        sub = detail[detail.method == meth].dropna(subset=["delta_S2_S0"])
        if sub.empty:
            continue
        rho, p = stats.spearmanr(sub["size"], sub["delta_S2_S0"])
        big = sub[sub["size"] >= sub["size"].median()]
        small = sub[sub["size"] < sub["size"].median()]
        summ.append(dict(
            method=meth,
            n_classes=len(sub),
            spearman_rho_size_vs_delta=round(float(rho), 3),
            spearman_p=round(float(p), 4),
            mean_delta_big_classes=round(float(big.delta_S2_S0.mean()), 4),
            mean_delta_small_classes=round(float(small.delta_S2_S0.mean()), 4),
            n_pos=int((sub.delta_S2_S0 > 0).sum()),
            n_neg=int((sub.delta_S2_S0 < 0).sum()),
        ))
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "e1_3_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== 类规模 vs Δ(S2−S0) ===")
    print(summ.to_string(index=False))

    # micro vs macro contrast
    ev = pd.read_csv(OUT / "e1_eval.csv", encoding="utf-8-sig")
    ev = ev[ev.ckpt == "best"]
    fm = ev.groupby(["method", "tier", "fold"])[["MAP", "macro_MAP"]].mean().reset_index()
    print("\n=== micro-MAP vs macro-MAP（折均值；差 = 大类红利）===")
    t = fm.groupby(["method", "tier"])[["MAP", "macro_MAP"]].mean()
    t["gap"] = t["MAP"] - t["macro_MAP"]
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()