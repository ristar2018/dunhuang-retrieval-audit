# -*- coding: utf-8 -*-
r"""
E1-3 类别层面拆解：扩展训练集的收益/损失是否被大类与小类分摊得一样。

疑问来源
--------
原稿只报 micro-MAP（全体 query 的 AP 均值），大类（wall_mural 103 张 query）
会主导结果；而文化遗产检索的实际价值往往在稀有细粒度类
（landscape / dance / other 每类仅 7 张）。若扩展主要抬升大类、压低小类，
则"扩展有益"这一结论对真实用途是误导。

做法
----
用 e1_perclass.csv（逐 run × 16 类的 AP），对每个方法在每折内先按 seed 平均，
再跨折平均，得到该类在每档的 AP 与 Δ(S2−S0)；然后：
  1. 列出逐类 AP 与 Δ；
  2. 用类规模（每折平均 query 数）与 Δ 做 Spearman 相关；
  3. 给出 macro-MAP（等权）与 micro-MAP（按 query 加权）的差，
     即"大类与小类的结论是否一致"。
产物：e1_3_perclass.csv、e1_3_summary.csv
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

    # 折内 seed 平均 → 跨折平均
    per_fold = pc.groupby(["method", "tier", "fold", "label"])["AP"].mean().reset_index()
    per_class = per_fold.groupby(["method", "tier", "label"])["AP"].agg(["mean", "std"]).reset_index()

    # 类规模：用干净 query 的分层规模（每折平均）
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

    # micro vs macro 对照
    ev = pd.read_csv(OUT / "e1_eval.csv", encoding="utf-8-sig")
    ev = ev[ev.ckpt == "best"]
    fm = ev.groupby(["method", "tier", "fold"])[["MAP", "macro_MAP"]].mean().reset_index()
    print("\n=== micro-MAP vs macro-MAP（折均值；差 = 大类红利）===")
    t = fm.groupby(["method", "tier"])[["MAP", "macro_MAP"]].mean()
    t["gap"] = t["MAP"] - t["macro_MAP"]
    print(t.round(4).to_string())


if __name__ == "__main__":
    main()