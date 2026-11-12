# -*- coding: utf-8 -*-
r"""
E1-1 统计分析：交叉拟合结果的主检验。

统计口径（写进稿件时必须与这里一致）
------------------------------------
- 配对单位有两层：同一图像会被 5 折重复使用，因此**折**是独立单位，
  seed 只是折内重复。主检验因此以折为单位（n=5），把折内 3 个 seed 先平均，
  再做配对 t（df=4）；同时报告以 (fold, seed) 为单位的 n=15 检验以便读者对照，
  但它把同一批图像重复计入，属于偏乐观口径，不能作为主结论依据。
- 双侧检验，不做单侧（原稿表 3 的 0.044/0.034/0.043/0.042 就是单侧 p×4 的产物）。
- 多重比较：预先声明一个检验族，Holm 校正，逐条报告未校正与校正后 p。
  族 = {Δ(S2−S0) × 4 方法, Δ(S3−S0) × 4 方法, 等样本量 Δ(S3′−S0) × 2 方法} = 10 项。
- 同时报告 micro-MAP（全体 query 的 AP 均值）与 macro-MAP（先按类平均再平均），
  用于判断效应是否只由大类驱动。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).parent
METHODS = ["ArcFace", "GCE", "SCE", "SubCenterArcFace_K3"]
TIERS = ["S0", "S1", "S2", "S3"]


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        run = max(run, val)
        adj[i] = min(run, 1.0)
    return adj


def paired(a, b, label, unit):
    """返回配对差异统计。a/b 为等长数组（已按配对单位对齐）。"""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    m = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    t = m / se if se > 0 else np.nan
    p = 2 * stats.t.sf(abs(t), n - 1) if se > 0 else np.nan
    tc = stats.t.ppf(0.975, n - 1)
    dz = m / sd if sd > 0 else np.nan
    return dict(contrast=label, unit=unit, n=n, mean_delta=m,
                ci_lo=m - tc * se, ci_hi=m + tc * se,
                t=t, p_raw=p, dz=dz)


def main():
    df = pd.read_csv(OUT / "e1_eval.csv", encoding="utf-8-sig")
    df = df[df.ckpt == "best"]
    print(f"[data] {len(df)} 行；fold={sorted(df.fold.unique())} "
          f"tier={sorted(df.tier.unique())} seed={sorted(df.seed.unique())}")

    # 折级：折内 3 seed 先平均
    fold_mean = (df.groupby(["method", "tier", "fold"])[["MAP", "macro_MAP"]]
                   .mean().reset_index())

    def series(method, tier, col="MAP"):
        s = fold_mean[(fold_mean.method == method) & (fold_mean.tier == tier)]
        return s.sort_values("fold")[col].values

    rows = []
    for meth in METHODS:
        for hi in ("S2", "S3"):
            rows.append(paired(series(meth, hi), series(meth, "S0"),
                               f"{meth}: Δ({hi}−S0)", "fold"))
    for meth in ("ArcFace", "GCE"):
        rows.append(paired(series(meth, "S3_prime"), series(meth, "S0_prime"),
                           f"{meth}: Δ(S3′−S0′) 等样本量", "fold"))
    fam = pd.DataFrame(rows)
    fam["p_holm"] = holm(fam.p_raw.values)
    fam["sig_holm"] = fam.p_holm < 0.05

    print("\n=== 主检验（折为单位，n=5，双侧，Holm 校正，族=10）===")
    print(fam.assign(**{
        "mean_delta": fam.mean_delta.round(4), "ci_lo": fam.ci_lo.round(4),
        "ci_hi": fam.ci_hi.round(4), "dz": fam.dz.round(2),
        "p_raw": fam.p_raw.round(5), "p_holm": fam.p_holm.round(5)})
        [["contrast", "n", "mean_delta", "ci_lo", "ci_hi", "dz", "p_raw", "p_holm", "sig_holm"]]
        .to_string(index=False))

    # 参考口径：(fold, seed) 为单位 n=15
    print("\n=== 参考口径（(fold,seed) 为单位，n=15，未校正；偏乐观，不作主结论）===")
    ref = []
    for meth in METHODS:
        for hi in ("S2", "S3"):
            a = df[(df.method == meth) & (df.tier == hi)].sort_values(["fold", "seed"]).MAP.values
            b = df[(df.method == meth) & (df.tier == "S0")].sort_values(["fold", "seed"]).MAP.values
            ref.append(paired(a, b, f"{meth}: Δ({hi}−S0)", "fold_seed"))
    ref = pd.DataFrame(ref)
    print(ref.assign(mean_delta=ref.mean_delta.round(4), ci_lo=ref.ci_lo.round(4),
                     ci_hi=ref.ci_hi.round(4), dz=ref.dz.round(2),
                     p_raw=ref.p_raw.round(5))[
        ["contrast", "n", "mean_delta", "ci_lo", "ci_hi", "dz", "p_raw"]].to_string(index=False))

    print("\n=== 各折各档 MAP（方法 × tier，5 折均值 ± 折间标准差）===")
    tab = fold_mean.groupby(["method", "tier"])["MAP"].agg(["mean", "std"]).round(4)
    print(tab.unstack().to_string())

    print("\n=== macro-MAP 对照（折均值）===")
    tab2 = fold_mean.groupby(["method", "tier"])["macro_MAP"].mean().round(4)
    print(tab2.unstack().to_string())

    fam.to_csv(OUT / "e1_1_main_tests.csv", index=False, encoding="utf-8-sig")
    fold_mean.to_csv(OUT / "e1_1_fold_summary.csv", index=False, encoding="utf-8-sig")
    print(f"\n[saved] e1_1_main_tests.csv / e1_1_fold_summary.csv")


if __name__ == "__main__":
    main()