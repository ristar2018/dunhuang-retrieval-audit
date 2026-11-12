# -*- coding: utf-8 -*-
r"""
E1-1 statistics: the primary test on the cross-fitted results

statistical definitions (the manuscript must match these)
------------------------------------
- the pairing has two levels: the same image is reused across the five folds, so the **fold** is the independent unit,
  the seeds are only within-fold replicates. The primary test therefore works in folds (n = 5), averaging the three seeds inside a fold first,
  then a paired t-test (df = 4); the n = 15 test on the (fold, seed) unit is reported alongside for comparison,
  but it counts the same images more than once, which is an optimistic definition and cannot carry the main conclusion
- two-sided tests throughout, never one-sided (the 0.044/0.034/0.043/0.042 in Table 3 of the manuscript are one-sided p multiplied by four)
- multiple comparisons: one pre-declared test family, Holm correction, and both the raw and the corrected p for every test
  family = {delta(S2-S0) x 4 methods, delta(S3-S0) x 4 methods, size-matched delta(S3'-S0) x 2 methods} = 10 tests
- report micro-MAP (mean AP over all queries) together with macro-MAP (average within class, then across classes),
  used to judge whether the effect is driven by the large classes alone
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
    """Return the paired-difference statistics. a and b are equal-length arrays already aligned by pairing unit."""
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

    # fold level: average the three seeds inside a fold first
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

    # for reference: the (fold, seed) unit, n = 15
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