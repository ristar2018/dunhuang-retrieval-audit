# -*- coding: utf-8 -*-
"""E0-1 结果分析：分变体做配对检验，量化「S2 峰值」随 gallery 标注来源的变化。"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).parent
df = pd.read_csv(BASE / "e0_1_reeval.csv", encoding="utf-8-sig")
TIERS = ["S0", "S1", "S2", "S3"]
CTRL = ["S0", "S1_prime", "S2_prime", "S3_prime"]
METHODS = ["ArcFace", "GCE", "SCE", "SubCenterArcFace_K1", "SubCenterArcFace_K3"]


def paired(a, b):
    """a,b: 逐 seed 配对数组。返回 均值差、t、双侧 p、Cohen's d_z、95%CI"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    t, p = stats.ttest_rel(a, b)
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else np.nan
    se = sd / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    return d.mean(), t, p, dz, (d.mean() - tc * se, d.mean() + tc * se)


print("=" * 108)
print("表 1  MAP 随训练层级的变化（按 gallery 变体），mean±std over 3 seeds")
print("=" * 108)
for v in ["A_current", "D_L0_L3", "E_labeled", "C_nocross", "B_seed_only"]:
    sub = df[df.variant == v]
    n_g = sub.n_gallery.iloc[0]
    print(f"\n--- {v}  (gallery n={n_g}) ---")
    hdr = f"{'method':<24s}" + "".join(f"{t:>18s}" for t in TIERS) + f"{'peak':>7s}{'Δ(S2-S0)':>11s}{'p':>9s}{'d_z':>8s}"
    print(hdr)
    for m in METHODS:
        s = sub[sub.method == m]
        if s.empty:
            continue
        cells, means = [], []
        for t in TIERS:
            x = s[s.dataset == t].sort_values("seed").MAP.values
            cells.append(f"{x.mean():.4f}±{x.std(ddof=1):.4f}" if len(x) else "   —")
            means.append(x.mean())
        a = s[s.dataset == "S2"].sort_values("seed").MAP.values
        b = s[s.dataset == "S0"].sort_values("seed").MAP.values
        if len(a) == len(b) == 3:
            md, tt, pp, dz, ci = paired(a, b)
            print(f"{m:<24s}" + "".join(f"{c:>18s}" for c in cells)
                  + f"{TIERS[int(np.argmax(means))]:>7s}{md:>+11.4f}{pp:>9.4f}{dz:>8.2f}")

print("\n" + "=" * 108)
print("表 2  等样本量控制序列 S0 → S1' → S2' → S3'（样本量恒为 1602，仅伪标签占比上升）")
print("=" * 108)
for v in ["A_current", "B_seed_only", "D_L0_L3"]:
    sub = df[df.variant == v]
    print(f"\n--- {v} ---")
    print(f"{'method':<24s}" + "".join(f"{t:>18s}" for t in CTRL) + f"{'趋势':>10s}")
    for m in METHODS:
        s = sub[sub.method == m]
        if s.empty:
            continue
        cells, means = [], []
        for t in CTRL:
            x = s[s.dataset == t].sort_values("seed").MAP.values
            cells.append(f"{x.mean():.4f}±{x.std(ddof=1):.4f}")
            means.append(x.mean())
        trend = "单调↓" if all(np.diff(means) < 0) else ("单调↑" if all(np.diff(means) > 0) else "非单调")
        print(f"{m:<24s}" + "".join(f"{c:>18s}" for c in cells) + f"{trend:>10s}")

print("\n" + "=" * 108)
print("表 3  关键配对对比（逐 seed 配对，n=3，双侧）")
print("=" * 108)
pairs = [
    ("B_seed_only", "S0", "S3", "干净gallery: 全量扩展 vs 仅专家标签"),
    ("B_seed_only", "S0", "S2", "干净gallery: S2 vs S0"),
    ("B_seed_only", "S0", "S2_prime", "干净gallery: 等量含伪标签 vs 全专家标签"),
    ("A_current", "S0", "S2", "污染gallery: S2 vs S0（论文口径）"),
]
print(f"{'变体':<14s}{'对比':<32s}{'method':<24s}{'ΔMAP':>9s}{'p':>8s}{'d_z':>7s}  95%CI")
for v, t1, t2, label in pairs:
    for m in METHODS:
        s = df[(df.variant == v) & (df.method == m)]
        a = s[s.dataset == t1].sort_values("seed").MAP.values
        b = s[s.dataset == t2].sort_values("seed").MAP.values
        if len(a) != 3 or len(b) != 3:
            continue
        md, tt, pp, dz, ci = paired(a, b)
        print(f"{v:<14s}{label:<32s}{m:<24s}{md:>+9.4f}{pp:>8.4f}{dz:>7.2f}  [{ci[0]:+.4f}, {ci[1]:+.4f}]")

print("\n" + "=" * 108)
print("表 4  R@1（同口径）")
print("=" * 108)
for v in ["A_current", "B_seed_only"]:
    sub = df[df.variant == v]
    print(f"\n--- {v} ---")
    print(f"{'method':<24s}" + "".join(f"{t:>18s}" for t in TIERS))
    for m in METHODS:
        s = sub[sub.method == m]
        if s.empty:
            continue
        cells = []
        for t in TIERS:
            x = s[s.dataset == t].sort_values("seed")["R@1"].values
            cells.append(f"{x.mean():.4f}±{x.std(ddof=1):.4f}")
        print(f"{m:<24s}" + "".join(f"{c:>18s}" for c in cells))

# gallery 污染度
print("\n" + "=" * 108)
print("表 5  gallery 标注来源构成")
print("=" * 108)
for v in ["A_current", "C_nocross", "D_L0_L3", "E_labeled", "B_seed_only"]:
    sub = df[df.variant == v]
    print(f"  {v:<14s} n_gallery={sub.n_gallery.iloc[0]:>6d}  n_query={sub.n_query.iloc[0]}")