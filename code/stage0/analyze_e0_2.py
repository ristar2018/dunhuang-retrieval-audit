# -*- coding: utf-8 -*-
"""E0-2 / E0-3 / E0-4 analysis: paired tests and the sensitivity of the checkpoint-selection rule."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).parent
df = pd.read_csv(OUT / "e0_2_eval.csv", encoding="utf-8-sig")
B, F = "B_seed_only", "F_seed_all"


def vec(variant, tag, method, ds, ck="best"):
    s = df[(df.variant == variant) & (df.tag == tag) & (df.method == method)
           & (df.dataset == ds) & (df.ckpt == ck)].sort_values("seed")
    return s.MAP.values


def comp(variant, a, b, label):
    ta, ma, da = a
    tb, mb, db = b
    x, y = vec(variant, ta, ma, da), vec(variant, tb, mb, db)
    if len(x) == 0 or len(y) == 0 or len(x) != len(y):
        print(f"  {label:<52s} 数据不足 ({len(x)},{len(y)})")
        return
    d = x - y
    if np.allclose(d, 0):
        print(f"  {label:<52s} Δ=0.0000（逐位相同）")
        return
    t, p = stats.ttest_rel(x, y)
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else np.nan
    print(f"  {label:<52s} {y.mean():.4f} → {x.mean():.4f}  Δ={d.mean():+.4f}  "
          f"p={p:.4f}  d_z={dz:+.2f}")


print("=" * 110)
print("E0-2｜修正 margin 单位、解耦 scale 与子中心（S0，3 seeds，F_seed_all / B_seed_only）")
print("=" * 110)
for v in (F, B):
    print(f"\n[{v}]")
    comp(v, ("e02a", "ArcFace", "S0"), ("orig", "ArcFace", "S0"),
         "原稿 ArcFace(s30,m0.5°) → 修正 ArcFace(s30,m28.6°)")
    comp(v, ("e02b", "ArcFace", "S0"), ("e02a", "ArcFace", "S0"),
         "scale 30 → 64（margin 固定 28.6°）")
    comp(v, ("e02c", "SubCenterArcFace_K1", "S0"), ("e02a", "ArcFace", "S0"),
         "SC-K1(s30,m28.6°) vs ArcFace(s30,m28.6°)")
    comp(v, ("orig", "SubCenterArcFace_K1", "S0"), ("e02b", "ArcFace", "S0"),
         "原稿 SC-K1(s64,m28.6°) vs 新训 ArcFace(s64,m28.6°)")

print("\n" + "=" * 110)
print("子中心数 K 的干净对照（三档同为 s=64、m=28.6°，属原稿内部干净比较）")
print("=" * 110)
for v in (F, B):
    print(f"\n[{v}]")
    comp(v, ("orig", "SubCenterArcFace_K1", "S0"), ("orig", "SubCenterArcFace_K3", "S0"), "S0: K=1 vs K=3")
    comp(v, ("orig", "SubCenterArcFace_K3", "S0"), ("orig", "SubCenterArcFace_K5", "S0"), "S0: K=3 vs K=5")
    comp(v, ("orig", "SubCenterArcFace_K1", "S2"), ("orig", "SubCenterArcFace_K3", "S2"), "S2: K=1 vs K=3")

print("\n" + "=" * 110)
print("统一超参后的损失函数比较（S0，干净标签）")
print("=" * 110)
for v in (F, B):
    print(f"\n[{v}]")
    comp(v, ("e02a", "ArcFace", "S0"), ("orig", "GCE", "S0"), "ArcFace(修正) vs GCE")
    comp(v, ("e02a", "ArcFace", "S0"), ("orig", "SCE", "S0"), "ArcFace(修正) vs SCE")
    comp(v, ("e02a", "ArcFace", "S0"), ("orig", "SubCenterArcFace_K3", "S0"),
         "ArcFace(修正) vs SC-K3(同 margin)")

print("\n" + "=" * 110)
print("E0-3｜固定优化步数预算（各数据集均 2,000 步；S0 = 40 epoch×50 步）")
print("=" * 110)
for v in (F, B):
    print(f"\n[{v}]")
    for ds in ["S1", "S2", "S3"]:
        comp(v, ("e03", "ArcFace", ds), ("e02b", "ArcFace", "S0"),
             f"ArcFace(s64,m28.6°): S0(2,000步) → {ds}(2,000步)")
    print()
    for ds in ["S1", "S2", "S3"]:
        comp(v, ("e03", "GCE", ds), ("orig", "GCE", "S0"),
             f"GCE: S0(2,000步) → {ds}(2,000步)")

print("\n" + "=" * 110)
print("反向预算检查：S0 用 9,440 步（= S2 原稿步数）是否有额外收益")
print("=" * 110)
for v in (F, B):
    comp(v, ("e03l", "ArcFace", "S0"), ("e02b", "ArcFace", "S0"),
         f"[{v}] S0: 2,000 步 → 9,440 步")

print("\n" + "=" * 110)
print("E0-4｜选模规则敏感性（F_seed_all，方法—数据集组合的 best / final / 各 epoch）")
print("=" * 110)
sub = df[(df.variant == F) & (df.tag.isin(["orig"]))]
piv = sub.pivot_table(index=["method", "dataset"], columns="ckpt", values="MAP", aggfunc="mean")
cols = [c for c in ["best", "final", "ep10", "ep20", "ep30", "ep40"] if c in piv.columns]
print(piv[cols].round(4).to_string())
print("\n各选模规则下「S2 是否仍为峰值」：")
for c in cols:
    s = piv[c]
    wid = s.unstack()
    if "S2" not in wid.columns:
        continue
    winners = wid.idxmax(axis=1)
    print(f"  {c:<6s} 峰值出现在 S2 的组合数 = {(winners == 'S2').sum()}/{len(winners)}")