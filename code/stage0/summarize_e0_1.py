# -*- coding: utf-8 -*-
"""E0-1 汇总：在干净 gallery（F_seed_all, n=1941）上做配对检验，并给出污染度核算。"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).parent
df = pd.read_csv(OUT / "e0_1b_clean_gallery.csv", encoding="utf-8-sig")
METHODS = ["ArcFace", "GCE", "SCE", "SubCenterArcFace_K1", "SubCenterArcFace_K3"]


def paired(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    t, p = stats.ttest_rel(a, b)
    sd = d.std(ddof=1)
    dz = d.mean() / sd if sd > 0 else np.nan
    tc = stats.t.ppf(0.975, len(d) - 1)
    se = sd / np.sqrt(len(d))
    return d.mean(), p, dz, (d.mean() - tc * se, d.mean() + tc * se)


def get(v, m, ds):
    s = df[(df.variant == v) & (df.method == m) & (df.dataset == ds)].sort_values("seed")
    return s.MAP.values


print("=" * 100)
print("F_seed_all (专家标签 gallery n=1941)：训练层级效应 vs 论文口径")
print("=" * 100)
print(f"{'method':<24s}{'S0':>9s}{'S2':>9s}{'S3':>9s} | "
      f"{'ΔS2-S0':>9s}{'p':>8s}{'d_z':>7s} | {'ΔS3-S0':>9s}{'p':>8s}{'d_z':>7s}")
for m in METHODS:
    s0, s2, s3 = get("F_seed_all", m, "S0"), get("F_seed_all", m, "S2"), get("F_seed_all", m, "S3")
    d2, p2, z2, _ = paired(s2, s0)
    d3, p3, z3, _ = paired(s3, s0)
    print(f"{m:<24s}{s0.mean():>9.4f}{s2.mean():>9.4f}{s3.mean():>9.4f} | "
          f"{d2:>+9.4f}{p2:>8.4f}{z2:>7.2f} | {d3:>+9.4f}{p3:>8.4f}{z3:>7.2f}")

print("\n" + "=" * 100)
print("等样本量控制序列（n=1602 恒定，伪标签占比 0 → 全量重抽）：F_seed_all vs A_current")
print("=" * 100)
for v in ["F_seed_all", "A_current"]:
    print(f"\n--- {v} ---")
    print(f"{'method':<24s}" + "".join(f"{t:>16s}" for t in ["S0", "S1'", "S2'", "S3'"])
          + f"{'Δ(S3\'-S0)':>11s}{'p':>8s}{'相对降幅':>10s}")
    for m in METHODS:
        vals = [get(v, m, t) for t in ["S0", "S1_prime", "S2_prime", "S3_prime"]]
        if any(len(x) == 0 for x in vals):
            print(f"{m:<24s}  (缺 prime 控制组，未跑)")
            continue
        cells = [f"{x.mean():.4f}±{x.std(ddof=1):.4f}" for x in vals]
        d, p, z, ci = paired(vals[3], vals[0])
        rel = d / vals[0].mean() * 100
        print(f"{m:<24s}" + "".join(f"{c:>16s}" for c in cells)
              + f"{d:>+11.4f}{p:>8.4f}{rel:>9.1f}%")

print("\n" + "=" * 100)
print("零样本（raw DINOv3，无任何训练）vs 最好训练配置")
print("=" * 100)
raw = df[df.run == "RAW_DINOv3_zeroshot"].set_index("variant")
print(f"{'variant':<20s}{'n_gallery':>10s}{'raw MAP':>10s}{'raw R@1':>10s}"
      f"{'best trained':>14s}{'best MAP':>10s}")
for v in ["A_current", "B_seed_only", "F_seed_all", "G_labeled_seed_all"]:
    sub = df[(df.variant == v) & (df.run != "RAW_DINOv3_zeroshot")]
    b = sub.loc[sub.MAP.idxmax()]
    print(f"{v:<20s}{int(raw.loc[v,'n_gallery']):>10d}{raw.loc[v,'MAP']:>10.4f}"
          f"{raw.loc[v,'R@1']:>10.4f}{b['run']:>14s}{b['MAP']:>10.4f}")

print("\n" + "=" * 100)
print("污染度核算")
print("=" * 100)
n_a, n_f, n_lab = 21703, 1941, 1235
print(f"  论文口径 gallery（A）= {n_a}；其中专家标注 = 339 ({339/n_a*100:.2f}%)，"
      f"is_distractor=True = 18000 ({18000/n_a*100:.2f}%)")
print(f"  cross_source（论文称留作 OOD）= 2468 张，实际全部并入 gallery 参与打分")
print(f"  干净 gallery（F）= {n_f} 张专家标注（含训练用 seed），query = 337 张专家标注")
print(f"  论文表 1 报 query 1,076，实际送入评估的 query = 337（{337/1076*100:.1f}%）")