# -*- coding: utf-8 -*-
"""E1-4：补跑基线的配对检验（独立于 E1-1 预声明族的次要族）。

族 = 4 项：NormSoftmax 与 CosFace 各自的 Δ(S2−S0) 与 Δ(S3−S0)。
单位 = 折（折内 3 seed 先平均，n=5，df=4），双侧，Holm 校正。
"""
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
CSV = HERE / "e1_eval.csv"

ROWS = list(csv.DictReader(open(CSV, encoding="utf-8")))
for r in ROWS:
    r["MAP"] = float(r["MAP"])
    r["fold"] = int(r["fold"])
    r["seed"] = int(r["seed"])

METHODS = ["NormSoftmax", "CosFace", "ArcFace", "GCE", "SCE", "SubCenterArcFace_K3"]
d = defaultdict(list)
for r in ROWS:
    if r["ckpt"] != "best":
        continue
    if r["method"] not in METHODS:
        continue
    d[(r["method"], r["tier"])].append((r["fold"], r["MAP"]))


def series(method, tier):
    """按 fold 排序的折均值（折内 seed 先平均）。"""
    d2 = defaultdict(list)
    for f, m in d[(method, tier)]:
        d2[f].append(m)
    folds = sorted(d2)
    return np.array([np.mean(d2[f]) for f in folds]), folds


def paired(method, t1, t2):
    a, fa = series(method, t1)
    b, fb = series(method, t2)
    assert fa == fb, (method, t1, t2, fa, fb)
    diff = a - b
    n = len(diff)
    t, p = stats.ttest_rel(a, b)
    sd = diff.std(ddof=1)
    dz = diff.mean() / sd
    se = sd / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    return diff.mean(), diff.mean() - tc * se, diff.mean() + tc * se, dz, p


print("=== E1-4 次要族（4 项，折为独立单位 n=5，双侧，Holm）===")
items = []
for m in ["NormSoftmax", "CosFace"]:
    for t in ["S2", "S3"]:
        md, lo, hi, dz, p = paired(m, t, "S0")
        items.append((f"{m}: Δ({t}−S0)", md, lo, hi, dz, p))

order = sorted(range(len(items)), key=lambda i: items[i][5])
holm = {}
run = 0.0
for rank, i in enumerate(order):
    adj = items[i][5] * (len(items) - rank)
    run = max(run, adj)
    holm[i] = min(run, 1.0)

out = []
for i, (name, md, lo, hi, dz, p) in enumerate(items):
    sig = holm[i] < 0.05
    out.append(dict(contrast=name, n=5, mean_delta=round(md, 4),
                    ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                    dz=round(dz, 2), p_raw=round(p, 5),
                    p_holm=round(holm[i], 5), sig_holm=bool(sig)))
    print(f"{name:26s} n=5 Δ={md:+.4f} CI[{lo:+.4f},{hi:+.4f}] dz={dz:+.2f} "
          f"p_raw={p:.5f} p_holm={holm[i]:.5f} {'*' if sig else 'ns'}")

with open(HERE / "e1_4_tests.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print("\n[saved] e1_4_tests.csv")

print("\n=== 干净协议 S0 档的方法排序（micro-MAP，5 折×3 seed）===")
for m in sorted(METHODS, key=lambda x: -series(x, "S0")[0].mean()):
    a, _ = series(m, "S0")
    s2, _ = series(m, "S2")
    s3, _ = series(m, "S3")
    print(f"  {m:22s} S0={a.mean():.4f} S2={s2.mean():.4f} S3={s3.mean():.4f}")