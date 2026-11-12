# -*- coding: utf-8 -*-
"""E1-0 探针：专家标签集（hf_dunhuang, 2,278 张）的近重复结构，
用于决定交叉拟合的分组粒度与净化阈值。"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RET = Path(r"<DATA_ROOT>\retrieval")
OUT = Path(__file__).parent

feat = np.load(RET / "features" / "dinov3_features.npy").astype(np.float32)
lab = pd.read_csv(RET / "annotations" / "all_labels.csv")
# 实测：features 行号 == labels 行号 == global_idx（32,415 条 path 逐条一致）
print("features", feat.shape, "| labels", lab.shape)
assert (lab.global_idx.values == np.arange(len(lab))).all()
lab["row"] = lab.global_idx.values
expert = lab[lab.label_type == "seed"].reset_index(drop=True)
X = feat[expert.row.values]
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
print("专家集", X.shape)

S = X @ X.T
np.fill_diagonal(S, -1)
for tau in [0.99, 0.98, 0.95, 0.92, 0.90]:
    n_pairs = int((S > tau).sum() // 2)
    print(f"tau={tau:.2f}: 近重复对 {n_pairs}")

# 连通分量（并查集），tau=0.95
tau = 0.95
parent = list(range(len(X)))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


iu, ju = np.where(np.triu(S > tau, 1))
for a, b in zip(iu, ju):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb
roots = np.array([find(i) for i in range(len(X))])
_, comp = np.unique(roots, return_inverse=True)
sizes = pd.Series(comp).value_counts()
print(f"\ntau=0.95 连通分量: {len(sizes)} 个，最大 {sizes.max()}，"
      f">1 的分量 {int((sizes > 1).sum())} 个，覆盖 {int(sizes[sizes > 1].sum())} 张")

# 跨 source 相似度：专家(2,278) vs 伪标签池(muraldh)
pool = lab[lab.label_type != "seed"]
Xp = feat[pool.row.values]
Xp = Xp / (np.linalg.norm(Xp, axis=1, keepdims=True) + 1e-12)
mx = np.zeros(len(X))
for i in range(0, len(X), 512):
    mx[i:i + 512] = (X[i:i + 512] @ Xp.T).max(axis=1)
print(f"\n专家→伪标签池 最大余弦：中位 {np.median(mx):.4f}  p95 {np.percentile(mx, 95):.4f}  max {mx.max():.4f}")
print(f"超过 0.95 的专家图：{(mx > 0.95).sum()} / {len(mx)}")

# 专家集内部：类内 vs 类间相似度
same = expert.label.values[:, None] == expert.label.values[None, :]
tri = np.triu(np.ones_like(S, dtype=bool), 1)
print(f"\n类内相似度 中位 {np.median(S[tri & same]):.4f}  类间 中位 {np.median(S[tri & ~same]):.4f}")