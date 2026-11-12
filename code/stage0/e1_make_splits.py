# -*- coding: utf-8 -*-
r"""
E1-1 交叉拟合划分构造器。

背景：原划分里人工标签（label_type=='seed'）只有 2,278 张，全部来自 hf_dunhuang
一个来源；原始 train/query/gallery 分别是 1,602 / 337 / 339。评估集小到 339 张
是原稿一切问题（乃至审稿人都会质疑）的根。本脚本把这 2,278 张人工标签图做
5 折交叉拟合，每折给出全局不相交的 query / gallery / 训练种子部分，从而把
干净的留出评估从 339 张扩到 2,278 张（每折约 456 张）。

规则：
- 折叠单位＝DINOv3 特征在 cos>0.95 下的近重复连通分量（实测 2,278 张里只有
  30 个分量大于 1、共 61 张，故分组几乎不影响平衡性，但保证近重复不跨折）。
- 按类别分层：分量按大小降序，逐个放进「该类当前图数最少」的折。
- 折 k：query = 折 k，gallery = 折 (k+1)%5，训练种子 = 其余 3 折。
  这样 query 与 gallery 不相交、都不参与该折训练。
- 伪标签池 L1/L2/L3 沿用原规则，且**固定取自原 train.csv 的非 seed 行**，
  因此三层规模与原稿一致（1,396 / 4,562 / 1,476），唯一变动的是 L0。

产物：e1_splits/*.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RET = Path(r"<DATA_ROOT>\retrieval")
OUT = Path(__file__).parent / "e1_splits"
OUT.mkdir(exist_ok=True)

N_FOLD = 5
TAU = 0.95
SEED = 42

lab = pd.read_csv(RET / "annotations" / "all_labels.csv")
train_orig = pd.read_csv(RET / "splits" / "train.csv")
feat = np.load(RET / "features" / "dinov3_features.npy").astype(np.float32)

expert = lab[lab.label_type == "seed"].reset_index(drop=True)
print(f"人工标签集 {len(expert)} 张，{expert.label.nunique()} 类，来源 {expert.source.unique()}")

# ---------- 1. 近重复连通分量 ----------
X = feat[expert.global_idx.values]
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
S = X @ X.T
np.fill_diagonal(S, -1)
parent = list(range(len(X)))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


iu, ju = np.where(np.triu(S > TAU, 1))
for a, b in zip(iu, ju):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[ra] = rb
roots = np.array([find(i) for i in range(len(X))])
_, comp = np.unique(roots, return_inverse=True)
expert["comp"] = comp
print(f"近重复分量 {expert.comp.nunique()} 个（cos>{TAU}）")

# ---------- 2. 分层分折 ----------
expert["fold"] = -1
for cls, grp in expert.groupby("label", sort=False):
    counts = np.zeros(N_FOLD, dtype=int)
    comps = (grp.groupby("comp").size().sort_values(ascending=False))
    for c, _ in comps.items():
        k = int(np.argmin(counts))
        expert.loc[grp.index[grp.comp == c], "fold"] = k
        counts[k] += int((grp.comp == c).sum())
    if cls in ("wall_mural", "buddha"):
        print(f"  {cls:<16s} 各折 {counts.tolist()}")

print("\n各折规模：", expert.fold.value_counts().sort_index().tolist())
cross = pd.crosstab(expert.label, expert.fold)
print("每类每折张数（前 6 类）：")
print(cross.head(6).to_string())

# ---------- 3. 伪标签层（固定在原 train 的非 seed 行上，复刻原口径） ----------
pool = lab.merge(train_orig[["global_idx"]], on="global_idx", how="inner")
non_seed = pool[pool.label_type != "seed"]
L1 = non_seed[(non_seed.label_type == "seed_knn") & (non_seed.confidence == 1.0)]
L2 = non_seed[(non_seed.label_type == "seed_knn") & (non_seed.confidence < 1.0)]
# 实测原产物：L3 = S3-S2 新增的 1,476 张 = expanded_knn 1,369 + uncertain 107，
# 即"train 中不属于 L0/L1/L2 的全部"，而不是只有 expanded_knn。
rest = non_seed[~non_seed.global_idx.isin(pd.concat([L1, L2]).global_idx)]
L3 = rest
print(f"\nL3 构成：{L3.label_type.value_counts().to_dict()}")
print(f"\n伪标签层：L1={len(L1)} L2={len(L2)} L3={len(L3)} "
      f"（原稿 1,396 / 4,562 / 1,476）")
assert len(L1) == 1396 and len(L2) == 4562 and len(L3) == 1476, "层规模与原稿不一致"

L1i, L2i, L3i = (list(map(int, x.global_idx.values)) for x in (L1, L2, L3))

# ---------- 4. 写盘 ----------
rng = np.random.default_rng(SEED)
summary = []
for k in range(N_FOLD):
    q = expert[expert.fold == k]
    g = expert[expert.fold == (k + 1) % N_FOLD]
    tr = expert[~expert.fold.isin([k, (k + 1) % N_FOLD])]
    q_idx = list(map(int, q.global_idx.values))
    g_idx = list(map(int, g.global_idx.values))
    t0 = list(map(int, tr.global_idx.values))
    n = len(t0)

    tiers = {
        "S0": t0,
        "S1": t0 + L1i,
        "S2": t0 + L1i + L2i,
        "S3": t0 + L1i + L2i + L3i,
    }
    # 等样本量控制组：每档从对应训练集里抽 n 张（S0′ 即 S0 本身）
    for name in ("S0", "S1", "S2", "S3"):
        if name == "S0":
            tiers["S0_prime"] = list(t0)
        else:
            idx = np.array(tiers[name])
            pick = rng.choice(len(idx), size=min(n, len(idx)), replace=False)
            tiers[f"{name}_prime"] = [int(v) for v in idx[np.sort(pick)]]

    for name, idx in tiers.items():
        with open(OUT / f"train_f{k}_{name}.json", "w", encoding="utf-8") as f:
            json.dump(idx, f)
    for nm, idx in (("query", q_idx), ("gallery", g_idx)):
        with open(OUT / f"fold{k}_{nm}.json", "w", encoding="utf-8") as f:
            json.dump(idx, f)

    rec = {"fold": k, "n_query": len(q_idx), "n_gallery": len(g_idx),
           "n_train_seed": len(t0),
           **{f"n_{nm}": len(v) for nm, v in tiers.items()},
           "query_classes": int(q.label.nunique()),
           "gallery_classes": int(g.label.nunique())}
    summary.append(rec)
    print(f"\nfold{k}: query {len(q_idx)} | gallery {len(g_idx)} | train_seed {len(t0)} | "
          f"S0..S3 = {[len(tiers[f'S{i}']) for i in range(4)]} | "
          f"prime = {[len(tiers[f'S{i}_prime']) for i in range(4)]}")

# 覆盖性断言：5 折 query 合并应恰为全体人工标签
allq = sorted(sum([json.load(open(OUT / f"fold{k}_query.json")) for k in range(N_FOLD)], []))
allg = sorted(sum([json.load(open(OUT / f"fold{k}_gallery.json")) for k in range(N_FOLD)], []))
assert allq == sorted(expert.global_idx.tolist()), "query 五折未覆盖全部人工标签"
assert allg == sorted(expert.global_idx.tolist()), "gallery 五折未覆盖全部人工标签"
print(f"\n覆盖性检查通过：5 折 query 与 5 折 gallery 合并均恰为 {len(expert)} 张人工标签")

# 无跨折泄漏断言：每折 query/gallery 与 train_seed 不相交
for k in range(N_FOLD):
    q = set(json.load(open(OUT / f"fold{k}_query.json")))
    g = set(json.load(open(OUT / f"fold{k}_gallery.json")))
    t = set(json.load(open(OUT / f"train_f{k}_S0.json")))
    assert not (q & t) and not (g & t) and not (q & g), f"fold{k} 存在交集"

with open(OUT / "folds_summary.json", "w", encoding="utf-8") as f:
    json.dump({"n_fold": N_FOLD, "tau": TAU, "seed": SEED, "folds": summary},
              f, indent=2, ensure_ascii=False)
print("所有不相交断言通过，folds_summary.json 已写盘")