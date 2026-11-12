# -*- coding: utf-8 -*-
"""E0-1 前置盘点：run 目录清单 + 评估集各变体规模。只读，不改任何数据。"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BASE = Path(r"<DATA_ROOT>\retrieval")
RES = BASE / "results"
SPL = BASE / "splits"
ANNO = BASE / "annotations"

def p(*a): print(*a, flush=True)

# ---------- 1. run 目录盘点 ----------
pat = re.compile(r"^(?P<method>[A-Za-z]+)(?:_K(?P<k>\d+))?_(?P<ds>S[0-3](?:_prime|p)?)_seed(?P<seed>\d+)$")
runs = defaultdict(list)
odd = []
for d in sorted(RES.iterdir()):
    if not d.is_dir():
        continue
    m = pat.match(d.name)
    if not m:
        odd.append(d.name); continue
    has_best = (d / "best_checkpoint.pt").exists()
    runs[(m["method"], m["k"] or "-", m["ds"], m["seed"])].append(has_best)

p(f"[runs] 匹配目录 {len(runs)} 个；未匹配 {len(odd)} 个")
if odd:
    p("  未匹配示例:", odd[:8])
by_ds = Counter(k[2] for k in runs)
p("[runs] 按数据集:", dict(sorted(by_ds.items())))
p("[runs] 无 best_checkpoint 的:", [k for k, v in runs.items() if not all(v)][:10])

# ---------- 2. 标签类型分布 ----------
lab = pd.read_csv(ANNO / "all_labels.csv", encoding="utf-8")
p(f"\n[labels] all_labels.csv 行数={len(lab)} 列={list(lab.columns)}")

splits = {}
for name, fn in [("train", "train.csv"), ("query", "query.csv"),
                 ("gallery", "gallery.csv"), ("cross", "cross_source_test.csv")]:
    df = pd.read_csv(SPL / fn, encoding="utf-8")
    splits[name] = df
    cols = list(df.columns)
    lt = Counter(df["label_type"]) if "label_type" in cols else {}
    p(f"[split:{name}] n={len(df)} label_type={dict(lt) if lt else 'NO label_type 列'}")
    if name == "cross" and "label_type" not in cols:
        p("   cross_source_test 列:", cols)

# ---------- 3. 评估集变体规模 ----------
q = set(splits["query"]["global_idx"])
g = set(splits["gallery"]["global_idx"])
c = set(splits["cross"]["global_idx"])
lt_map = dict(zip(lab["global_idx"], lab["label_type"]))
# query 用 seed
q_seed = {i for i in q if lt_map.get(i) == "seed"}
p(f"\n[eval] query 总数={len(q)} → 仅 seed={len(q_seed)}")

variants = {
    "A_current(all+cross, all types)": (g | c) - q_seed,
    "B_L0only(all+cross, seed)":       {i for i in (g | c) - q_seed if lt_map.get(i) == "seed"},
    "C_nocross(all types)":            g - q_seed,
    "D_L0_L3(all+cross, no uncertain)": {i for i in (g | c) - q_seed
                                         if lt_map.get(i) in {"seed", "seed_knn", "expanded_knn"}},
    "E_L0only_nocross(gallery seed)":  {i for i in g - q_seed if lt_map.get(i) == "seed"},
}
for k, v in variants.items():
    p(f"[eval] {k:<36s} n={len(v)}")

# ---------- 4. 类别覆盖：仅 seed 的 gallery 每类多少 ----------
cats = sorted(lab["label"].dropna().unique())
lab_map = dict(zip(lab["global_idx"], lab["label"]))
for name in ("A_current(all+cross, all types)", "B_L0only(all+cross, seed)"):
    cnt = Counter(lab_map.get(i) for i in variants[name])
    p(f"\n[class] {name}  (共 {len(cats)} 类)")
    p("  " + "  ".join(f"{c}:{cnt.get(c,0)}" for c in cats))
    empty = [c for c in cats if cnt.get(c, 0) == 0]
    p(f"  零图类别: {empty if empty else '无'}")

# ---------- 5. is_distractor 分布 ----------
if "is_distractor" in splits["gallery"].columns:
    cc = Counter(splits["gallery"]["is_distractor"])
    p(f"\n[gallery] is_distractor={dict(cc)}")
    seed_d = sum(1 for _, r in splits["gallery"].iterrows()
                 if lt_map.get(r["global_idx"]) == "seed" and not r["is_distractor"])
    p(f"[gallery] seed 且非 distractor = {seed_d}")

# ---------- 6. 特征矩阵 ----------
import numpy as np
feat = np.load(BASE / "features" / "dinov3_features.npy", mmap_mode="r")
p(f"\n[features] shape={feat.shape} dtype={feat.dtype}")
mx = max(lab["global_idx"].max(), max(q), max(g), max(c))
p(f"[features] max(global_idx)={mx}  rows={feat.shape[0]}  索引越界={mx >= feat.shape[0]}")