# -*- coding: utf-8 -*-
r"""
E1-2 污染剂量—反应曲线（不重训，复用原稿 129 个 checkpoint）。

要回答的问题
-------------
原稿的「扩展训练集有益」只在一个 96% 靠伪标签填出来的 gallery 上成立。
那么：**需要多少同源污染，这个结论才会出现？**

设计（同图、同大小、只改标签来源）
----------------------------------
query 固定 = 原稿 337 张 hf_dunhuang 人工标签图。
gallery 三臂，每臂都用**同一批**图像、同样的总规模，只有伪标签图的标签处理不同：

  G_null   干净 339 + n_p 张伪标签图，标签设为 -1（永不命中）
           → 这正是 code/03_dataset_split.py:250-254 原本要做的事
             （`r["label"]="distractor"`），后来被 02_label_propagation.py 覆盖掉了。
  G_knn    干净 339 + 同一批 n_p 张伪标签图，标签取传播结果
           → 这正是原稿实际做的事。
  G_clean  只保留干净 339（n_p=0 的参照点）。

被判分的图像、特征、query 全同，唯一差别是那 n_p 张图**能不能作为正确答案**。
于是 (G_knn − G_null) 就是「标签来源」的净效应，
而 n_p 的扫描给出这个效应随同源污染量变化的曲线。

产物：e1_2_dose.csv（逐 run × 臂 × n_p）。
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from e0_1b_clean_gallery import BASE, ANNO, SPL, ProjectionHead, embed  # noqa: E402

OUT = Path(__file__).parent
RES = BASE / "results"
PAT = re.compile(r"^(?P<method>[A-Za-z]+)(?:_K(?P<k>\d+))?_(?P<ds>S[0-3])_seed(?P<seed>\d+)$")

N_P_GRID = [0, 500, 1000, 2000, 4000, 8000, 17000]
RNG_SEED = 7
METHODS = {"ArcFace", "GCE", "SCE", "SubCenterArcFace_K3"}

# 池子选择：
#   all      —— 原稿实际使用的伪标签池（gallery∪cross 中所有非 seed 图），
#               其中一部分同时出现在训练集里，因此"记住训练图"会被计分。
#   disjoint —— 只用**不在原 train.csv 里**的伪标签图，把"标签来源"效应
#               与"记不记得住训练图"彻底分开。
POOL = sys.argv[1] if len(sys.argv) > 1 else "all"
N_P_GRID = N_P_GRID if POOL == "all" else [0, 250, 500, 1000, 2000, 4000, 8000]
TAG = f"e1_2_dose_{POOL}.csv"


def metric_map(qe, ql, ge, gl):
    sim = qe @ ge.T
    order = np.argsort(-sim, axis=1)
    aps = np.empty(len(ql))
    for i in range(len(ql)):
        rel = (gl[order[i]] == ql[i]).astype(np.float64)
        if rel.sum() == 0:
            aps[i] = 0.0
            continue
        c = np.cumsum(rel)
        aps[i] = ((c / (np.arange(len(rel)) + 1.0)) * rel).sum() / rel.sum()
    r1 = float((gl[order[:, 0]] == ql).mean())
    return float(aps.mean()), r1


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat = np.load(BASE / "features" / "dinov3_features.npy")
    lab = pd.read_csv(ANNO / "all_labels.csv", encoding="utf-8")
    cat = dict(zip(lab["global_idx"], lab["label"]))
    lt = dict(zip(lab["global_idx"], lab["label_type"]))
    cats = sorted(lab["label"].dropna().unique())
    cidx = {c: i for i, c in enumerate(cats)}

    sp = {n: set(pd.read_csv(SPL / f, encoding="utf-8")["global_idx"])
          for n, f in [("train", "train.csv"), ("query", "query.csv"),
                       ("gallery", "gallery.csv"), ("cross", "cross_source_test.csv")]}

    q_idx = sorted(i for i in sp["query"] if lt.get(i) == "seed")
    qs = set(q_idx)
    clean_g = sorted(i for i in (sp["gallery"] | sp["cross"])
                     if lt.get(i) == "seed" and i not in qs)
    pseudo_pool = sorted(i for i in (sp["gallery"] | sp["cross"])
                         if lt.get(i) != "seed" and i not in qs)
    n_all = len(pseudo_pool)
    if POOL == "disjoint":
        pseudo_pool = [i for i in pseudo_pool if i not in sp["train"]]
    top_overlap = len(set(pseudo_pool) & sp["train"])
    print(f"[data] query={len(q_idx)} 干净 gallery={len(clean_g)} "
          f"伪标签池({POOL})={len(pseudo_pool)}（其中在 train 里 {top_overlap} 张）")
    print(f"[data] 原始池 {n_all} 张，与 train 重叠 "
          f"{len(set(sorted(i for i in (sp['gallery'] | sp['cross']) if lt.get(i) != 'seed' and i not in qs)) & sp['train'])} 张")
    N_P_GRID = [n for n in N_P_GRID if n <= len(pseudo_pool)]

    ql = np.array([cidx[cat[i]] for i in q_idx])
    gl_clean = np.array([cidx[cat[i]] for i in clean_g])
    qf = feat[np.array(q_idx)]

    # 每个 n_p 用同一批伪标签图（嵌套取前 n_p，保证曲线可比）
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(len(pseudo_pool))
    pool_ord = [pseudo_pool[i] for i in perm[:max(N_P_GRID)]]
    gl_pool = np.array([cidx[cat[i]] for i in pool_ord])
    arms = {}
    for n_p in N_P_GRID:
        arms[n_p] = dict(
            n=n_p,
            G_knn=np.concatenate([gl_clean, gl_pool[:n_p]]),
            G_null=np.concatenate([gl_clean, np.full(n_p, -1)]),
        )

    runs = []
    for d in sorted(RES.iterdir()):
        if not d.is_dir():
            continue
        m = PAT.match(d.name)
        if not m:
            continue
        gd = m.groupdict()
        meth = gd["method"] + (f"_K{gd['k']}" if gd.get("k") else "")
        if meth not in METHODS:
            continue
        if gd["ds"] not in ("S0", "S1", "S2", "S3"):
            continue
        runs.append(dict(dir=d, method=meth, dataset=gd["ds"], seed=int(gd["seed"])))
    print(f"[runs] {len(runs)} 个（{sorted({r['method'] for r in runs})}）")

    rows = []
    for n, r in enumerate(runs, 1):
        p = r["dir"] / "best_checkpoint.pt"
        if not p.exists():
            continue
        ck = torch.load(p, map_location=device, weights_only=False)
        model = ProjectionHead(embed_dim=512).to(device)
        model.load_state_dict(ck["model_state_dict"])
        qe = embed(model, qf, device)
        ge_clean = embed(model, feat[np.array(clean_g)], device)
        ge_pool = embed(model, feat[np.array(pool_ord)], device)
        for n_p, a in arms.items():
            ge = np.vstack([ge_clean, ge_pool[:n_p]]) if n_p else ge_clean
            for arm, gl in (("G_knn", a["G_knn"]), ("G_null", a["G_null"])):
                if n_p == 0 and arm == "G_null":
                    continue  # n_p=0 时两臂同值，只记一次
                m_ap, m_r1 = metric_map(qe, ql, ge, gl)
                rows.append(dict(run=r["dir"].name, method=r["method"],
                                 dataset=r["dataset"], seed=r["seed"], n_p=n_p,
                                 arm="G_clean" if n_p == 0 else arm,
                                 n_gallery=len(a["G_knn"]), MAP=m_ap, R1=m_r1))
        del model
        if n % 20 == 0:
            print(f"  [{n}/{len(runs)}] {r['dir'].name}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / TAG, index=False, encoding="utf-8-sig")
    print(f"\n[saved] {TAG} rows={len(df)}")

    df["rho"] = df.n_p / (df.n_p + len(clean_g))
    for arm in ("G_knn", "G_null"):
        sub = df[df.arm.isin([arm, "G_clean"])]
        pv = sub.groupby(["method", "dataset", "n_p"])["MAP"].mean().round(4).unstack()
        print(f"\n=== {arm}：MAP（行=method×dataset 列=n_p）===")
        print(pv.to_string())

    for arm in ("G_knn", "G_null"):
        piv = df[df.arm.isin([arm, "G_clean"])]
        print(f"\n=== Δ(S2−S0) 随 n_p 变化（{arm} 臂，3 seed 均值）===")
        g = piv.groupby(["method", "dataset", "n_p"])["MAP"].mean().unstack("dataset")
        for meth in g.index.get_level_values(0).unique():
            s = g.loc[meth]
            if "S2" in s.columns and "S0" in s.columns:
                print(f"  {meth:<22s} " + "  ".join(
                    f"n_p={int(n)}:{s.loc[n,'S2']-s.loc[n,'S0']:+.4f}" for n in s.index))


if __name__ == "__main__":
    main()