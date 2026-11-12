# -*- coding: utf-8 -*-
r"""
E1-2 contamination dose-response curve (no retraining; the 129 manuscript checkpoints are reused)

the question to answer
-------------
the manuscript's "the enlarged training set helps" holds only on a gallery that is 96% filled with pseudo-labels
So: **how much same-source contamination does this conclusion need?**

design (same images, same size, only the label source changes)
----------------------------------
query is fixed: the 337 human-labelled hf_dunhuang images of the manuscript
three gallery arms, each with **the same** images and the same total size; only the labelling of the pseudo-labelled images differs:

  G_null   the clean 339 plus the same n_p pseudo-labelled images, labelled -1 so they never count as hits
           -> this is exactly what code/03_dataset_split.py:250-254 was meant to do
             (r["label"] = "distractor"), which 02_label_propagation.py later overwrote
  G_knn    the clean 339 plus the same n_p pseudo-labelled images, labelled by the propagation result
           -> this is exactly what the manuscript actually did
  G_clean  the clean 339 only (the n_p = 0 reference point)

the scored images, the features and the queries are all identical; the only difference is whether those n_p images **can count as correct answers**
so (G_knn - G_null) is the net effect of the label source,
while the n_p sweep gives the curve of that effect against the amount of same-source contamination

output: e1_2_dose.csv (one row per run x arm x n_p)
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

# choice of pool:
# all      -- the pseudo-label pool the manuscript actually used (every non-seed image in gallery and cross),
# part of them also sit in the training set, so "remembering the training images" gets scored
# disjoint -- use only pseudo-labelled images **absent from the original train.csv**, so that the label-source effect
# keeps this cleanly apart from "does the model remember its training images"
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

    # every n_p uses the same pseudo-labelled images (nested prefixes, so the curve stays comparable)
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
                    continue  # at n_p = 0 the two arms coincide, so it is recorded once
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