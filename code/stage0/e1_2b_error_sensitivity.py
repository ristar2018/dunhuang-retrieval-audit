# -*- coding: utf-8 -*-
r"""
E1-2b how sensitive the "enlargement helps" conclusion is to the pseudo-label error rate

why this exists
----
E1-2 already showed that, for one and the same gallery, counting propagated labels as correct answers makes the enlarged training set
looks beneficial (delta(S2-S0) = +0.03 to +0.07); if they are not scored, enlargement looks harmful
(-0.12 to -0.18). The two treatments are the extremes "every propagated label is right" and "every one is wrong",
The truth lies between the two, at a position set by the **actual error rate epsilon** of those labels

The script does not guess epsilon; it measures the **flip point** of the conclusion directly:
hold the gallery composition fixed (the clean 339 plus 17,000 pseudo-labelled images) and give a fraction rho of the pseudo-labelled
relabel a fraction rho of them as a random wrong class and watch delta(S2-S0) go from + to -

which yields a verdict that does not depend on any estimate of epsilon:
"the gain holds only while the error rate of the propagated gallery labels stays below rho*" --
rho* is a recomputable constant; once epsilon is supplied by a later manual audit or by external evidence, the verdict follows

the vote-margin distribution of the pseudo-label pool is output as well, as a lower-bound hint on the size of epsilon
(where the margin is near zero, the propagated label is close to a random pick among several classes)

output: e1_2b_error_sensitivity.csv, e1_2b_vote_margin.csv
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from e0_1b_clean_gallery import BASE, ANNO, SPL, ProjectionHead, embed  # noqa: E402
from e1_2_dose import metric_map  # noqa: E402

OUT = Path(__file__).parent
RES = BASE / "results"
PAT = re.compile(r"^(?P<method>[A-Za-z]+)(?:_K(?P<k>\d+))?_(?P<ds>S[0-3])_seed(?P<seed>\d+)$")
METHODS = {"ArcFace", "GCE", "SCE", "SubCenterArcFace_K3"}
N_P = 17000
RHO_GRID = [round(x, 2) for x in np.arange(0, 1.01, 0.1)]
RNG_SEED = 11


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat = np.load(BASE / "features" / "dinov3_features.npy")
    lab = pd.read_csv(ANNO / "all_labels.csv", encoding="utf-8")
    cat = dict(zip(lab["global_idx"], lab["label"]))
    lt = dict(zip(lab["global_idx"], lab["label_type"]))
    cats = sorted(lab["label"].dropna().unique())
    cidx = {c: i for i, c in enumerate(cats)}
    n_cls = len(cats)

    sp = {n: set(pd.read_csv(SPL / f, encoding="utf-8")["global_idx"])
          for n, f in [("train", "train.csv"), ("query", "query.csv"),
                       ("gallery", "gallery.csv"), ("cross", "cross_source_test.csv")]}

    q_idx = sorted(i for i in sp["query"] if lt.get(i) == "seed")
    qs = set(q_idx)
    clean_g = sorted(i for i in (sp["gallery"] | sp["cross"])
                     if lt.get(i) == "seed" and i not in qs)
    pool = sorted(i for i in (sp["gallery"] | sp["cross"])
                  if lt.get(i) != "seed" and i not in qs)
    assert not (set(pool) & sp["train"]), "伪标签池与训练集有重叠——记忆化混淆需单独处理"
    print(f"[data] query={len(q_idx)} 干净={len(clean_g)} 伪标签池={len(pool)}（与 train 重叠 0）")

    # ---- vote-margin distribution (an order-of-magnitude hint for epsilon, not an estimate of it) ----
    rows = []
    gdf = pd.read_csv(SPL / "gallery.csv", encoding="utf-8")
    cdf = pd.read_csv(SPL / "cross_source_test.csv", encoding="utf-8")
    for name, d in (("gallery", gdf), ("cross", cdf)):
        d = d[d.label_type != "seed"].copy()

        def margin(s):
            if not isinstance(s, str) or not s:
                return np.nan
            vs = []
            for part in s.split(";"):
                part = part.strip()
                if ":" not in part:
                    continue
                try:
                    vs.append(float(part.rsplit(":", 1)[1]))
                except ValueError:
                    continue
            vs.sort(reverse=True)
            if len(vs) < 2:
                return np.nan
            return vs[0] - vs[1]

        d["margin"] = d.vote_details.map(margin)
        rows.append(dict(split=name, n=len(d), margin_median=float(d.margin.median()),
                         share_lt_0p01=float((d.margin < 0.01).mean()),
                         share_lt_0p02=float((d.margin < 0.02).mean()),
                         share_lt_0p05=float((d.margin < 0.05).mean()),
                         share_lt_0p10=float((d.margin < 0.10).mean()),
                         conf_median=float(d.confidence.median())))
    vm = pd.DataFrame(rows)
    vm.to_csv(OUT / "e1_2b_vote_margin.csv", index=False, encoding="utf-8-sig")
    print("\n=== 伪标签池投票差额（top1−top2）===")
    print(vm.round(3).to_string(index=False))

    # ---- gallery composition ----
    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(len(pool))
    sel = [pool[i] for i in perm[:N_P]]
    gl_true = np.array([cidx[cat[i]] for i in sel])
    fpool = feat[np.array(sel)]
    fclean = feat[np.array(clean_g)]
    qf = feat[np.array(q_idx)]
    ql = np.array([cidx[cat[i]] for i in q_idx])

    # semantic error: draw uniformly from the other 15 classes (not "blanked", but relabelled as another class)
    wrong = (gl_true + rng.integers(1, n_cls, size=len(gl_true))) % n_cls
    ord_corrupt = rng.permutation(len(gl_true))

    runs = []
    for d in sorted(RES.iterdir()):
        if not d.is_dir():
            continue
        m = PAT.match(d.name)
        if not m:
            continue
        gd = m.groupdict()
        meth = gd["method"] + (f"_K{gd['k']}" if gd.get("k") else "")
        if meth not in METHODS or gd["ds"] not in ("S0", "S1", "S2", "S3"):
            continue
        runs.append(dict(dir=d, method=meth, dataset=gd["ds"], seed=int(gd["seed"])))
    print(f"\n[runs] {len(runs)} 个")

    out = []
    for i, r in enumerate(runs, 1):
        p = r["dir"] / "best_checkpoint.pt"
        if not p.exists():
            continue
        ck = torch.load(p, map_location=device, weights_only=False)
        model = ProjectionHead(embed_dim=512).to(device)
        model.load_state_dict(ck["model_state_dict"])
        qe = embed(model, qf, device)
        ge_c = embed(model, fclean, device)
        ge_p = embed(model, fpool, device)
        ge = np.vstack([ge_c, ge_p])
        for rho in RHO_GRID:
            k = int(round(rho * len(gl_true)))
            gl = gl_true.copy()
            if k:
                gl[ord_corrupt[:k]] = wrong[ord_corrupt[:k]]
            gl_full = np.concatenate([np.array([cidx[cat[i]] for i in clean_g]), gl])
            m_ap, _ = metric_map(qe, ql, ge, gl_full)
            out.append(dict(run=r["dir"].name, method=r["method"], dataset=r["dataset"],
                            seed=r["seed"], rho=rho, MAP=m_ap))
        del model
        if i % 20 == 0:
            print(f"  [{i}/{len(runs)}]", flush=True)

    df = pd.DataFrame(out)
    df.to_csv(OUT / "e1_2b_error_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(f"\n[saved] rows={len(df)}")

    print("\n=== Δ(S2−S0) 随伪标签错误率 ρ 变化（3 seed 均值）===")
    g = df.groupby(["method", "dataset", "rho"])["MAP"].mean().unstack("dataset")
    for meth in sorted({m for m, _ in g.index}):
        s = g.loc[meth]
        line = "  ".join(f"ρ={r:.1f}:{s.loc[r,'S2']-s.loc[r,'S0']:+.4f}" for r in RHO_GRID)
        print(f"  {meth:<22s} {line}")

    print("\n=== 各方法在 ρ=0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0 的 Δ(S2−S0) 与翻转点 ===")
    for meth in sorted({m for m, _ in g.index}):
        s = g.loc[meth]
        d = np.array([s.loc[r, "S2"] - s.loc[r, "S0"] for r in RHO_GRID])
        cross = None
        for j in range(1, len(d)):
            if d[j - 1] > 0 >= d[j]:
                cross = RHO_GRID[j - 1] + (RHO_GRID[j] - RHO_GRID[j - 1]) * d[j - 1] / (d[j - 1] - d[j])
                break
        print(f"  {meth:<22s} " + "  ".join(f"{x:+.4f}" for x in d) +
              f"   | 翻转点 ρ*={cross if cross is None else round(cross,3)}")


if __name__ == "__main__":
    main()