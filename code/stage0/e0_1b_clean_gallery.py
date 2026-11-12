# -*- coding: utf-8 -*-
"""E0-1b：扩大干净评估集（F_seed_all）+ 原始 DINOv3 零样本基线。

动机：B_seed_only 只有 339 张 gallery，统计力弱。把「全部专家标签但非 query」的 seed
图像（1941 张，含训练用 seed）并入 gallery，可在不引入任何伪标签的前提下把干净
gallery 扩大 5.7 倍。另算不做任何训练的原始 DINOv3 余弦检索，判断训练是否真的
提升了专家标签上的检索质量。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(r"<DATA_ROOT>\retrieval")
RES, SPL, ANNO = BASE / "results", BASE / "splits", BASE / "annotations"
OUT = Path(__file__).parent


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, embed_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim), nn.BatchNorm1d(embed_dim))

    def forward(self, x):
        return self.net(x)


def embed(model, feat, device, bs=512):
    if model is not None:
        model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(feat), bs):
            b = torch.from_numpy(np.ascontiguousarray(feat[s:s + bs])).float().to(device)
            e = model(b) if model is not None else b
            out.append(F.normalize(e, dim=1).cpu().numpy())
    return np.vstack(out)


def metrics(q, ql, g, gl, ks=(1, 5, 10)):
    sim = q @ g.T
    order = np.argsort(-sim, axis=1)
    res = {}
    for k in ks:
        res[f"R@{k}"] = float((gl[order[:, :k]] == ql[:, None]).any(1).mean())
    aps = np.empty(len(ql))
    for i in range(len(ql)):
        rel = (gl[order[i]] == ql[i]).astype(np.float64)
        c = np.cumsum(rel)
        aps[i] = ((c / (np.arange(len(rel)) + 1.0)) * rel).sum() / max(rel.sum(), 1.0)
    res["MAP"] = float(aps.mean())
    return res


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat = np.load(BASE / "features" / "dinov3_features.npy")
    lab = pd.read_csv(ANNO / "all_labels.csv", encoding="utf-8")
    gdf = pd.read_csv(SPL / "gallery.csv", encoding="utf-8")
    sp = {n: set(pd.read_csv(SPL / f, encoding="utf-8")["global_idx"])
          for n, f in [("train", "train.csv"), ("query", "query.csv"),
                       ("gallery", "gallery.csv"), ("cross", "cross_source_test.csv")]}
    lt = dict(zip(lab["global_idx"], lab["label_type"]))
    cats = sorted(lab["label"].dropna().unique())
    cat = dict(zip(lab["global_idx"], lab["label"]))

    q_idx = sorted(i for i in sp["query"] if lt.get(i) == "seed")
    qs = set(q_idx)
    g_all = sp["gallery"] | sp["cross"]
    seed_all = {i for i in lab["global_idx"] if lt.get(i) == "seed"}

    variants = {
        "A_current": sorted(g_all - qs),
        "B_seed_only": sorted(i for i in g_all - qs if lt.get(i) == "seed"),
        "F_seed_all": sorted(seed_all - qs),
        "G_labeled_seed_all": sorted(
            ((set(gdf.loc[gdf.is_distractor == False, "global_idx"]) & sp["gallery"]) | seed_all) - qs),
    }
    for k, v in variants.items():
        print(f"[data] {k:<20s} n_gallery={len(v)}")
    print(f"[data] query(seed)={len(q_idx)}")

    qa = np.array(q_idx); ql = np.array([cats.index(cat[i]) for i in q_idx])
    vr = {k: (np.array(v), np.array([cats.index(cat[i]) for i in v])) for k, v in variants.items()}
    qf = feat[qa]

    rows = []

    def run_one(name, model):
        qe = embed(model, qf, device)
        for v, (garr, glab) in vr.items():
            ge = embed(model, feat[garr], device)
            m = metrics(qe, ql, ge, glab)
            rows.append({"run": name, "variant": v, "n_gallery": len(garr),
                         "MAP": m["MAP"], "R@1": m["R@1"], "R@5": m["R@5"], "R@10": m["R@10"]})

    print("\n[baseline] 原始 DINOv3 余弦（无投影头、未训练）")
    run_one("RAW_DINOv3_zeroshot", None)

    pat = re.compile(r"^(?P<m>[A-Za-z]+)(?:_K(?P<k>\d+))?_"
                     r"(?P<ds>S[0-3](?:_prime)?)_seed(?P<seed>\d+)$")
    targets = []
    for d in sorted(RES.iterdir()):
        mm = pat.match(d.name)
        if mm and (d / "best_checkpoint.pt").exists():
            key = mm["m"] + (f"_K{mm['k']}" if mm["k"] else "")
            if key.startswith("SubCenter") and key not in ("SubCenterArcFace_K1", "SubCenterArcFace_K3"):
                continue
            if key in ("CosFace", "CircleLoss", "ProxyNCA"):
                continue
            targets.append((d, key, mm["ds"], int(mm["seed"])))

    print(f"[runs] {len(targets)} checkpoints × {len(vr)} variants")
    for n, (d, key, ds, seed) in enumerate(targets, 1):
        ck = torch.load(d / "best_checkpoint.pt", map_location=device, weights_only=False)
        model = ProjectionHead(embed_dim=ck.get("config", {}).get("embed_dim", 512)).to(device)
        model.load_state_dict(ck["model_state_dict"])
        qe = embed(model, qf, device)
        for v, (garr, glab) in vr.items():
            ge = embed(model, feat[garr], device)
            m = metrics(qe, ql, ge, glab)
            rows.append({"run": d.name, "method": key, "dataset": ds, "seed": seed,
                         "variant": v, "n_gallery": len(garr),
                         "MAP": m["MAP"], "R@1": m["R@1"], "R@5": m["R@5"], "R@10": m["R@10"]})
        if n % 20 == 0:
            print(f"  [{n}/{len(targets)}] {d.name}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e0_1b_clean_gallery.csv", index=False, encoding="utf-8-sig")
    print(f"\n[saved] rows={len(df)}")

    print("\n零样本基线（无训练）:")
    print(df[df.run == "RAW_DINOv3_zeroshot"][["variant", "n_gallery", "MAP", "R@1"]].to_string(index=False))

    print("\n各变体下按训练层级 MAP（方法均值 over 3 seeds；raw 基线单列于上）:")
    d2 = df[df.run != "RAW_DINOv3_zeroshot"]
    for v in vr:
        p = d2[d2.variant == v].groupby(["method", "dataset"])["MAP"].mean().unstack()
        print(f"\n--- {v} ---")
        print(p.round(4).to_string())


if __name__ == "__main__":
    main()