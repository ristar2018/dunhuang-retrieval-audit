# -*- coding: utf-8 -*-
"""E0-1: recompute MAP/R@K from the existing best_checkpoint.pt under several definitions of the evaluation set.

Purpose: decide whether the S2 peak is a real gain from training-set size or noise, or an artefact of the evaluation ground truth sharing its source with the training pseudo-labels
causes a measurement bias. Nothing is retrained, no raw data is written, and only CSV/JSON is produced

evaluation-set variants (query is always label_type == 'seed', as in eval_retrieval.py at run time):
  A_current   gallery U cross, every label_type            <- the definition the paper uses; must reproduce the paper's numbers
  B_seed_only gallery U cross, seed labels only (expert)           <- clean ground truth
  C_nocross   gallery, every label_type (cross excluded)
  D_L0_L3     gallery ∪ cross，label_type ∈ {seed,seed_knn,expanded_knn}
  E_labeled   gallery, is_distractor == False, every label_type (1,235 images)
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(r"<DATA_ROOT>\retrieval")
RES = BASE / "results"
SPL = BASE / "splits"
ANNO = BASE / "annotations"


class ProjectionHead(nn.Module):
    """Word-for-word the same as retrieval/code/train_metric.py:50-64"""
    def __init__(self, input_dim=768, hidden_dim=512, embed_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )

    def forward(self, x):
        return self.net(x)


def extract_embeddings(model, features, device, batch_size=512):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(features), batch_size):
            b = torch.from_numpy(np.ascontiguousarray(features[s:s + batch_size])).float().to(device)
            out.append(F.normalize(model(b), dim=1).cpu().numpy())
    return np.vstack(out)


def retrieval_metrics(q_emb, q_labels, g_emb, g_labels, ks=(1, 5, 10)):
    """Same convention as retrieval/code/eval_retrieval.py:60-141 (AP accumulated over the full ranking)"""
    sim = q_emb @ g_emb.T
    order = np.argsort(-sim, axis=1)
    Nq = len(q_labels)
    res = {}
    for k in ks:
        topk = g_labels[order[:, :k]]
        res[f"Recall@{k}"] = float((topk == q_labels[:, None]).any(axis=1).mean())
    aps = np.empty(Nq, dtype=np.float64)
    for i in range(Nq):
        rel = (g_labels[order[i]] == q_labels[i]).astype(np.float64)
        c = np.cumsum(rel)
        prec = c / (np.arange(len(rel)) + 1.0)
        aps[i] = (prec * rel).sum() / max(rel.sum(), 1.0)
    res["MAP"] = float(aps.mean())
    return res


def build_variants(splits, lab, g_df):
    q = set(splits["query"]); g = set(splits["gallery"]); c = set(splits["cross"])
    lt = dict(zip(lab["global_idx"], lab["label_type"]))
    q_seed = sorted(i for i in q if lt.get(i) == "seed")
    variants = {}
    variants["A_current"] = sorted((g | c) - set(q_seed))
    variants["B_seed_only"] = sorted(i for i in (g | c) - set(q_seed) if lt.get(i) == "seed")
    variants["C_nocross"] = sorted(g - set(q_seed))
    variants["D_L0_L3"] = sorted(i for i in (g | c) - set(q_seed)
                                 if lt.get(i) in {"seed", "seed_knn", "expanded_knn"})
    non_dist = set(g_df.loc[g_df["is_distractor"] == False, "global_idx"])
    variants["E_labeled"] = sorted((non_dist & g) - set(q_seed))
    return q_seed, variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "e0_1_reeval.csv"))
    ap.add_argument("--variants", default="A_current,B_seed_only,D_L0_L3,E_labeled,C_nocross")
    ap.add_argument("--methods", default="ArcFace,GCE,SCE,SubCenterArcFace_K1,SubCenterArcFace_K3")
    ap.add_argument("--datasets", default="S0,S1,S2,S3,S0_prime,S1_prime,S2_prime,S3_prime")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[env] device={device}", flush=True)

    features = np.load(BASE / "features" / "dinov3_features.npy")
    lab = pd.read_csv(ANNO / "all_labels.csv", encoding="utf-8")
    g_df = pd.read_csv(SPL / "gallery.csv", encoding="utf-8")
    splits = {}
    for name, fn in [("train", "train.csv"), ("query", "query.csv"),
                     ("gallery", "gallery.csv"), ("cross", "cross_source_test.csv")]:
        splits[name] = set(pd.read_csv(SPL / fn, encoding="utf-8")["global_idx"].tolist())

    q_idx, variants = build_variants(splits, lab, g_df)
    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    cats = sorted(lab["label"].dropna().unique())
    cat_of = dict(zip(lab["global_idx"], lab["label"]))
    print(f"[data] features={features.shape} query(seed)={len(q_idx)}", flush=True)
    for v in want:
        print(f"[data] {v:<12s} gallery={len(variants[v])}", flush=True)

    varr = {v: (np.array(variants[v], dtype=np.int64),
                np.array([cats.index(cat_of[i]) for i in variants[v]], dtype=np.int64))
            for v in want}
    q_arr = np.array(q_idx, dtype=np.int64)
    q_lab = np.array([cats.index(cat_of[i]) for i in q_idx], dtype=np.int64)
    q_feat = features[q_arr]

    methods = set(args.methods.split(","))
    datasets = set(args.datasets.split(","))
    pat = re.compile(r"^(?P<method>[A-Za-z]+)(?:_K(?P<k>\d+))?_"
                     r"(?P<ds>S[0-3](?:_prime)?)_seed(?P<seed>\d+)$")

    targets = []
    for d in sorted(RES.iterdir()):
        m = pat.match(d.name)
        if not m:
            continue
        key = m["method"] + (f"_K{m['k']}" if m["k"] else "")
        if key in methods and m["ds"] in datasets and (d / "best_checkpoint.pt").exists():
            targets.append((d, key, m["ds"], int(m["seed"])))
    print(f"[runs] 待评估 {len(targets)} 个 checkpoint × {len(want)} 变体", flush=True)

    rows = []
    for n, (d, key, ds, seed) in enumerate(targets, 1):
        ckpt = torch.load(d / "best_checkpoint.pt", map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        model = ProjectionHead(input_dim=768, hidden_dim=512,
                               embed_dim=cfg.get("embed_dim", 512)).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        q_emb = extract_embeddings(model, q_feat, device)
        for v in want:
            g_arr, g_lab = varr[v]
            g_emb = extract_embeddings(model, features[g_arr], device)
            met = retrieval_metrics(q_emb, q_lab, g_emb, g_lab)
            rows.append({"run": d.name, "method": key, "k": cfg.get("subcenter_k", ""),
                         "dataset": ds, "seed": seed, "variant": v,
                         "n_query": len(q_idx), "n_gallery": len(g_arr),
                         "MAP": met["MAP"], "R@1": met["Recall@1"],
                         "R@5": met["Recall@5"], "R@10": met["Recall@10"]})
        if n % 10 == 0:
            print(f"  [{n}/{len(targets)}] {d.name}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\n[saved] {args.out}  rows={len(df)}", flush=True)

    piv = df.groupby(["variant", "method", "dataset"])["MAP"].mean().unstack(
        ["method", "dataset"])
    print("\n[MAP mean over seeds]")
    print(piv.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()