# -*- coding: utf-8 -*-
r"""
E1-1 evaluation: score each fold's trained artefact against **that fold's own** query / gallery (both drawn from hf_dunhuang
human-labelled images, disjoint from the training seed set)

differences from E0:
- B_seed_only in E0 holds 339 images (that is all the human labels the original split left in the gallery);
  E1 gives roughly 455 gallery and 455 query images per fold, and the five folds together cover all 2,278 images,
  equivalent to enlarging the clean evaluation set 4.5-fold, and repeated over five independent splits
- output micro-MAP (mean AP over all queries) and macro-MAP (AP per class, then averaged),
  the latter checks whether the gain comes from the large classes alone

output: e1_eval.csv (one row per run), e1_perclass.csv (one row per run x class)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from e0_1b_clean_gallery import BASE, ANNO, ProjectionHead, embed  # noqa: E402

OUT = Path(__file__).parent
SPLITS = OUT / "e1_splits"
RUNS = OUT / "runs_e1"

PAT = re.compile(r"^(?P<tag>e1f(?P<fold>\d))_(?P<method>[A-Za-z]+)(?:_K(?P<k>\d+))?_"
                 r"(?P<tier>S[0-3](?:_prime)?)_seed(?P<seed>\d+)$")


def ap_per_query(sim, ql, gl):
    order = np.argsort(-sim, axis=1)
    aps = np.empty(len(ql))
    for i in range(len(ql)):
        rel = (gl[order[i]] == ql[i]).astype(np.float64)
        c = np.cumsum(rel)
        aps[i] = ((c / (np.arange(len(rel)) + 1.0)) * rel).sum() / max(rel.sum(), 1.0)
    return aps, order


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat = np.load(BASE / "features" / "dinov3_features.npy")
    lab = pd.read_csv(ANNO / "all_labels.csv", encoding="utf-8")
    cat = dict(zip(lab["global_idx"], lab["label"]))
    cats = sorted(lab["label"].dropna().unique())
    cidx = {c: i for i, c in enumerate(cats)}

    folds = {}
    for k in range(5):
        q = json_load(SPLITS / f"fold{k}_query.json")
        g = json_load(SPLITS / f"fold{k}_gallery.json")
        folds[k] = dict(
            q=np.array(q), ql=np.array([cidx[cat[i]] for i in q]),
            g=np.array(g), gl=np.array([cidx[cat[i]] for i in g]))
        assert len(set(q) & set(g)) == 0

    runs = []
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir():
            continue
        m = PAT.match(d.name)
        if m:
            gd = m.groupdict()
            runs.append(dict(dir=d, fold=int(gd["fold"]),
                             method=gd["method"] + (f"_K{gd['k']}" if gd.get("k") else ""),
                             tier=gd["tier"], seed=int(gd["seed"])))
    print(f"[runs] {len(runs)} 个 run 目录", flush=True)

    rows, pcrows = [], []
    for n, r in enumerate(runs, 1):
        f = folds[r["fold"]]
        qf = feat[f["q"]]
        gcd = {}
        for c in cats:
            sel = f["gl"] == cidx[c]
            gcd[c] = int(sel.sum())
        for ck_name, ck_file in [("best", "best_checkpoint.pt"), ("final", "final_checkpoint.pt")]:
            p = r["dir"] / ck_file
            if not p.exists():
                continue
            ck = torch.load(p, map_location=device, weights_only=False)
            ed = ck.get("config", {}).get("embed_dim", 512) if isinstance(ck.get("config"), dict) else 512
            model = ProjectionHead(embed_dim=ed).to(device)
            model.load_state_dict(ck["model_state_dict"])
            qe = embed(model, qf, device)
            ge = embed(model, feat[f["g"]], device)
            sim = qe @ ge.T
            aps, order = ap_per_query(sim, f["ql"], f["gl"])
            order_k = order[:, :10]
            rec = dict(run=r["dir"].name, fold=r["fold"], method=r["method"],
                       tier=r["tier"], seed=r["seed"], ckpt=ck_name,
                       epoch=ck.get("epoch", -1), global_step=ck.get("global_step", -1),
                       n_query=len(f["q"]), n_gallery=len(f["g"]),
                       MAP=float(aps.mean()),
                       R1=float((f["gl"][order_k[:, :1]] == f["ql"][:, None]).any(1).mean()),
                       R5=float((f["gl"][order_k[:, :5]] == f["ql"][:, None]).any(1).mean()),
                       R10=float((f["gl"][order_k] == f["ql"][:, None]).any(1).mean()))
            per = []
            for c in cats:
                sel = f["ql"] == cidx[c]
                per.append(aps[sel].mean() if sel.any() else np.nan)
                pcrows.append(dict(run=r["dir"].name, fold=r["fold"], method=r["method"],
                                   tier=r["tier"], seed=r["seed"], ckpt=ck_name, label=c,
                                   AP=float(per[-1]), n_query=int(sel.sum()),
                                   n_gallery=gcd[c]))
            per = np.array(per, dtype=float)
            rec["macro_MAP"] = float(np.nanmean(per))
            rec["n_null_class"] = int(np.isnan(per).sum())
            rows.append(rec)
            del model
        if n % 20 == 0:
            print(f"  [{n}/{len(runs)}] {r['dir'].name}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e1_eval.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pcrows).to_csv(OUT / "e1_perclass.csv", index=False, encoding="utf-8-sig")
    print(f"\n[saved] eval rows={len(df)} perclass rows={len(pcrows)}")
    b = df[df.ckpt == "best"]
    print("\n=== best，micro-MAP（行=tier 列=method，5 折×3 seed 均值）===")
    print(b.groupby(["tier", "method"])["MAP"].mean().round(4).unstack().to_string())
    print("\n=== best，macro-MAP ===")
    print(b.groupby(["tier", "method"])["macro_MAP"].mean().round(4).unstack().to_string())


def json_load(p):
    import json
    with open(p, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    main()