# -*- coding: utf-8 -*-
"""E0-2/E0-3/E0-4 评估：把「原稿 129 个 run」与「阶段 0 新训 run」放到同一评估协议下。

三个 gallery 口径：
  A_current   原稿口径（gallery∪cross，全部 label_type，21,703）——复现论文数字
  B_seed_only 严格干净（339 张专家标签，与训练集完全不相交）
  F_seed_all  干净扩容（1,941 张专家标签）

同时评估 best / final / epoch{10,20,30,40}，用于 E0-4（选模规则敏感性）。
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from e0_1b_clean_gallery import BASE, RES, SPL, ANNO, ProjectionHead, embed, metrics  # noqa: E402

OUT = Path(__file__).parent
NEW_RUNS = OUT / "runs_e0"

PAT_OLD = re.compile(r"^(?P<method>[A-Za-z]+)(?:_K(?P<k>\d+))?_(?P<ds>S[0-3](?:_prime)?)_seed(?P<seed>\d+)$")
PAT_NEW = re.compile(r"^(?P<tag>e\d+[a-z]?)_(?P<method>[A-Za-z]+)(?:_K(?P<k>\d+))?_"
                     r"(?P<ds>S[0-3](?:_prime)?)_seed(?P<seed>\d+)$")


def collect():
    rows = []
    for root, tag_default in [(RES, "orig"), (NEW_RUNS, None)]:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            m = PAT_NEW.match(d.name) if tag_default is None else PAT_OLD.match(d.name)
            if not m:
                continue
            gd = m.groupdict()
            method = gd["method"] + (f"_K{gd['k']}" if gd.get("k") else "")
            rows.append(dict(dir=d, tag=gd.get("tag") or tag_default, method=method,
                             dataset=gd["ds"], seed=int(gd["seed"])))
    return rows


def ckpts(d):
    out = []
    if (d / "best_checkpoint.pt").exists():
        out.append(("best", d / "best_checkpoint.pt"))
    if (d / "final_checkpoint.pt").exists():
        out.append(("final", d / "final_checkpoint.pt"))
    for p in sorted(d.glob("checkpoint_epoch*.pt")):
        ep = re.search(r"epoch(\d+)", p.name).group(1)
        out.append((f"ep{ep}", p))
    # 新训 run 的 best 已写入 checkpoint 的 epoch 字段，避免重复计分
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat = np.load(BASE / "features" / "dinov3_features.npy")
    lab = pd.read_csv(ANNO / "all_labels.csv", encoding="utf-8")
    sp = {n: set(pd.read_csv(SPL / f, encoding="utf-8")["global_idx"])
          for n, f in [("train", "train.csv"), ("query", "query.csv"),
                       ("gallery", "gallery.csv"), ("cross", "cross_source_test.csv")]}
    lt = dict(zip(lab["global_idx"], lab["label_type"]))
    cat = dict(zip(lab["global_idx"], lab["label"]))
    cats = sorted(lab["label"].dropna().unique())

    q_idx = sorted(i for i in sp["query"] if lt.get(i) == "seed")
    qs = set(q_idx)
    g_all = sp["gallery"] | sp["cross"]
    seed_all = {i for i in lab["global_idx"] if lt.get(i) == "seed"}
    variants = {
        "A_current": sorted(g_all - qs),
        "B_seed_only": sorted(i for i in g_all - qs if lt.get(i) == "seed"),
        "F_seed_all": sorted(seed_all - qs),
    }
    qa = np.array(q_idx)
    ql = np.array([cats.index(cat[i]) for i in q_idx])
    vr = {k: (np.array(v), np.array([cats.index(cat[i]) for i in v])) for k, v in variants.items()}
    qf = feat[qa]
    print(f"[data] query={len(q_idx)} " + " ".join(f"{k}={len(v[0])}" for k, v in vr.items()))

    runs = collect()
    print(f"[runs] {len(runs)} 个 run 目录")
    rows = []
    for n, r in enumerate(runs, 1):
        for ck_name, ck_path in ckpts(r["dir"]):
            try:
                ck = torch.load(ck_path, map_location=device, weights_only=False)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {ck_path.name}: {e}")
                continue
            ed = 512
            if isinstance(ck.get("config"), dict):
                ed = ck["config"].get("embed_dim", 512)
            model = ProjectionHead(embed_dim=ed).to(device)
            model.load_state_dict(ck["model_state_dict"])
            qe = embed(model, qf, device)
            for v, (garr, glab) in vr.items():
                ge = embed(model, feat[garr], device)
                mt = metrics(qe, ql, ge, glab)
                rows.append(dict(run=r["dir"].name, tag=r["tag"], method=r["method"],
                                 dataset=r["dataset"], seed=r["seed"], ckpt=ck_name,
                                 epoch=ck.get("epoch", -1), global_step=ck.get("global_step", -1),
                                 variant=v, n_gallery=len(garr), MAP=mt["MAP"],
                                 R1=mt["R@1"], R5=mt["R@5"], R10=mt["R@10"]))
            del model
        if n % 20 == 0:
            print(f"  [{n}/{len(runs)}] {r['dir'].name}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "e0_2_eval.csv", index=False, encoding="utf-8-sig")
    print(f"\n[saved] rows={len(df)}")

    b = df[df.ckpt == "best"]
    print("\n=== best checkpoint，B_seed_only（严格干净）===")
    p = b[b.variant == "B_seed_only"].groupby(["tag", "method", "dataset"])["MAP"].mean().round(4)
    print(p.unstack().to_string())
    print("\n=== best checkpoint，F_seed_all（干净扩容）===")
    p = b[b.variant == "F_seed_all"].groupby(["tag", "method", "dataset"])["MAP"].mean().round(4)
    print(p.unstack().to_string())


if __name__ == "__main__":
    main()