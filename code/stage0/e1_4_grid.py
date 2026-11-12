# -*- coding: utf-8 -*-
"""E1-4：干净协议下的「缺两条基线」补跑。

审稿意见 Major #2 要求补两类基线，原稿一条都没跑：
  1. 无 margin 的归一化 softmax（= 普通 CE 的 cosine 版）；
  2. 至少一种标准度量学习目标（已在 E0-2 跑过 CosFace / CircleLoss / ProxyNCA 的 S0）。

这里在 E1-1 的 5 折干净协议上补 NormSoftmax(m=0) 与 CosFace(m=0.5 rad)，
tier 取 S0 / S2 / S3（S0 = 只有人工标签，S2 = 原稿声称的最优档，S3 = 全量）。
5 折 × 3 档 × 2 方法 × 3 seed = 90 run。
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable
TRAIN = HERE / "train_v2.py"
SPLITS = HERE / "e1_splits"
OUT = HERE / "runs_e1"

METHODS = [
    ("NormSoftmax", 30.0, 0.0),   # 无 margin
    ("CosFace", 30.0, 0.5),       # 弧度
]
TIERS = ["S0", "S2", "S3"]
SEEDS = [42, 123, 456]
FOLDS = [0, 1, 2, 3, 4]


def main():
    jobs = []
    for fold in FOLDS:
        for method, scale, marg in METHODS:
            for tier in TIERS:
                idx = SPLITS / f"train_f{fold}_{tier}.json"
                if not idx.exists():
                    raise SystemExit(f"[missing] {idx}")
                for seed in SEEDS:
                    tag = f"e1f{fold}"
                    name = f"{tag}_{method}_{tier}_seed{seed}"
                    d = OUT / name
                    if (d / "best_checkpoint.pt").exists():
                        continue
                    jobs.append(dict(fold=fold, method=method, scale=scale,
                                     marg=marg, tier=tier, seed=seed,
                                     idx=str(idx), tag=tag, name=name))

    print(f"[extra] 共 {len(jobs)} 个 run 待跑", flush=True)
    done = skip = fail = 0
    t0 = time.time()
    for i, j in enumerate(jobs, 1):
        cmd = [PY, str(TRAIN),
               "--method", j["method"], "--dataset", j["tier"],
               "--seed", str(j["seed"]),
               "--scale", str(j["scale"]), "--margin_rad", str(j["marg"]),
               "--subcenter_k", "3",
               "--idx_json", j["idx"], "--out_root", str(OUT),
               "--tag", j["tag"]]
        t = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        dt = time.time() - t
        if r.returncode == 0:
            done += 1
            st = f"{dt:.0f}s"
        elif (OUT / j["name"] / "best_checkpoint.pt").exists():
            skip += 1
            st = "skip"
        else:
            fail += 1
            st = "FAIL"
            print(f"[fail] {j['name']}\n{r.stdout[-800:]}\n{r.stderr[-1500:]}", flush=True)
        if i % 10 == 0 or i == len(jobs):
            print(f"  [{i}/{len(jobs)}] {j['name']} {st} | 累计 {(time.time()-t0)/60:.1f} min "
                  f"| done={done} skip={skip} fail={fail}", flush=True)
    print(f"\n[extra] 结束：done={done} skip={skip} fail={fail} 总耗时 {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()