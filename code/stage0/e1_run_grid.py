# -*- coding: utf-8 -*-
r"""
E1-1 交叉拟合训练网格驱动。

网格：
  主网格  5 折 × {S0,S1,S2,S3} × 4 方法 × 3 seeds = 240 run
  控制组  5 折 × {S0',S1',S2',S3'} × 2 方法 × 3 seeds = 120 run

方法配置（全部按已修正的 margin 语义，margin 一律按弧度书写）：
  ArcFace   s=30, m=0.5 rad —— 这是原稿"标准 ArcFace"本该有的配置
            （原稿传 m=0.5 被 PML 当度，实际 0.0087 rad ≈ 无 margin）
  GCE       s=30, q=0.7
  SCE       s=30
  SC-K3     SubCenterArcFace s=64, m=0.5 rad, K=3

不跑 SC-K1：E0-2 已证明 sub_centers=1 与 ArcFaceLoss 在同 scale/margin 下
state_dict 逐位相同（最大绝对差 0.000e+00），故与 ArcFace s=64 重复。

断点续跑：目标目录已存在 best_checkpoint.pt 即跳过。
"""

import subprocess
import sys
import time
from pathlib import Path

OUT = Path(__file__).parent
SPLITS = OUT / "e1_splits"
PY = sys.executable
SEEDS = [42, 123, 456]
N_FOLD = 5
TIERS = ["S0", "S1", "S2", "S3"]

METHODS = {
    "ArcFace":  ["--method", "ArcFace", "--scale", "30", "--margin_rad", "0.5"],
    "GCE":      ["--method", "GCE", "--scale", "30"],
    "SCE":      ["--method", "SCE", "--scale", "30"],
    "SCK3":     ["--method", "SubCenterArcFace", "--subcenter_k", "3",
                 "--scale", "64", "--margin_rad", "0.5"],
}
MAIN_METHODS = ["ArcFace", "GCE", "SCE", "SCK3"]
PRIME_METHODS = ["ArcFace", "GCE"]

RUN_ROOT = OUT / "runs_e1"


def job_list(mode):
    jobs = []
    methods = MAIN_METHODS if mode == "main" else PRIME_METHODS
    tiers = TIERS if mode == "main" else [f"{t}_prime" for t in TIERS]
    for k in range(N_FOLD):
        for tier in tiers:
            for mname in methods:
                for seed in SEEDS:
                    tag = f"e1f{k}"
                    jobs.append(dict(fold=k, tier=tier, method=mname, seed=seed,
                                     tag=tag,
                                     idx_json=str(SPLITS / f"train_f{k}_{tier}.json")))
    return jobs


def run_dir_name(j):
    if j["method"] == "SCK3":
        return f"{j['tag']}_SubCenterArcFace_K3_{j['tier']}_seed{j['seed']}"
    return f"{j['tag']}_{j['method']}_{j['tier']}_seed{j['seed']}"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    jobs = job_list(mode)
    print(f"[{mode}] 共 {len(jobs)} 个 run 待检查", flush=True)
    done = skip = fail = 0
    t0 = time.time()
    for i, j in enumerate(jobs, 1):
        d = RUN_ROOT / run_dir_name(j)
        if (d / "best_checkpoint.pt").exists():
            skip += 1
            continue
        cmd = [PY, str(OUT / "train_v2.py")] + METHODS[j["method"]] + [
            "--dataset", j["tier"].replace("_prime", "_prime") if "_prime" in j["tier"] else j["tier"],
            "--seed", str(j["seed"]), "--tag", j["tag"],
            "--idx_json", j["idx_json"], "--epochs", "40", "--batch_size", "32",
            "--save_every", "10", "--out_root", str(RUN_ROOT),
        ]
        t = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        if p.returncode != 0:
            fail += 1
            print(f"  !! {run_dir_name(j)} 失败\n{p.stdout[-800:]}\n{p.stderr[-1500:]}",
                  flush=True)
            continue
        done += 1
        if done % 10 == 0 or i == len(jobs):
            el = time.time() - t0
            print(f"  [{i}/{len(jobs)}] {run_dir_name(j)} {time.time()-t:.0f}s "
                  f"| 累计 {el/60:.1f} min | done={done} skip={skip} fail={fail}",
                  flush=True)
    print(f"\n[{mode}] 结束：done={done} skip={skip} fail={fail} "
          f"总耗时 {(time.time()-t0)/60:.1f} min", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())