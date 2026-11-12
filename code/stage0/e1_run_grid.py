# -*- coding: utf-8 -*-
r"""
E1-1 driver for the cross-fitted training grid

grids:
  main grid  5 folds x {S0,S1,S2,S3} x 4 methods x 3 seeds = 240 runs
  control grid  5 folds x {S0',S1',S2',S3'} x 2 methods x 3 seeds = 120 runs

method configuration (all with the corrected margin semantics; every margin is written in radians):
  ArcFace   s=30, m=0.5 rad -- what the manuscript's "standard ArcFace" should have been
            (the manuscript passed m = 0.5, PML read it as degrees, so the real margin was 0.0087 rad, i.e. none)
  GCE       s=30, q=0.7
  SCE       s=30
  SC-K3     SubCenterArcFace s=64, m=0.5 rad, K=3

SC-K1 is not re-run: E0-2 already showed that with sub_centers = 1, the same scale and the same margin, ArcFaceLoss
state_dict is bit-identical (largest absolute difference 0.000e+00), so it duplicates ArcFace s=64

Resumable: a target directory that already holds best_checkpoint.pt is skipped
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