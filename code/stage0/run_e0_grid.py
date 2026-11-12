# -*- coding: utf-8 -*-
"""
run_e0_grid.py -- build and run the E0-2 / E0-3 training grids (resumable: a run whose best_checkpoint.pt already exists is skipped)

E0-2  decouple scale / margin / sub-centre count (retrained after fixing the margin unit)
      e02a ArcFace s=30 m=0.5 rad      baseline: what the manuscript's "standard ArcFace" should look like
      e02b ArcFace s=64 m=0.5 rad      same scale as K1, which isolates the scale
      e02c SC-K1   s=30 m=0.5 rad      same scale as ArcFace, which isolates the architecture
      (existing) SC-K1 s=64 m=0.5 rad      the manuscript's K1
E0-3  a fixed optimiser-step budget (S0's 40 epochs = 2,000 steps, used as the common budget)
      e03  ArcFace s=64 m=0.5rad  max_steps=2000  on S1/S2/S3
      e03  GCE      s=30          max_steps=2000  on S1/S2/S3
      e03l ArcFace s=64 m=0.5 rad  max_steps=9440  on S0 (matching the S2 budget in the other direction)
"""
import subprocess
import sys
from pathlib import Path

PY = r"<DATA_ROOT>\envs\chartdetect\python.exe"
HERE = Path(__file__).parent
TRAIN = HERE / "train_v2.py"
RUNS = HERE / "runs_e0"
SEEDS = [42, 123, 456]


def job(tag, method, dataset, seed, scale, margin_rad, k=1, max_steps=0, epochs=40):
    name = f"{tag}_{method}_K{k}_{dataset}_seed{seed}" if method == "SubCenterArcFace" \
        else f"{tag}_{method}_{dataset}_seed{seed}"
    return dict(tag=tag, method=method, dataset=dataset, seed=seed, scale=scale,
                margin_rad=margin_rad, k=k, max_steps=max_steps, epochs=epochs, name=name)


JOBS = []
# ---- E0-2 ----
for ds in ["S0", "S2", "S3"]:
    for sd in SEEDS:
        JOBS.append(job("e02a", "ArcFace", ds, sd, 30, 0.5))
        JOBS.append(job("e02b", "ArcFace", ds, sd, 64, 0.5))
        JOBS.append(job("e02c", "SubCenterArcFace", ds, sd, 30, 0.5, k=1))
# ---- E0-3, fixed step budget ----
for ds in ["S1", "S2", "S3"]:
    for sd in SEEDS:
        JOBS.append(job("e03", "ArcFace", ds, sd, 64, 0.5, k=1, max_steps=2000))
        JOBS.append(job("e03", "GCE", ds, sd, 30, None, k=1, max_steps=2000))
# ---- E0-3, reverse direction: pad S0 to the S2 step budget ----
for sd in SEEDS:
    JOBS.append(job("e03l", "ArcFace", "S0", sd, 64, 0.5, k=1, max_steps=9440,
                    epochs=189))


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    todo = [j for j in JOBS if not (RUNS / j["name"] / "best_checkpoint.pt").exists()]
    print(f"总任务 {len(JOBS)}，待跑 {len(todo)}（已完成 {len(JOBS)-len(todo)}）", flush=True)
    fail = []
    for i, j in enumerate(todo, 1):
        cmd = [PY, str(TRAIN), "--method", j["method"], "--dataset", j["dataset"],
               "--seed", str(j["seed"]), "--scale", str(j["scale"]),
               "--epochs", str(j["epochs"]), "--tag", j["tag"],
               "--subcenter_k", str(j["k"]), "--save_every", "10"]
        if j["margin_rad"] is not None:
            cmd += ["--margin_rad", str(j["margin_rad"])]
        if j["max_steps"]:
            cmd += ["--max_steps", str(j["max_steps"])]
        print(f"[{i}/{len(todo)}] {j['name']} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        tail = [ln for ln in (r.stdout or "").strip().splitlines() if ln][-1:]
        print("     " + (tail[0] if tail else "(no output)"), flush=True)
        if r.returncode != 0 or not (RUNS / j["name"] / "best_checkpoint.pt").exists():
            fail.append((j["name"], r.returncode, (r.stderr or "")[-500:]))
            print("     !! FAILED", flush=True)
    print(f"\n完成。失败 {len(fail)} 项", flush=True)
    for name, rc, err in fail:
        print(f"  {name} rc={rc}\n    {err}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())