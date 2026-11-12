# -*- coding: utf-8 -*-
"""
run_e0_grid.py —— 生成并执行 E0-2 / E0-3 训练网格（可断点续跑：已有 best_checkpoint.pt 即跳过）

E0-2  解耦 scale / margin / 子中心数（修正 margin 单位后重训）
      e02a ArcFace s=30 m=0.5rad      基线：原稿"标准 ArcFace"应有的样子
      e02b ArcFace s=64 m=0.5rad      与 K1 同 scale，隔离 scale
      e02c SC-K1   s=30 m=0.5rad      与 ArcFace 同 scale，隔离架构
      (已有) SC-K1 s=64 m=0.5rad      原稿 K1
E0-3  固定优化步数预算（S0 的 40 epoch = 2000 步，作为各数据集共同预算）
      e03  ArcFace s=64 m=0.5rad  max_steps=2000  on S1/S2/S3
      e03  GCE      s=30          max_steps=2000  on S1/S2/S3
      e03l ArcFace s=64 m=0.5rad  max_steps=9440  on S0（反向对齐 S2 的预算）
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
# ---- E0-3 固定步数 ----
for ds in ["S1", "S2", "S3"]:
    for sd in SEEDS:
        JOBS.append(job("e03", "ArcFace", ds, sd, 64, 0.5, k=1, max_steps=2000))
        JOBS.append(job("e03", "GCE", ds, sd, 30, None, k=1, max_steps=2000))
# ---- E0-3 反向：S0 补足到 S2 的步数 ----
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