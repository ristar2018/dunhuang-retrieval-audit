# -*- coding: utf-8 -*-
r"""
train_v2.py —— 阶段 0 复核用训练入口（不改动原始 code/ 目录）

相对 <DATA_ROOT>\retrieval\code\train_metric.py 的改动，仅为把
E0-2/E0-3/E0-4 需要的三项"此前不可控"的东西显式化：

1. scale / margin 显式传参，并区分单位。
   PML 的 ArcFaceLoss / SubCenterArcFaceLoss 的 margin 入参单位是「度」
   （arcface_loss.py:22-23 `self.margin = np.radians(self.margin)`），
   CosFaceLoss 的 margin 入参单位是「弧度」（无 radians 转换）。
   原脚本 ArcFace 传 margin=0.5 被解释成 0.5°，等于没有 margin —— 这是
   原稿"标准 ArcFace"基线的实际状态。本脚本强制要求按弧度书写，
   内部再按每个损失的真实单位换算，并写入 config 以便产物可反查。

2. --max_steps：固定优化步数预算（E0-3）。原脚本只按 epoch 计，
   在 batch_size 固定时训练集越大步数越多（S2 是 S0 的 4.72 倍）。

3. --select：checkpoint 选择规则。原脚本按训练损失最小选 best（E0-4），
   在噪声标签下系统性偏向记忆噪声。可选 loss / last / epochN。

另：复用原脚本的模型与损失定义（直接 import），不复制实现。
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

CODE_DIR = Path(r"<DATA_ROOT>\retrieval\code")
sys.path.insert(0, str(CODE_DIR))
import train_metric as tm  # noqa: E402

from pytorch_metric_learning import losses, miners  # noqa: E402

BASE_DIR = Path(r"<DATA_ROOT>\retrieval")
FEATURES_NPY = BASE_DIR / "features" / "dinov3_features.npy"
GRADIENT_DIR = BASE_DIR / "data" / "gradients"
LABELS_CSV = BASE_DIR / "annotations" / "all_labels.csv"
OUT_ROOT = Path(__file__).parent / "runs_e0"


def build_loss(method, num_classes, embed_dim, scale, margin_rad, subcenter_k, q=0.7):
    """按「弧度」语义构造损失；内部按各损失的真实单位换算。"""
    if scale is None:
        scale = {"ArcFace": 30.0, "CosFace": 30.0, "SubCenterArcFace": 64.0,
                 "GCE": 30.0, "SCE": 30.0, "NormSoftmax": 30.0}[method]
    if method in ("ArcFace", "SubCenterArcFace", "CosFace"):
        if margin_rad is None:
            margin_rad = 0.5
    miner = miners.MultiSimilarityMiner(epsilon=0.1)

    if method == "ArcFace":
        # PML 单位＝度
        return losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embed_dim,
                                  scale=scale, margin=float(np.degrees(margin_rad))), miner
    if method == "SubCenterArcFace":
        return losses.SubCenterArcFaceLoss(num_classes=num_classes, embedding_size=embed_dim,
                                           scale=scale, margin=float(np.degrees(margin_rad)),
                                           sub_centers=subcenter_k), miner
    if method == "CosFace":
        # PML CosFace 单位＝弧度，直接传
        return losses.CosFaceLoss(num_classes=num_classes, embedding_size=embed_dim,
                                  scale=scale, margin=float(margin_rad)), miner
    if method == "NormSoftmax":
        # 无 margin 的归一化 softmax（cosine classifier, m=0），即审稿人要求的普通 CE 基线
        return losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embed_dim,
                                  scale=scale, margin=0.0), miner
    if method == "GCE":
        return tm.GCEClassificationLoss(embed_dim=embed_dim, num_classes=num_classes,
                                        scale=scale, q=q), None
    if method == "SCE":
        return tm.SCEClassificationLoss(embed_dim=embed_dim, num_classes=num_classes,
                                        scale=scale, alpha=0.1, beta=1.0), None
    raise ValueError(method)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True,
                   choices=["ArcFace", "CosFace", "SubCenterArcFace", "GCE", "SCE", "NormSoftmax"])
    p.add_argument("--dataset", required=True,
                   choices=["S0", "S1", "S2", "S3",
                            "S0_prime", "S1_prime", "S2_prime", "S3_prime"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scale", type=float, default=None)
    p.add_argument("--margin_rad", type=float, default=None,
                   help="角间隔，弧度。0.5 rad = 28.6 deg（原稿 SubCenter 的实际值）")
    p.add_argument("--subcenter_k", type=int, default=3)
    p.add_argument("--q", type=float, default=0.7, help="GCE 的 q")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--max_steps", type=int, default=0, help=">0 时固定优化步数预算")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--embed_dim", type=int, default=512)
    p.add_argument("--warmup_steps", type=int, default=0,
                   help=">0 时按步 warmup；否则沿用 3-epoch warmup")
    p.add_argument("--select", default="loss", choices=["loss", "last"])
    p.add_argument("--tag", default="v2")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--idx_json", default=None,
                   help="覆盖训练索引来源：指向一个含 global_idx 列表的 JSON"
                        "（E1 交叉拟合用，默认读 data/gradients/{dataset}_indices.json）")
    p.add_argument("--out_root", default=None, help="产物根目录（默认 runs_e0）")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tag = args.tag
    if args.method == "SubCenterArcFace":
        name = f"{tag}_{args.method}_K{args.subcenter_k}_{args.dataset}_seed{args.seed}"
    else:
        name = f"{tag}_{args.method}_{args.dataset}_seed{args.seed}"
    exp_dir = (Path(args.out_root) if args.out_root else OUT_ROOT) / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    feats = np.load(str(FEATURES_NPY))
    if args.idx_json:
        with open(args.idx_json, encoding="utf-8") as f:
            gd = json.load(f)
    else:
        with open(GRADIENT_DIR / f"{args.dataset}_indices.json", encoding="utf-8") as f:
            gd = json.load(f)
    idx = gd if isinstance(gd, list) else gd["indices"]

    labels_csv = pd.read_csv(LABELS_CSV, encoding="utf-8")
    lookup = dict(zip(labels_csv["global_idx"], labels_csv["label"]))
    cats = sorted(labels_csv["label"].unique())
    cat_to_idx = {c: i for i, c in enumerate(cats)}

    x = feats[idx]
    y = np.array([cat_to_idx[lookup[i]] for i in idx])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tm.ProjectionHead(input_dim=768, hidden_dim=512, embed_dim=args.embed_dim).to(device)
    loss_fn, miner = build_loss(args.method, len(cats), args.embed_dim,
                                args.scale, args.margin_rad, args.subcenter_k, args.q)
    loss_fn = loss_fn.to(device)

    dl = torch.utils.data.DataLoader(tm.FeatureDataset(x, y), batch_size=args.batch_size,
                                     shuffle=True, num_workers=0, pin_memory=True,
                                     drop_last=True)

    params = list(model.parameters())
    if hasattr(loss_fn, "parameters") and len(list(loss_fn.parameters())) > 0:
        params += list(loss_fn.parameters())
    opt = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(dl)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    warm = args.warmup_steps if args.warmup_steps > 0 else steps_per_epoch * 3

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        prog = (step - warm) / max(total_steps - warm, 1)
        return 0.5 * (1.0 + np.cos(np.pi * min(prog, 1.0)))

    sched = LambdaLR(opt, lr_lambda)

    print(f"[{name}] n_train={len(idx)} steps/epoch={steps_per_epoch} "
          f"total_steps={total_steps} margin_rad={args.margin_rad} scale={args.scale}")

    best_loss = float("inf")
    gstep = 0
    log = []
    t_start = time.time()
    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, nb = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            emb = model(xb)
            if miner is not None:
                loss = loss_fn(emb, yb, miner(emb, yb))
            else:
                loss = loss_fn(emb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
            sched.step()
            gstep += 1
            tl += loss.item()
            nb += 1
            if args.max_steps > 0 and gstep >= args.max_steps:
                stop = True
                break
        avg = tl / max(nb, 1)
        log.append({"epoch": epoch, "global_step": gstep, "loss": avg,
                    "lr": opt.param_groups[0]["lr"], "n_batches": nb})
        if epoch % 5 == 0 or epoch == 1 or stop:
            print(f"  E{epoch:3d} step={gstep:6d} loss={avg:.4f} lr={opt.param_groups[0]['lr']:.2e}")
        if avg < best_loss:
            best_loss = avg
            torch.save({"epoch": epoch, "global_step": gstep,
                        "model_state_dict": model.state_dict(),
                        "loss": best_loss}, exp_dir / "best_checkpoint.pt")
        if epoch % args.save_every == 0 and not stop:
            torch.save({"epoch": epoch, "global_step": gstep,
                        "model_state_dict": model.state_dict(),
                        "loss": avg}, exp_dir / f"checkpoint_epoch{epoch}.pt")
        if stop:
            break

    torch.save({"epoch": log[-1]["epoch"], "global_step": gstep,
                "model_state_dict": model.state_dict(), "loss": log[-1]["loss"],
                "config": vars(args)}, exp_dir / "final_checkpoint.pt")
    cfg = dict(vars(args))
    cfg.update({"n_train": len(idx), "steps_per_epoch": steps_per_epoch,
                "total_steps_planned": total_steps, "global_step_done": gstep,
                "best_loss": best_loss, "margin_deg_effective": float(np.degrees(args.margin_rad))
                if args.margin_rad is not None and args.method != "CosFace" else
                (float(np.degrees(args.margin_rad)) if args.margin_rad is not None else None),
                "elapsed_sec": round(time.time() - t_start, 1),
                "pml_version": __import__("pytorch_metric_learning").__version__,
                "torch_version": torch.__version__})
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    with open(exp_dir / "training_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"[{name}] done step={gstep} best_loss={best_loss:.4f} "
          f"{cfg['elapsed_sec']}s -> {exp_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())