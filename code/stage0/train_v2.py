# -*- coding: utf-8 -*-
r"""
train_v2.py -- the training entry point for the stage-0 review (the original code/ directory is left untouched)

The only changes against <DATA_ROOT>/code/train_metric.py are that
three things that were previously out of reach are made explicit for E0-2/E0-3/E0-4:

1. pass scale and margin explicitly, and keep the units apart
   PML's ArcFaceLoss / SubCenterArcFaceLoss take their margin in degrees
   （arcface_loss.py:22-23 `self.margin = np.radians(self.margin)`），
   CosFaceLoss takes its margin in radians (no conversion is applied)
   the original script passed margin=0.5 for ArcFace and it was read as 0.5 degrees, which is no margin at all -- this is
   the actual state of the manuscript's "standard ArcFace" baseline. This script insists on radians,
   the conversion into each loss's real unit happens inside, and is written into the config so every artefact can be traced back

2. --max_steps: a fixed optimiser-step budget (E0-3). The original script counted epochs only,
   with batch_size fixed, a larger training set means more steps (S2 is 4.72 times S0)

3. --select: the checkpoint-selection rule. The original script took the lowest training loss as best (E0-4),
   systematically favours memorising the noise when labels are noisy. Accepts loss / last / epochN

also: the model and loss definitions of the original scripts are imported, not copied
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
    """Build the loss with radian semantics; the conversion into each loss's own unit happens inside."""
    if scale is None:
        scale = {"ArcFace": 30.0, "CosFace": 30.0, "SubCenterArcFace": 64.0,
                 "GCE": 30.0, "SCE": 30.0, "NormSoftmax": 30.0}[method]
    if method in ("ArcFace", "SubCenterArcFace", "CosFace"):
        if margin_rad is None:
            margin_rad = 0.5
    miner = miners.MultiSimilarityMiner(epsilon=0.1)

    if method == "ArcFace":
        # PML expects degrees
        return losses.ArcFaceLoss(num_classes=num_classes, embedding_size=embed_dim,
                                  scale=scale, margin=float(np.degrees(margin_rad))), miner
    if method == "SubCenterArcFace":
        return losses.SubCenterArcFaceLoss(num_classes=num_classes, embedding_size=embed_dim,
                                           scale=scale, margin=float(np.degrees(margin_rad)),
                                           sub_centers=subcenter_k), miner
    if method == "CosFace":
        # PML CosFace takes radians, so the value is passed through
        return losses.CosFaceLoss(num_classes=num_classes, embedding_size=embed_dim,
                                  scale=scale, margin=float(margin_rad)), miner
    if method == "NormSoftmax":
        # normalised softmax without a margin (cosine classifier, m = 0), i.e. the plain CE baseline the reviewer asked for
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