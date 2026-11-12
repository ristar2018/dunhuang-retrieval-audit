# -*- coding: utf-8 -*-
"""
Step 6: 训练 Pipeline
DINOv3 + LoRA + SubCenterArcFace 度量学习训练。
- 优化器: AdamW (backbone LoRA lr=1e-4, head lr=5e-4)
- 调度器: CosineAnnealingLR + warmup
- 损失: CrossEntropy (ArcFace/CosFace logits)
- 评估: 每 EVAL_EVERY epoch 在 query/gallery/cross-source 上评估
- 混合精度: BF16 (Blackwell GPU支持)
"""
import os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    EPOCHS, LR_BACKBONE, LR_HEAD, WEIGHT_DECAY, WARMUP_EPOCHS,
    CKPT_DIR, LOG_DIR, EVAL_EVERY, SAVE_EVERY, EXP_NAME,
    BATCH_SIZE, TRAIN_BS, RANDOM_SEED, USE_MINER, MINER_NAME, MINER_EPSILON,
    EXP_DIR, RESULT_DIR
)
from importlib import import_module

# Import from sibling scripts
_dataset_mod = import_module("04_dataset")
_model_mod = import_module("05_model")
_eval_mod = import_module("07_evaluate")

build_dataloaders = _dataset_mod.build_dataloaders
DINOv3MetricModel = _model_mod.DINOv3MetricModel
evaluate = _eval_mod.evaluate


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model):
    """分层学习率：LoRA参数低学习率，head高学习率"""
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    optimizer = AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
        {"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
    ])
    print(f"[Optimizer] Backbone params: {len(backbone_params)} tensors, "
          f"lr={LR_BACKBONE}")
    print(f"[Optimizer] Head params: {len(head_params)} tensors, lr={LR_HEAD}")
    return optimizer


def build_scheduler(optimizer, train_loader):
    """Warmup + Cosine Annealing"""
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = steps_per_epoch * WARMUP_EPOCHS

    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                             milestones=[warmup_steps])
    return scheduler


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = 0
    correct = 0
    total = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        imgs = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            embeddings, logits = model(imgs, labels)
            loss = F.cross_entropy(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=5.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 50 == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(f"  E{epoch} [{batch_idx}/{len(loader)}] "
                  f"loss={loss.item():.4f} acc={correct/total:.3f} "
                  f"lr={lr:.2e} {elapsed:.0f}s")

    return total_loss / n_batches, correct / total


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    print(f"[Experiment] {EXP_NAME}")

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Data
    print("\n[1/4] Building dataloaders...")
    loaders = build_dataloaders()

    # Model
    print("\n[2/4] Building model...")
    model = DINOv3MetricModel(use_lora=True).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total: {total_params/1e6:.1f}M, Trainable: {trainable_params/1e6:.2f}M "
          f"({100*trainable_params/total_params:.2f}%)")

    # Optimizer & scheduler
    print("\n[3/4] Building optimizer...")
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, loaders["train"])

    # BF16 mixed precision (no GradScaler needed for bf16)
    use_amp = device.type == "cuda"
    print(f"[AMP] BF16 mixed precision: {use_amp}")

    # Training loop
    print(f"\n[4/4] Training for {EPOCHS} epochs...")
    history = []
    best_map = 0.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{EPOCHS}")
        print(f"{'='*60}")

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, scheduler,
            device, epoch
        )

        print(f"  Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")

        epoch_log = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        # Evaluate
        if epoch % EVAL_EVERY == 0 or epoch == EPOCHS:
            metrics = evaluate(model, loaders, device, epoch=epoch,
                               save_dir=RESULT_DIR)
            epoch_log["eval"] = metrics
            current_map = metrics["in_domain"]["mAP"]
            if current_map > best_map:
                best_map = current_map
                model.save_lora(os.path.join(CKPT_DIR, "best"))
                print(f"  [BEST] New best mAP: {best_map:.4f}")

        # Save checkpoint
        if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:
            ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch}")
            model.save_lora(ckpt_path)

        history.append(epoch_log)

        # Save history
        with open(os.path.join(LOG_DIR, "history.json"), "w",
                  encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final evaluation...")
    print(f"{'='*60}")

    # Load best model
    best_path = os.path.join(CKPT_DIR, "best")
    if os.path.exists(best_path):
        model.load_lora(best_path)
        print("Loaded best checkpoint")

    final_metrics = evaluate(model, loaders, device, epoch=None,
                             save_dir=RESULT_DIR)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best in-domain mAP: {best_map:.4f}")
    print(f"Results: {RESULT_DIR}")
    print(f"Checkpoints: {CKPT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
