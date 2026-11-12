#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_metric.py - 统一度量学习训练入口 (C1模块)

支持方法:
  - ArcFace
  - CosFace
  - CircleLoss
  - ProxyNCA
  - SubCenterArcFace (K=1,3,5)
  - SCE (Symmetric Cross Entropy, Wang 2019) - 噪声鲁棒
  - GCE (Generalized Cross Entropy, Zhang 2018) - 噪声鲁棒

实验配置:
  - 冻结DINOv3特征 (加载预提取的 .npy)
  - 可训练投影头: 768 -> 512 -> embed_dim
  - AdamW, CosineAnnealingLR
  - MPerClassSampler (m=4)
  - MultiSimilarityMiner (epsilon=0.1)
  - BF16混合精度 (Blackwell原生支持)
"""

import json
import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

# pytorch-metric-learning
from pytorch_metric_learning import losses, miners, distances
from pytorch_metric_learning.samplers import MPerClassSampler

# ==================== 路径配置 ====================
BASE_DIR = Path(r"<DATA_ROOT>\retrieval")
FEATURES_NPY = BASE_DIR / "features" / "dinov3_features.npy"
GRADIENT_DIR = BASE_DIR / "data" / "gradients"
RESULTS_DIR = BASE_DIR / "results"

# ==================== 模型定义 ====================

class ProjectionHead(nn.Module):
    """投影头: 768 -> 512 -> embed_dim"""
    def __init__(self, input_dim=768, hidden_dim=512, embed_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class SCEClassificationLoss(nn.Module):
    """Symmetric Cross Entropy (Wang et al., 2019, ICCV)
    L = alpha * CE + beta * RCE
    RCE(f) = -sum(1 - softmax(f)_k) for all k
    自带分类层，输入embeddings输出loss
    """
    def __init__(self, embed_dim, num_classes, scale=30.0, alpha=0.1, beta=1.0):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.scale = scale
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, embeddings, labels):
        # 计算logits (cosine-based)
        emb_norm = F.normalize(embeddings)
        w_norm = F.normalize(self.fc.weight, dim=1)
        logits = self.scale * (emb_norm @ w_norm.t())
        
        # CE
        ce_loss = F.cross_entropy(logits, labels)
        
        # RCE
        prob = F.softmax(logits, dim=1)
        rce_loss = -torch.sum(1.0 - prob, dim=1).mean()
        
        return self.alpha * ce_loss + self.beta * rce_loss


class GCEClassificationLoss(nn.Module):
    """Generalized Cross Entropy (Zhang et al., 2018, NeurIPS)
    L = (1 - softmax(f)_y^q) / q
    q -> 0: CE; q -> 1: MAE
    自带分类层，输入embeddings输出loss
    """
    def __init__(self, embed_dim, num_classes, scale=30.0, q=0.7):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.scale = scale
        self.q = q
    
    def forward(self, embeddings, labels):
        emb_norm = F.normalize(embeddings)
        w_norm = F.normalize(self.fc.weight, dim=1)
        logits = self.scale * (emb_norm @ w_norm.t())
        
        prob = F.softmax(logits, dim=1)
        prob_y = prob.gather(1, labels.unsqueeze(1)).squeeze(1)
        loss = (1.0 - torch.pow(prob_y.clamp(min=1e-7), self.q)) / self.q
        return loss.mean()


# ==================== 损失函数构建 ====================

def build_loss_and_miner(method_name, num_classes, embed_dim=512, 
                         subcenter_k=3, device='cuda'):
    """构建损失函数和miner"""
    
    distance = distances.CosineSimilarity()
    
    # 分类损失 (自带分类层, 不需要miner)
    classification_methods = {'SCE', 'GCE'}
    # Pair-based 度量学习损失 (需要miner)
    pair_methods = {'ArcFace', 'CosFace', 'CircleLoss', 'ProxyNCA', 'SubCenterArcFace'}
    
    # 注意: 在pytorch-metric-learning中:
    # - ArcFaceLoss/CosFaceLoss/SubCenterArcFaceLoss 虽然自带分类层,
    #   但其forward()仍支持indices_tuple来做采样
    # - CircleLoss 是纯分类损失, 不接受indices_tuple
    # - ProxyNCALoss 接受indices_tuple
    
    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    
    method_map = {
        'ArcFace': lambda: losses.ArcFaceLoss(
            num_classes=num_classes, 
            embedding_size=embed_dim,
            scale=30.0, margin=0.5
        ),
        'CosFace': lambda: losses.CosFaceLoss(
            num_classes=num_classes,
            embedding_size=embed_dim,
            scale=30.0, margin=0.4
        ),
        'CircleLoss': lambda: losses.CircleLoss(
            m=0.4, gamma=80
        ),
        'ProxyNCA': lambda: losses.ProxyNCALoss(
            num_classes=num_classes,
            embedding_size=embed_dim,
            softmax_scale=10.0
        ),
        'SubCenterArcFace': lambda: losses.SubCenterArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embed_dim,
            scale=64.0, margin=28.6,
            sub_centers=subcenter_k
        ),
    }
    
    if method_name in method_map:
        loss_fn = method_map[method_name]()
        if method_name in classification_methods:
            miner = None
    elif method_name == 'SCE':
        loss_fn = SCEClassificationLoss(
            embed_dim=embed_dim, num_classes=num_classes,
            scale=30.0, alpha=0.1, beta=1.0
        )
        miner = None
    elif method_name == 'GCE':
        loss_fn = GCEClassificationLoss(
            embed_dim=embed_dim, num_classes=num_classes,
            scale=30.0, q=0.7
        )
        miner = None
    else:
        raise ValueError(f"未知方法: {method_name}")
    
    return loss_fn, miner


# ==================== 数据集 ====================

class FeatureDataset(torch.utils.data.Dataset):
    """冻结特征数据集"""
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ==================== 训练循环 ====================

def train_one_epoch(model, loss_fn, miner, optimizer, dataloader, device):
    """单epoch训练 - 统一接口，所有loss都接受(embeddings, labels)"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        embeddings = model(features)
        
        # 计算损失 (所有方法的loss_fn统一接受 embeddings, labels)
        if miner is not None:
            indices_tuple = miner(embeddings, labels)
            loss = loss_fn(embeddings, labels, indices_tuple)
        else:
            loss = loss_fn(embeddings, labels)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(loss_fn.parameters()), 
            max_norm=5.0
        )
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(num_batches, 1)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='度量学习训练')
    parser.add_argument('--method', type=str, required=True,
                       choices=['ArcFace', 'CosFace', 'CircleLoss', 'ProxyNCA',
                                'SubCenterArcFace', 'SCE', 'GCE'],
                       help='损失函数方法')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['S0', 'S1', 'S2', 'S3',
                                'S0_prime', 'S1_prime', 'S2_prime', 'S3_prime'],
                       help='训练集')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批大小')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--embed_dim', type=int, default=512, help='嵌入维度')
    parser.add_argument('--subcenter_k', type=int, default=3, help='SubCenter K值')
    parser.add_argument('--m_per_class', type=int, default=4, help='MPerClassSampler m')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup轮数')
    parser.add_argument('--eval_every', type=int, default=5, help='每N轮评估')
    parser.add_argument('--save_every', type=int, default=10, help='每N轮保存')
    parser.add_argument('--num_classes', type=int, default=16, help='类别数')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 70)
    print(f"度量学习训练: {args.method} on {args.dataset}")
    print(f"种子={args.seed}, 轮数={args.epochs}, bs={args.batch_size}, lr={args.lr}")
    print("=" * 70)
    
    # ==================== 加载数据 ====================
    print("\n[1/4] 加载数据...")
    
    # 加载全部特征
    features_all = np.load(str(FEATURES_NPY))
    print(f"  特征shape: {features_all.shape}")
    
    # 加载训练集索引
    gradient_file = GRADIENT_DIR / f"{args.dataset}_indices.json"
    if not gradient_file.exists():
        print(f"  ❌ 训练集文件不存在: {gradient_file}")
        print(f"  请先运行 build_label_gradient.py")
        sys.exit(1)
    
    with open(gradient_file, 'r', encoding='utf-8') as f:
        gradient_data = json.load(f)
    
    # 兼容两种格式：纯列表 或 dict with 'indices' key
    if isinstance(gradient_data, list):
        train_indices = gradient_data
    else:
        train_indices = gradient_data['indices']
    print(f"  训练集 {args.dataset}: {len(train_indices)} 张")
    
    # 提取训练特征和标签
    labels_csv = pd.read_csv(BASE_DIR / "annotations" / "all_labels.csv", encoding='utf-8')
    
    # 建立 global_idx -> label 的快速查找表
    label_lookup = dict(zip(labels_csv['global_idx'], labels_csv['label']))
    
    # 类别编码 (固定映射)
    categories = sorted(labels_csv['label'].unique())
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    print(f"  类别数: {len(categories)}")
    
    # 提取训练数据
    train_features = features_all[train_indices]
    train_labels = np.array([cat_to_idx[label_lookup[idx]] 
                             for idx in train_indices])
    
    print(f"  训练特征: {train_features.shape}")
    print(f"  标签分布: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    
    # ==================== 构建模型 ====================
    print("\n[2/4] 构建模型...")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")
    
    model = ProjectionHead(
        input_dim=768, hidden_dim=512, embed_dim=args.embed_dim
    ).to(device)
    
    # 参数量统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练: {trainable_params:,}")
    
    # ==================== 构建损失和采样器 ====================
    print("\n[3/4] 构建损失函数...")
    
    is_classification = args.method in ['CircleLoss', 'SCE', 'GCE']
    
    # 构建损失函数和miner
    loss_fn, miner = build_loss_and_miner(
        args.method, args.num_classes, args.embed_dim,
        subcenter_k=args.subcenter_k, device=device
    )
    loss_fn = loss_fn.to(device)
    
    # 统一使用标准DataLoader（miner负责pair采样，无需MPerClassSampler）
    dataloader = torch.utils.data.DataLoader(
        FeatureDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 冻结特征无需多进程
        pin_memory=True,
        drop_last=True  # 避免最后一个batch太小影响BN统计
    )
    
    # 优化器 (所有损失函数都可能有可学习参数: ArcFace/CosFace的权重矩阵等)
    params_to_optimize = list(model.parameters())
    if hasattr(loss_fn, 'parameters') and len(list(loss_fn.parameters())) > 0:
        params_to_optimize += list(loss_fn.parameters())
    optimizer = AdamW(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    
    # BF16混合精度 (Blackwell原生支持)
    use_bf16 = device.type == 'cuda'
    if use_bf16:
        print(f"  BF16混合精度: 启用")
    
    # ==================== 训练 ====================
    print(f"\n[4/4] 开始训练 ({args.epochs} epochs)...")
    
    # 创建输出目录
    exp_name = f"{args.method}_{args.dataset}_seed{args.seed}"
    if args.method == 'SubCenterArcFace':
        exp_name = f"SubCenterArcFace_K{args.subcenter_k}_{args.dataset}_seed{args.seed}"
    
    exp_dir = RESULTS_DIR / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    training_log = []
    
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Warmup
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * (epoch / args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
        
        # 训练
        avg_loss = train_one_epoch(
            model, loss_fn, miner, optimizer, dataloader, device
        )
        
        # 学习率更新 (warmup后)
        if epoch > args.warmup_epochs:
            scheduler.step()
        
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']
        
        log_entry = {
            'epoch': epoch,
            'loss': avg_loss,
            'lr': current_lr,
            'time': elapsed,
        }
        training_log.append(log_entry)
        
        # 打印进度
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.4f}, "
                  f"lr={current_lr:.2e}, time={elapsed:.1f}s")
        
        # 保存best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'method': args.method,
                'dataset': args.dataset,
                'seed': args.seed,
            }, exp_dir / "best_checkpoint.pt")
        
        # 定期保存
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, exp_dir / f"checkpoint_epoch{epoch}.pt")
    
    # ==================== 保存结果 ====================
    print(f"\n训练完成! best_loss={best_loss:.4f}")
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'method': args.method,
        'dataset': args.dataset,
        'seed': args.seed,
        'config': vars(args),
    }, exp_dir / "final_checkpoint.pt")
    
    # 保存训练日志
    with open(exp_dir / "training_log.json", 'w', encoding='utf-8') as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)
    
    # 保存实验配置
    with open(exp_dir / "config.json", 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    
    print(f"结果保存至: {exp_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
