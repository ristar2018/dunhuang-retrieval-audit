#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_retrieval.py - 检索评估模块 (C2)

评估指标:
  - Recall@K (K=1, 5, 10, 20)
  - MAP (Mean Average Precision)
  - NDCG@10

评估协议:
  - Query集: 只用L0标签 (金标准)
  - Gallery集: 包含所有L0-L3样本 (模拟真实检索)
  - 冻结DINOv3特征 + 训练好的投影头 → 嵌入空间
  - SubCenterArcFace: 用子中心max相似度

用法:
  python eval_retrieval.py --checkpoint results/xxx/best_checkpoint.pt --query_split L0 --gallery_split all
  python eval_retrieval.py --experiment_dir results/ --eval_all  # 评估所有实验
"""

import json
import sys
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

# ==================== 路径 ====================
BASE_DIR = Path(r"<DATA_ROOT>\retrieval")
FEATURES_NPY = BASE_DIR / "features" / "dinov3_features.npy"
LABELS_CSV = BASE_DIR / "annotations" / "all_labels.csv"
SPLIT_DIR = BASE_DIR / "splits"  # 4个CSV: train.csv, query.csv, gallery.csv, cross_source_test.csv


# ==================== 投影头 (需与训练时一致) ====================

class ProjectionHead(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, embed_dim=512, dropout=0.1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, embed_dim),
            torch.nn.BatchNorm1d(embed_dim),
        )
    
    def forward(self, x):
        return self.net(x)


# ==================== 检索核心 ====================

def compute_retrieval_metrics(query_embeddings, query_labels,
                               gallery_embeddings, gallery_labels,
                               k_values=[1, 5, 10, 20]):
    """计算检索指标
    
    Args:
        query_embeddings: (Nq, D) L2-normalized
        query_labels: (Nq,) 类别索引
        gallery_embeddings: (Ng, D) L2-normalized
        gallery_labels: (Ng,) 类别索引
        k_values: 评估的K值列表
    
    Returns:
        dict: {metric_name: value}
    """
    # 计算相似度矩阵
    sim_matrix = query_embeddings @ gallery_embeddings.T  # (Nq, Ng)
    
    # 排序 (降序)
    sorted_indices = np.argsort(-sim_matrix, axis=1)  # (Nq, Ng)
    
    Nq = len(query_labels)
    max_k = max(k_values)
    
    # ---- Recall@K ----
    recalls = {}
    for k in k_values:
        top_k_indices = sorted_indices[:, :k]
        # 判断top-k中是否有与query相同类别的gallery样本
        top_k_labels = gallery_labels[top_k_indices]  # (Nq, k)
        correct = (top_k_labels == query_labels[:, None]).any(axis=1)
        recalls[f'Recall@{k}'] = float(correct.mean())
    
    # ---- AP (Average Precision) for each query ----
    aps = []
    for q_idx in range(Nq):
        true_label = query_labels[q_idx]
        sorted_labels = gallery_labels[sorted_indices[q_idx]]
        
        # 判断每个位置是否正确
        is_relevant = (sorted_labels == true_label).astype(float)
        
        # 计算AP
        cumsum = np.cumsum(is_relevant)
        precision_at_k = cumsum / (np.arange(len(is_relevant)) + 1)
        ap = (precision_at_k * is_relevant).sum() / max(is_relevant.sum(), 1)
        aps.append(ap)
    
    map_score = float(np.mean(aps))
    
    # ---- NDCG@10 ----
    dcg_at_10_list = []
    idcg_at_10_list = []
    
    for q_idx in range(Nq):
        true_label = query_labels[q_idx]
        sorted_labels = gallery_labels[sorted_indices[q_idx]]
        
        # 相关性判断
        is_relevant = (sorted_labels == true_label).astype(float)
        
        # DCG@10
        k = min(10, len(is_relevant))
        discounts = np.log2(np.arange(2, k + 2))
        dcg = (is_relevant[:k] / discounts).sum()
        
        # IDCG@10 (ideal DCG)
        n_relevant = int(is_relevant.sum())
        ideal_relevant = np.zeros_like(is_relevant)
        ideal_relevant[:min(n_relevant, k)] = 1.0
        idcg = (ideal_relevant[:k] / discounts).sum()
        
        dcg_at_10_list.append(dcg)
        idcg_at_10_list.append(max(idcg, 1e-8))
    
    ndcg_at_10 = float(np.mean([d/i for d, i in zip(dcg_at_10_list, idcg_at_10_list)]))
    
    return {
        **recalls,
        'MAP': map_score,
        'NDCG@10': ndcg_at_10,
    }


def extract_embeddings(model, features, device, batch_size=256):
    """用投影头提取嵌入向量"""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start+batch_size]).float().to(device)
            emb = model(batch)
            emb = F.normalize(emb, dim=1)
            embeddings.append(emb.cpu().numpy())
    
    return np.vstack(embeddings)


# ==================== 数据加载 ====================

def load_evaluation_data(features_all, labels_df, split_data,
                         query_label_type='L0', gallery_label_type='all'):
    """加载评估数据
    
    Args:
        query_label_type: Query标签类型 ('L0' only, 或 'all')
        gallery_label_type: Gallery标签类型 ('all' = L0-L3, 'L0_only' 等)
    """
    # split_data 是 dict: {'train': set, 'query': set, 'gallery': set, 'cross': set}
    query_indices = set(split_data.get('query', []))
    gallery_indices = set(split_data.get('gallery', []))
    cross_indices = set(split_data.get('cross', []))
    
    # 类别编码 (列名是'label')
    categories = sorted(labels_df['label'].unique())
    cat_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    # 构建 global_idx → DataFrame行号 的映射
    idx_to_row = dict(zip(labels_df['global_idx'], labels_df.index))
    # 构建 global_idx → label 的映射
    idx_to_label = dict(zip(labels_df['global_idx'], labels_df['label']))
    
    # Query: 默认只用L0 (seed)
    if query_label_type == 'L0':
        query_global = [g for g in query_indices if g in idx_to_row]
        query_rows = [idx_to_row[g] for g in query_global]
        query_mask = labels_df.iloc[query_rows]['label_type'] == 'seed'
        query_rows = [r for r, keep in zip(query_rows, query_mask) if keep]
    else:
        query_global = [g for g in query_indices if g in idx_to_row]
        query_rows = [idx_to_row[g] for g in query_global]
    
    query_idx = labels_df.iloc[query_rows]['global_idx'].tolist()
    query_labels = np.array([cat_to_idx[idx_to_label[g]] for g in query_idx])
    # features_all 按 global_idx 索引
    query_features = features_all[query_idx]
    
    print(f"  Query: {len(query_idx)} (label_type={query_label_type})")
    print(f"  Query类别分布: {dict(zip(*np.unique(query_labels, return_counts=True)))}")
    
    # Gallery: 所有非query的样本 (或指定标签范围)
    all_gallery_global = gallery_indices | cross_indices
    if gallery_label_type == 'L0_L3':
        valid_types = {'seed', 'seed_knn', 'expanded_knn'}
        all_gallery_global = {g for g in all_gallery_global 
                             if g in idx_to_row and labels_df.loc[idx_to_row[g], 'label_type'] in valid_types}
    
    gallery_global = [g for g in all_gallery_global if g in idx_to_row and g not in set(query_idx)]
    gallery_rows = [idx_to_row[g] for g in gallery_global]
    
    gallery_idx = labels_df.iloc[gallery_rows]['global_idx'].tolist()
    gallery_labels = np.array([cat_to_idx[idx_to_label[g]] for g in gallery_idx])
    gallery_features = features_all[gallery_idx]
    
    print(f"  Gallery: {len(gallery_idx)} (label_type={gallery_label_type})")
    
    return (query_features, query_labels, query_idx,
            gallery_features, gallery_labels, gallery_idx)


# ==================== 主函数 ====================

def evaluate_checkpoint(checkpoint_path, features_all, labels_df, split_data,
                        device='cuda'):
    """评估单个checkpoint"""
    print(f"\n  加载checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 重建模型
    config = ckpt.get('config', {})
    embed_dim = config.get('embed_dim', 512)
    model = ProjectionHead(input_dim=768, hidden_dim=512, embed_dim=embed_dim).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    # 加载评估数据
    q_feat, q_labels, q_idx, g_feat, g_labels, g_idx = load_evaluation_data(
        features_all, labels_df, split_data
    )
    
    # 提取嵌入
    print("  提取嵌入...")
    q_emb = extract_embeddings(model, q_feat, device)
    g_emb = extract_embeddings(model, g_feat, device)
    
    # 计算指标
    print("  计算检索指标...")
    metrics = compute_retrieval_metrics(q_emb, q_labels, g_emb, g_labels)
    
    print(f"  结果: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='检索评估')
    parser.add_argument('--checkpoint', type=str, help='单个checkpoint路径')
    parser.add_argument('--experiment_dir', type=str, help='实验目录(评估所有)')
    parser.add_argument('--eval_all', action='store_true', help='评估目录下所有实验')
    parser.add_argument('--output', type=str, help='输出JSON路径')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    # 加载全局数据
    print("[1/3] 加载全局数据...")
    features_all = np.load(str(FEATURES_NPY))
    labels_df = pd.read_csv(LABELS_CSV, encoding='utf-8')
    
    # 从4个CSV加载数据划分
    split_data = {}
    for split_name, csv_name in [('train', 'train.csv'), ('query', 'query.csv'),
                                  ('gallery', 'gallery.csv'), ('cross', 'cross_source_test.csv')]:
        csv_path = SPLIT_DIR / csv_name
        df_split = pd.read_csv(csv_path, encoding='utf-8')
        split_data[split_name] = set(df_split['global_idx'].tolist())
    print(f"  数据划分: train={len(split_data['train'])}, query={len(split_data['query'])}, "
          f"gallery={len(split_data['gallery'])}, cross={len(split_data['cross'])}")
    
    all_results = {}
    
    if args.checkpoint:
        # 评估单个checkpoint
        print(f"\n[2/3] 评估: {args.checkpoint}")
        metrics = evaluate_checkpoint(
            args.checkpoint, features_all, labels_df, split_data, device
        )
        exp_name = Path(args.checkpoint).parent.name
        all_results[exp_name] = metrics
    
    elif args.experiment_dir and args.eval_all:
        # 评估目录下所有实验
        exp_dir = Path(args.experiment_dir)
        checkpoints = list(exp_dir.glob('*/best_checkpoint.pt'))
        print(f"\n[2/3] 找到 {len(checkpoints)} 个实验")
        
        for ckpt_path in sorted(checkpoints):
            exp_name = ckpt_path.parent.name
            print(f"\n--- 评估: {exp_name} ---")
            try:
                metrics = evaluate_checkpoint(
                    str(ckpt_path), features_all, labels_df, split_data, device
                )
                all_results[exp_name] = metrics
            except Exception as e:
                print(f"  ❌ 失败: {e}")
                all_results[exp_name] = {'error': str(e)}
    
    else:
        print("请指定 --checkpoint 或 --experiment_dir --eval_all")
        sys.exit(1)
    
    # 保存结果
    print(f"\n[3/3] 保存结果...")
    output_path = args.output
    if output_path is None:
        if args.checkpoint:
            output_path = str(Path(args.checkpoint).parent / "eval_results.json")
        else:
            output_path = str(Path(args.experiment_dir) / "all_eval_results.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"结果保存至: {output_path}")
    
    # 打印汇总表
    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    header = f"{'实验名':<50s} {'R@1':>6s} {'R@5':>6s} {'R@10':>6s} {'MAP':>6s} {'NDCG':>6s}"
    print(header)
    print("-" * 80)
    
    for exp_name, metrics in sorted(all_results.items()):
        if 'error' in metrics:
            print(f"{exp_name:<50s} ERROR: {metrics['error']}")
        else:
            print(f"{exp_name:<50s} "
                  f"{metrics.get('Recall@1', 0):>6.4f} "
                  f"{metrics.get('Recall@5', 0):>6.4f} "
                  f"{metrics.get('Recall@10', 0):>6.4f} "
                  f"{metrics.get('MAP', 0):>6.4f} "
                  f"{metrics.get('NDCG@10', 0):>6.4f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
