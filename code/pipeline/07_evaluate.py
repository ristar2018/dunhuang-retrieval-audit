# -*- coding: utf-8 -*-
"""
Step 7: 检索评估
提取query/gallery/cross-source embeddings，用faiss计算Recall@K和mAP。
支持两种模式：
  1. 同域评估: query -> gallery (同分布)
  2. 跨源评估: cross-source query -> gallery (跨数据源泛化)
"""
import os, sys, json, time
import numpy as np
import torch
import faiss
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FEATURE_DIM, TOP_K, RESULT_DIR, EXP_NAME, CKPT_DIR, CATEGORY_NAMES_CN
)


@torch.no_grad()
def extract_embeddings(model, dataloader, device):
    """提取embedding和标签"""
    model.eval()
    all_emb, all_labels, all_sources, all_paths = [], [], [], []
    for batch in tqdm(dataloader, desc="  Extracting", leave=False):
        imgs = batch["image"].to(device)
        emb = model.get_embeddings(imgs)
        all_emb.append(emb.cpu().numpy())
        all_labels.extend(batch["label"].numpy().tolist())
        all_sources.extend(batch["source"])
        all_paths.extend(batch["path"])
    embeddings = np.concatenate(all_emb, axis=0).astype(np.float32)
    # L2 normalize (redundant but safe)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    return embeddings, np.array(all_labels), all_sources, all_paths


def build_gallery_index(gallery_emb):
    """构建faiss内积索引（embedding已归一化，内积=余弦相似度）"""
    index = faiss.IndexFlatIP(gallery_emb.shape[1])
    index.add(gallery_emb)
    return index


def compute_metrics(query_emb, query_labels, gallery_emb, gallery_labels,
                    gallery_is_distractor, k_list=TOP_K):
    """计算 Recall@K 和 mAP"""
    index = build_gallery_index(gallery_emb)
    max_k = max(k_list)
    # search more than max_k to skip distractors
    search_k = min(max_k * 5, gallery_emb.shape[0])
    sims, indices = index.search(query_emb, search_k)

    results = {}
    # Recall@K considering only labeled gallery items as relevant
    gallery_labels = np.array(gallery_labels)
    gallery_is_distractor = np.array(gallery_is_distractor)

    for k in k_list:
        hits = 0
        ap_sum = 0.0
        for i in range(len(query_emb)):
            # Get top-k, skip self-matches (same path would have sim=1.0)
            topk_idx = indices[i, :k]
            topk_labels = gallery_labels[topk_idx]
            topk_dist = gallery_is_distractor[topk_idx]
            # Relevant = same label AND not distractor
            relevant = (topk_labels == query_labels[i]) & (~topk_dist)
            if relevant.any():
                hits += 1
            # AP for this query
            # Use full search results for AP
            full_idx = indices[i]
            full_labels = gallery_labels[full_idx]
            full_dist = gallery_is_distractor[full_idx]
            rel = (full_labels == query_labels[i]) & (~full_dist)
            if rel.any():
                # Filter out distractors from ranking for AP
                valid = ~full_dist
                if valid.sum() > 0:
                    rel_filtered = rel[valid]
                    precisions = np.cumsum(rel_filtered) / (np.arange(len(rel_filtered)) + 1)
                    ap_sum += (precisions * rel_filtered).sum() / max(rel_filtered.sum(), 1)

        results[f"Recall@{k}"] = hits / len(query_emb)

    results["mAP"] = ap_sum / len(query_emb)
    return results


def evaluate(model, dataloaders, device, epoch=None, save_dir=None):
    """完整评估流程"""
    model.eval()
    metrics = {}

    print("\n[EVAL] Extracting gallery embeddings...")
    gal_emb, gal_labels, gal_srcs, gal_paths = extract_embeddings(
        model, dataloaders["gallery"], device)
    gal_dist = []
    for i, row in enumerate(dataloaders["gallery"].dataset.rows):
        d = row.get("is_distractor", "False")
        gal_dist.append(d in ("True", "true", "1"))

    # In-domain evaluation
    print("[EVAL] In-domain: query -> gallery")
    q_emb, q_labels, q_srcs, q_paths = extract_embeddings(
        model, dataloaders["query"], device)
    in_domain = compute_metrics(q_emb, q_labels, gal_emb, gal_labels, gal_dist)
    metrics["in_domain"] = in_domain
    print(f"  " + " | ".join(f"{k}: {v:.4f}" for k, v in in_domain.items()))

    # Cross-source evaluation
    print("[EVAL] Cross-source: cross_source -> gallery")
    c_emb, c_labels, c_srcs, c_paths = extract_embeddings(
        model, dataloaders["cross"], device)
    cross = compute_metrics(c_emb, c_labels, gal_emb, gal_labels, gal_dist)
    metrics["cross_source"] = cross
    print(f"  " + " | ".join(f"{k}: {v:.4f}" for k, v in cross.items()))

    # Per-source cross-source breakdown
    metrics["cross_by_source"] = {}
    for src in set(c_srcs):
        mask = np.array(c_srcs) == src
        if mask.sum() > 0:
            src_metrics = compute_metrics(
                c_emb[mask], c_labels[mask], gal_emb, gal_labels, gal_dist)
            metrics["cross_by_source"][src] = src_metrics
            print(f"  [{src}] (n={mask.sum()}): " +
                  " | ".join(f"{k}: {v:.4f}" for k, v in src_metrics.items()))

    # Per-class in-domain Recall@1
    print("[EVAL] Per-class Recall@1 (in-domain):")
    q_labels_arr = np.array(q_labels)
    index = build_gallery_index(gal_emb)
    _, top1_indices = index.search(q_emb, 10)
    per_class = {}
    for cls_id in range(len(CATEGORY_NAMES_CN)):
        mask = q_labels_arr == cls_id
        if mask.sum() == 0:
            continue
        top1_labels = np.array(gal_labels)[top1_indices[mask, 0]]
        top1_dist = np.array(gal_dist)[top1_indices[mask, 0]]
        correct = ((top1_labels == cls_id) & (~top1_dist)).sum()
        r1 = correct / mask.sum()
        cls_name = list(CATEGORY_NAMES_CN.keys())[cls_id]
        per_class[cls_name] = {"Recall@1": float(r1), "n_query": int(mask.sum())}
    metrics["per_class_in_domain"] = per_class

    if epoch is not None:
        metrics["epoch"] = epoch

    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        suffix = f"_epoch{epoch}" if epoch is not None else "_final"
        path = os.path.join(save_dir, f"metrics{suffix}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] Metrics -> {path}")

    return metrics


if __name__ == "__main__":
    # Standalone test with frozen DINOv3 (zero-shot baseline)
    import torch.nn.functional as Fn
    from config import EXP_DIR
    from importlib import import_module
    mod = import_module("04_dataset")
    build_dataloaders = mod.build_dataloaders

    device = torch.device("cuda")
    print("Loading DINOv3 for zero-shot baseline evaluation...")

    class ZeroShotModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AutoModel.from_pretrained(DINOV3_PATH, torch_dtype=torch.float16)
        def get_embeddings(self, pixel_values):
            out = self.backbone(pixel_values=pixel_values.half())
            emb = out.last_hidden_state[:, 0, :]
            return Fn.normalize(emb.float(), p=2, dim=1)

    model = ZeroShotModel().to(device)
    loaders = build_dataloaders()
    metrics = evaluate(model, loaders, device, save_dir=os.path.join(EXP_DIR, "zeroshot"))
