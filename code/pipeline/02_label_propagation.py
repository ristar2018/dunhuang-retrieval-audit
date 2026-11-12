# -*- coding: utf-8 -*-
"""
Step 2: 标签传播
以 hf_dunhuang 的 16 类精确语义标签为种子，
在 DINOv2 特征空间中通过 k-NN 投票将标签传播到全部 32,415 张图像。

策略:
  - 种子集: hf_dunhuang 2,278 张 (metadata.csv 中的 16 类标签)
  - 方法: faiss 精确内积搜索 (特征已 L2 归一化，内积=余弦相似度)
  - k=7 近邻加权投票 (权重=余弦相似度)
  - 置信度 = top-1 类别加权票数 / 总加权票数
  - >=0.60: 可靠标签; <0.60: uncertain
  - 对 muraldh_sr 超分辨率变体做特殊处理: 同原图标签一致

输出:
  annotations/seed_labels.csv  — 种子集标签
  annotations/all_labels.csv   — 全部图像传播标签
  annotations/label_statistics.json — 统计信息
"""
import os, sys, json, csv, time
import numpy as np
import faiss
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FEATURE_NPY, FEATURE_INDEX, HF_CSV, HF_IMG_DIR,
    SEED_LABELS, ALL_LABELS, LABEL_STATS,
    CATEGORY_NAMES, CATEGORY_NAMES_CN,
    KNN_K, CONF_THRESHOLD, N_CLUSTERS,
    ANNO_DIR
)


def load_features():
    """加载 Step 1 产出的特征矩阵和索引"""
    print(f"[INFO] 加载特征矩阵: {FEATURE_NPY}")
    feats = np.load(FEATURE_NPY)
    with open(FEATURE_INDEX, "r", encoding="utf-8") as f:
        idx = json.load(f)
    print(f"  特征矩阵: {feats.shape}, 已归一化: {idx.get('normalized', False)}")
    print(f"  索引记录数: {idx['num_images']}")
    return feats, idx


def load_seed_labels(feature_index):
    """
    从 hf_dunhuang/metadata.csv 加载种子标签。
    需要将文件名映射到 feature_index 中的全局索引。
    metadata.csv 的 file 字段如 'animals_00000.jpg'，
    对应路径为 data/raw/dunhuang_hf/images/{label_name}/{file}
    """
    print(f"[INFO] 加载种子标签: {HF_CSV}")

    # 构建 path -> global_index 映射
    path_to_idx = {}
    for rec in feature_index["records"]:
        path_to_idx[rec["path"]] = rec["index"]

    seeds = []
    missing = 0

    with open(HF_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["file"]
            label = row["label_name"]
            # 构造完整路径
            full_path = os.path.join(HF_IMG_DIR, label, fname)
            full_path_norm = os.path.normpath(full_path)

            if full_path_norm in path_to_idx:
                global_idx = path_to_idx[full_path_norm]
                seeds.append({
                    "global_idx": global_idx,
                    "path": full_path_norm,
                    "label": label,
                    "label_cn": CATEGORY_NAMES_CN.get(label, label),
                    "source": "hf_dunhuang",
                    "is_seed": True,
                })
            else:
                # 尝试大小写不敏感匹配
                found = False
                for p, gi in path_to_idx.items():
                    if p.lower() == full_path_norm.lower():
                        seeds.append({
                            "global_idx": gi,
                            "path": p,
                            "label": label,
                            "label_cn": CATEGORY_NAMES_CN.get(label, label),
                            "source": "hf_dunhuang",
                            "is_seed": True,
                        })
                        found = True
                        break
                if not found:
                    missing += 1

    print(f"  种子标签数: {len(seeds)}, 未匹配: {missing}")

    # 按类别统计
    label_counts = Counter(s["label"] for s in seeds)
    print(f"  类别数: {len(label_counts)}")
    for cat in CATEGORY_NAMES:
        cnt = label_counts.get(cat, 0)
        cn = CATEGORY_NAMES_CN.get(cat, cat)
        print(f"    {cat:20s} ({cn:6s}): {cnt:4d}")

    return seeds


def build_faiss_index(feats):
    """构建 faiss 内积索引 (特征已 L2 归一化)"""
    dim = feats.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(feats.astype(np.float32))
    print(f"[INFO] faiss 索引构建完成: {index.ntotal} vectors, dim={dim}")
    return index


def knn_propagate(feats, index, seeds, k=KNN_K, threshold=CONF_THRESHOLD):
    """
    k-NN 加权投票标签传播。
    对每张非种子图像，找 k 个最近邻（优先从种子集中搜索），
    但为了让种子标签在全局传播，我们对所有图搜索 k 近邻，
    其中种子标签的近邻投票更有分量。

    改进策略: 两阶段传播
      Stage 1: 只使用种子标签做 k-NN (k=7)，得到初始传播
      Stage 2: 对高置信度(>=0.7)的传播结果，纳入扩展种子集再做一轮
               但为了防止错误扩散，Stage 2 只用于标注 uncertain 的图
    """
    n_total = feats.shape[0]
    seed_indices = set(s["global_idx"] for s in seeds)
    seed_label_map = {s["global_idx"]: s["label"] for s in seeds}

    # 将种子特征单独建索引用于 Stage 1
    seed_feats = np.array([feats[s["global_idx"]] for s in seeds], dtype=np.float32)
    seed_index = faiss.IndexFlatIP(feats.shape[1])
    seed_index.add(seed_feats)
    seed_idx_list = [s["global_idx"] for s in seeds]

    print(f"\n[INFO] Stage 1: 种子集 k-NN 传播 (k={k})")
    t0 = time.time()

    # 对所有图像搜索 k 个最近的种子
    # 分批搜索避免内存峰值
    batch_size = 4096
    all_labels = [""] * n_total
    all_conf = np.zeros(n_total, dtype=np.float32)
    all_votes = [""] * n_total  # 投票详情 JSON 字符串
    all_source_type = [""] * n_total

    # 种子图像直接赋标签
    for s in seeds:
        all_labels[s["global_idx"]] = s["label"]
        all_conf[s["global_idx"]] = 1.0
        all_source_type[s["global_idx"]] = "seed"

    # 对非种子图像做 k-NN
    non_seed_mask = np.ones(n_total, dtype=bool)
    for si in seed_indices:
        non_seed_mask[si] = False
    non_seed_indices = np.where(non_seed_mask)[0]
    print(f"  待传播图像: {len(non_seed_indices)}")

    for start in range(0, len(non_seed_indices), batch_size):
        end = min(start + batch_size, len(non_seed_indices))
        batch_global = non_seed_indices[start:end]
        queries = feats[batch_global].astype(np.float32)

        # 搜索 k+1 因为可能搜到自己（但自己不在种子集中所以不需要）
        D, I = seed_index.search(queries, k)

        for local_i in range(len(batch_global)):
            gi = batch_global[local_i]
            sims = D[local_i]  # 余弦相似度
            nn_seed_positions = I[local_i]  # 在 seed_idx_list 中的位置

            # 加权投票
            vote_weights = defaultdict(float)
            vote_details = []
            for rank, (sim, sp) in enumerate(zip(sims, nn_seed_positions)):
                if sp < 0:
                    continue
                seed_gi = seed_idx_list[sp]
                label = seed_label_map[seed_gi]
                w = float(max(sim, 0))  # 相似度作为权重
                vote_weights[label] += w
                vote_details.append(f"{label}:{w:.3f}")

            if vote_weights:
                sorted_votes = sorted(vote_weights.items(), key=lambda x: -x[1])
                top_label, top_weight = sorted_votes[0]
                total_weight = sum(vote_weights.values())
                conf = top_weight / total_weight if total_weight > 0 else 0

                all_labels[gi] = top_label
                all_conf[gi] = conf
                all_votes[gi] = "; ".join(vote_details)
                all_source_type[gi] = "seed_knn" if conf >= threshold else "uncertain"
            else:
                all_labels[gi] = "uncertain"
                all_conf[gi] = 0.0
                all_votes[gi] = ""
                all_source_type[gi] = "uncertain"

        if (start // batch_size) % 2 == 0 or end == len(non_seed_indices):
            elapsed = time.time() - t0
            done = end
            print(f"  [{done}/{len(non_seed_indices)}] {elapsed:.1f}s")

    elapsed1 = time.time() - t0
    reliable = sum(1 for st in all_source_type if st == "seed_knn")
    uncertain_count = sum(1 for st in all_source_type if st == "uncertain")
    seed_count = sum(1 for st in all_source_type if st == "seed")
    print(f"\n  Stage 1 完成 ({elapsed1:.1f}s):")
    print(f"    种子: {seed_count}, 可靠传播: {reliable}, uncertain: {uncertain_count}")

    # Stage 2: 用高置信度传播结果扩展种子集，对 uncertain 图再做一轮
    print(f"\n[INFO] Stage 2: 高置信度扩展种子集，重新标注 uncertain")
    high_conf_mask = (all_conf >= 0.75) & (np.array(all_source_type) == "seed_knn")
    high_conf_indices = np.where(high_conf_mask)[0]
    print(f"  高置信度(>=0.75)扩展种子: {len(high_conf_indices)}")

    if len(high_conf_indices) > 0 and uncertain_count > 0:
        # 扩展种子索引 = 原始种子 + 高置信度传播
        ext_seed_feats = np.vstack([
            seed_feats,
            feats[high_conf_indices].astype(np.float32)
        ])
        ext_seed_labels = [seed_label_map[seed_idx_list[i]] for i in range(len(seeds))]
        ext_seed_labels += [all_labels[gi] for gi in high_conf_indices]
        ext_seed_origins = (["original"] * len(seeds) +
                           ["propagated"] * len(high_conf_indices))

        ext_index = faiss.IndexFlatIP(feats.shape[1])
        ext_index.add(ext_seed_feats)

        uncertain_indices = np.where(
            np.array(all_source_type) == "uncertain"
        )[0]
        print(f"  待重新标注 uncertain: {len(uncertain_indices)}")

        upgraded = 0
        for start in range(0, len(uncertain_indices), batch_size):
            end = min(start + batch_size, len(uncertain_indices))
            batch_global = uncertain_indices[start:end]
            queries = feats[batch_global].astype(np.float32)
            D, I = ext_index.search(queries, k)

            for local_i in range(len(batch_global)):
                gi = batch_global[local_i]
                sims = D[local_i]
                nn_positions = I[local_i]

                vote_weights = defaultdict(float)
                vote_details = []
                n_orig = 0  # 原始种子投票数
                for sim, pos in zip(sims, nn_positions):
                    if pos < 0:
                        continue
                    label = ext_seed_labels[pos]
                    origin = ext_seed_origins[pos]
                    w = float(max(sim, 0))
                    if origin == "original":
                        w *= 1.5  # 原始种子权重加成
                        n_orig += 1
                    vote_weights[label] += w
                    vote_details.append(f"{label}({origin[0]}):{w:.3f}")

                if vote_weights:
                    sorted_votes = sorted(vote_weights.items(), key=lambda x: -x[1])
                    top_label, top_weight = sorted_votes[0]
                    total_weight = sum(vote_weights.values())
                    conf = top_weight / total_weight if total_weight > 0 else 0

                    # 只有置信度提升且至少有一个原始种子投票时才升级
                    if conf >= threshold and n_orig >= 2 and conf > all_conf[gi]:
                        all_labels[gi] = top_label
                        all_conf[gi] = conf
                        all_votes[gi] = "; ".join(vote_details)
                        all_source_type[gi] = "expanded_knn"
                        upgraded += 1

        print(f"  Stage 2 升级: {upgraded} / {uncertain_count}")

    # 最终统计
    final_reliable = sum(1 for st in all_source_type
                        if st in ("seed", "seed_knn", "expanded_knn"))
    final_uncertain = sum(1 for st in all_source_type if st == "uncertain")
    print(f"\n[FINAL] 可靠标签: {final_reliable} "
          f"({final_reliable/n_total*100:.1f}%), "
          f"uncertain: {final_uncertain} "
          f"({final_uncertain/n_total*100:.1f}%)")

    return all_labels, all_conf, all_votes, all_source_type


def save_results(feats, feature_index, seeds,
                 all_labels, all_conf, all_votes, all_source_type):
    """保存种子标签和全量标签 CSV"""
    os.makedirs(ANNO_DIR, exist_ok=True)

    # --- seed_labels.csv ---
    seed_path_set = set(s["path"] for s in seeds)
    with open(SEED_LABELS, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["global_idx", "path", "source",
                         "label", "label_cn", "label_type"])
        for s in seeds:
            writer.writerow([
                s["global_idx"], s["path"], s["source"],
                s["label"], s["label_cn"], "seed"
            ])
    print(f"[SAVE] 种子标签 -> {SEED_LABELS} ({len(seeds)} rows)")

    # --- all_labels.csv ---
    records = feature_index["records"]
    with open(ALL_LABELS, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "global_idx", "path", "source",
            "label", "label_cn", "confidence",
            "label_type", "vote_details"
        ])
        for rec in records:
            gi = rec["index"]
            label = all_labels[gi] if gi < len(all_labels) else "uncertain"
            conf = float(all_conf[gi]) if gi < len(all_conf) else 0.0
            votes = all_votes[gi] if gi < len(all_votes) else ""
            stype = all_source_type[gi] if gi < len(all_source_type) else "uncertain"
            writer.writerow([
                gi, rec["path"], rec["source"],
                label, CATEGORY_NAMES_CN.get(label, label),
                f"{conf:.4f}", stype, votes
            ])
    print(f"[SAVE] 全量标签 -> {ALL_LABELS} ({len(records)} rows)")

    # --- label_statistics.json ---
    stats = {
        "total_images": len(records),
        "seed_count": len(seeds),
        "seed_categories": len(CATEGORY_NAMES),
        "knn_k": KNN_K,
        "confidence_threshold": CONF_THRESHOLD,
        "feature_model": "dinov2-base",
        "feature_dim": 768,
    }

    # 按 source 统计
    by_source = defaultdict(lambda: {"total": 0, "labeled": 0, "uncertain": 0})
    for rec in records:
        gi = rec["index"]
        src = rec["source"]
        by_source[src]["total"] += 1
        if all_source_type[gi] == "uncertain":
            by_source[src]["uncertain"] += 1
        else:
            by_source[src]["labeled"] += 1
    stats["by_source"] = dict(by_source)

    # 按 label 统计
    by_label = Counter()
    for gi in range(len(all_labels)):
        by_label[all_labels[gi]] += 1
    stats["by_label"] = dict(by_label.most_common())

    # 按 label_type 统计
    by_type = Counter(all_source_type)
    stats["by_label_type"] = dict(by_type)

    # 置信度分布
    conf_arr = np.array(all_conf)
    stats["confidence_distribution"] = {
        "mean": float(np.mean(conf_arr)),
        "median": float(np.median(conf_arr)),
        "std": float(np.std(conf_arr)),
        "min": float(np.min(conf_arr)),
        "max": float(np.max(conf_arr)),
        "pct_ge_0.9": float(np.mean(conf_arr >= 0.9)),
        "pct_ge_0.8": float(np.mean(conf_arr >= 0.8)),
        "pct_ge_0.7": float(np.mean(conf_arr >= 0.7)),
        "pct_ge_0.6": float(np.mean(conf_arr >= 0.6)),
    }

    with open(LABEL_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] 统计信息 -> {LABEL_STATS}")

    return stats


def print_summary(stats):
    """打印最终汇总"""
    print("\n" + "=" * 60)
    print("Step 2 标签传播 — 结果汇总")
    print("=" * 60)

    print(f"\n总图像数: {stats['total_images']}")
    print(f"种子标签: {stats['seed_count']} ({stats['seed_categories']} 类)")
    print(f"k-NN 参数: k={stats['knn_k']}, 阈值={stats['confidence_threshold']}")

    print(f"\n--- 标注类型分布 ---")
    for t, c in sorted(stats["by_label_type"].items(),
                       key=lambda x: -x[1]):
        pct = c / stats["total_images"] * 100
        print(f"  {t:15s}: {c:6d} ({pct:5.1f}%)")

    print(f"\n--- 各数据源标注情况 ---")
    for src, info in sorted(stats["by_source"].items(),
                            key=lambda x: -x[1]["total"]):
        rate = info["labeled"] / info["total"] * 100 if info["total"] > 0 else 0
        print(f"  {src:20s}: {info['total']:6d} total, "
              f"{info['labeled']:6d} labeled ({rate:5.1f}%), "
              f"{info['uncertain']:5d} uncertain")

    print(f"\n--- 类别分布 (top 20) ---")
    for label, cnt in list(stats["by_label"].items())[:20]:
        cn = CATEGORY_NAMES_CN.get(label, label)
        print(f"  {label:20s} ({cn:8s}): {cnt:6d}")

    cd = stats["confidence_distribution"]
    print(f"\n--- 置信度分布 ---")
    print(f"  mean={cd['mean']:.3f}, median={cd['median']:.3f}, "
          f"std={cd['std']:.3f}")
    print(f"  >=0.9: {cd['pct_ge_0.9']*100:.1f}% | "
          f">=0.8: {cd['pct_ge_0.8']*100:.1f}% | "
          f">=0.7: {cd['pct_ge_0.7']*100:.1f}% | "
          f">=0.6: {cd['pct_ge_0.6']*100:.1f}%")

    print("\n[DONE] Step 2 完成。")


def main():
    print("=" * 60)
    print("Step 2: 标签传播 (k-NN)")
    print("=" * 60)

    feats, feature_index = load_features()
    seeds = load_seed_labels(feature_index)
    index = build_faiss_index(feats)

    all_labels, all_conf, all_votes, all_source_type = knn_propagate(
        feats, index, seeds, k=KNN_K, threshold=CONF_THRESHOLD
    )

    stats = save_results(
        feats, feature_index, seeds,
        all_labels, all_conf, all_votes, all_source_type
    )
    print_summary(stats)


if __name__ == "__main__":
    main()
