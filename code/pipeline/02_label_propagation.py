# -*- coding: utf-8 -*-
"""
Step 2: label propagation
with the 16 exact semantic classes of hf_dunhuang as seeds,
Propagate labels to all 32,415 images by k-NN voting in DINOv2 feature space

strategy:
  - seed set: 2,278 images from hf_dunhuang (the 16 classes in metadata.csv)
  - method: exact inner-product search with faiss (features are L2-normalised, so the inner product is the cosine similarity)
  - k = 7 weighted nearest-neighbour vote (weight = cosine similarity)
  - confidence = weighted top-1 class votes / total weighted votes
  - >= 0.60: reliable label; < 0.60: uncertain
  - super-resolution variants in muraldh_sr are handled specially: one label per source image

output:
  annotations/seed_labels.csv  - seed-set labels
  annotations/all_labels.csv   - propagated labels for every image
  annotations/label_statistics.json - statistics
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
    """Load the feature matrix and index produced by step 1"""
    print(f"[INFO] 加载特征矩阵: {FEATURE_NPY}")
    feats = np.load(FEATURE_NPY)
    with open(FEATURE_INDEX, "r", encoding="utf-8") as f:
        idx = json.load(f)
    print(f"  特征矩阵: {feats.shape}, 已归一化: {idx.get('normalized', False)}")
    print(f"  索引记录数: {idx['num_images']}")
    return feats, idx


def load_seed_labels(feature_index):
    """
    Load the seed labels from hf_dunhuang/metadata.csv.
    the file names have to be mapped to the global indices in feature_index.
    The file field of metadata.csv looks like 'animals_00000.jpg',
    the corresponding path is data/raw/dunhuang_hf/images/{label_name}/{file}
    """
    print(f"[INFO] 加载种子标签: {HF_CSV}")

    # map path -> global_index
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
            # build the full path
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
                # try a case-insensitive match
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

    # statistics by class
    label_counts = Counter(s["label"] for s in seeds)
    print(f"  类别数: {len(label_counts)}")
    for cat in CATEGORY_NAMES:
        cnt = label_counts.get(cat, 0)
        cn = CATEGORY_NAMES_CN.get(cat, cat)
        print(f"    {cat:20s} ({cn:6s}): {cnt:4d}")

    return seeds


def build_faiss_index(feats):
    """Build a faiss inner-product index (the features are already L2-normalised)"""
    dim = feats.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(feats.astype(np.float32))
    print(f"[INFO] faiss 索引构建完成: {index.ntotal} vectors, dim={dim}")
    return index


def knn_propagate(feats, index, seeds, k=KNN_K, threshold=CONF_THRESHOLD):
    """
    k-NN weighted-vote label propagation.
    for every non-seed image, find the k nearest neighbours (searching the seed set first),
    but for the seed labels to propagate globally we search the k nearest neighbours over all images,
    in which the votes of the seed labels carry more weight.

    improved strategy: two-stage propagation
      Stage 1: k-NN (k=7) over the seed labels only, which gives the initial propagation
      Stage 2: take the high-confidence (>=0.7) propagations, add them to the seed set and run one more round
               but to stop errors from spreading, stage 2 is applied only to images marked uncertain
    """
    n_total = feats.shape[0]
    seed_indices = set(s["global_idx"] for s in seeds)
    seed_label_map = {s["global_idx"]: s["label"] for s in seeds}

    # index the seed features separately for Stage 1
    seed_feats = np.array([feats[s["global_idx"]] for s in seeds], dtype=np.float32)
    seed_index = faiss.IndexFlatIP(feats.shape[1])
    seed_index.add(seed_feats)
    seed_idx_list = [s["global_idx"] for s in seeds]

    print(f"\n[INFO] Stage 1: 种子集 k-NN 传播 (k={k})")
    t0 = time.time()

    # search the k nearest seeds for every image
    # search in batches to avoid a memory peak
    batch_size = 4096
    all_labels = [""] * n_total
    all_conf = np.zeros(n_total, dtype=np.float32)
    all_votes = [""] * n_total  # vote detail as a JSON string
    all_source_type = [""] * n_total

    # seed images are labelled directly
    for s in seeds:
        all_labels[s["global_idx"]] = s["label"]
        all_conf[s["global_idx"]] = 1.0
        all_source_type[s["global_idx"]] = "seed"

    # k-NN over the non-seed images
    non_seed_mask = np.ones(n_total, dtype=bool)
    for si in seed_indices:
        non_seed_mask[si] = False
    non_seed_indices = np.where(non_seed_mask)[0]
    print(f"  待传播图像: {len(non_seed_indices)}")

    for start in range(0, len(non_seed_indices), batch_size):
        end = min(start + batch_size, len(non_seed_indices))
        batch_global = non_seed_indices[start:end]
        queries = feats[batch_global].astype(np.float32)

        # searching k+1 would guard against matching oneself, but self is not in the seed set, so it is unnecessary
        D, I = seed_index.search(queries, k)

        for local_i in range(len(batch_global)):
            gi = batch_global[local_i]
            sims = D[local_i]  # cosine similarity
            nn_seed_positions = I[local_i]  # position inside seed_idx_list

            # weighted vote
            vote_weights = defaultdict(float)
            vote_details = []
            for rank, (sim, sp) in enumerate(zip(sims, nn_seed_positions)):
                if sp < 0:
                    continue
                seed_gi = seed_idx_list[sp]
                label = seed_label_map[seed_gi]
                w = float(max(sim, 0))  # similarity as the weight
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

    # Stage 2: extend the seed set with the high-confidence propagation results and run another pass over the uncertain images
    print(f"\n[INFO] Stage 2: 高置信度扩展种子集，重新标注 uncertain")
    high_conf_mask = (all_conf >= 0.75) & (np.array(all_source_type) == "seed_knn")
    high_conf_indices = np.where(high_conf_mask)[0]
    print(f"  高置信度(>=0.75)扩展种子: {len(high_conf_indices)}")

    if len(high_conf_indices) > 0 and uncertain_count > 0:
        # extended seed index = original seeds + high-confidence propagation
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
                n_orig = 0  # number of original seed votes
                for sim, pos in zip(sims, nn_positions):
                    if pos < 0:
                        continue
                    label = ext_seed_labels[pos]
                    origin = ext_seed_origins[pos]
                    w = float(max(sim, 0))
                    if origin == "original":
                        w *= 1.5  # bonus weight for original seed votes
                        n_orig += 1
                    vote_weights[label] += w
                    vote_details.append(f"{label}({origin[0]}):{w:.3f}")

                if vote_weights:
                    sorted_votes = sorted(vote_weights.items(), key=lambda x: -x[1])
                    top_label, top_weight = sorted_votes[0]
                    total_weight = sum(vote_weights.values())
                    conf = top_weight / total_weight if total_weight > 0 else 0

                    # upgrade only when confidence rises and at least one original seed vote exists
                    if conf >= threshold and n_orig >= 2 and conf > all_conf[gi]:
                        all_labels[gi] = top_label
                        all_conf[gi] = conf
                        all_votes[gi] = "; ".join(vote_details)
                        all_source_type[gi] = "expanded_knn"
                        upgraded += 1

        print(f"  Stage 2 升级: {upgraded} / {uncertain_count}")

    # final statistics
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
    """Save the seed labels and the full label CSV"""
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

    # statistics by source
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

    # statistics by label
    by_label = Counter()
    for gi in range(len(all_labels)):
        by_label[all_labels[gi]] += 1
    stats["by_label"] = dict(by_label.most_common())

    # statistics by label_type
    by_type = Counter(all_source_type)
    stats["by_label_type"] = dict(by_type)

    # confidence distribution
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
    """Print the final summary"""
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
