# -*- coding: utf-8 -*-
"""
Step 3b: 全局去重与重新划分
基于 DINOv2 特征余弦相似度检测跨集合重复图像，使用 Union-Find 聚类后
按优先级分配集合，确保 train/query/gallery 之间无 sim > 0.99 的重复。

优先级: train > gallery_labeled > query > gallery_distractor
- 重复组中有 train 成员: 全部非distractor成员归入 train
- 无 train 但有 gallery_labeled: 归入 gallery_labeled
- 只有 query: 保留在 query
- distractor 始终保留在 gallery (作为干扰项不影响评估)
- muraldh_sr 组保持完整 (5变体同进退)
"""
import os, sys, json, csv, re
import numpy as np
import faiss
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FEATURE_NPY, FEATURE_INDEX, ALL_LABELS,
    TRAIN_CSV, QUERY_CSV, GALLERY_CSV, CROSS_SRC_CSV,
    SPLIT_DIR, CATEGORY_NAMES, CATEGORY_NAMES_CN,
    TRAIN_RATIO, QUERY_RATIO, GALLERY_RATIO, RANDOM_SEED,
    CROSS_SOURCE_HOLDOUT
)

SIM_THRESHOLD = 0.99  # 严格去重阈值


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def load_all_labels():
    rows = []
    with open(ALL_LABELS, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["global_idx"] = int(row["global_idx"])
            row["confidence"] = float(row["confidence"])
            rows.append(row)
    return rows


def extract_sr_group_key(path):
    # SR 组数据实际路径包含 "Mural_SR"，用小写匹配以兼容 "muraldh_sr" 和 "Mural_SR"
    p_lower = path.lower().replace("\\", "/")
    if "mural_sr" not in p_lower and "muraldh_sr" not in p_lower:
        return None
    fname = os.path.basename(path)
    m = re.search(r'(\d+)', fname)
    if m:
        return f"sr_{m.group(1)}"
    return None


def find_duplicates(feats, threshold=SIM_THRESHOLD):
    """用 faiss 批量搜索自最近邻，找出所有 sim > threshold 的重复对"""
    n = feats.shape[0]
    print(f"[INFO] 在 {n} 张图像中搜索重复对 (sim > {threshold})...")

    index = faiss.IndexFlatIP(feats.shape[1])
    index.add(feats.astype(np.float32))

    # 搜索 k=10 近邻 (一个重复簇可能有多张图)
    k = min(10, n)
    dup_pairs = []
    batch_size = 4096

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        queries = feats[start:end].astype(np.float32)
        D, I = index.search(queries, k)

        for local_i in range(end - start):
            gi = start + local_i
            for rank in range(k):
                nj = int(I[local_i, rank])
                sim = float(D[local_i, rank])
                if nj < 0 or nj <= gi:  # 避免自匹配和重复计数
                    continue
                if sim >= threshold:
                    dup_pairs.append((gi, nj, sim))

        if (start // batch_size) % 2 == 0:
            print(f"  [{end}/{n}] found {len(dup_pairs)} pairs so far")

    print(f"[INFO] 共找到 {len(dup_pairs)} 对重复")
    return dup_pairs


def analyze_duplicate_sources(dup_pairs, rows):
    """分析重复对的来源组合"""
    print("\n[INFO] 重复对来源分析:")
    source_pairs = Counter()
    sim_buckets = Counter()

    for gi, nj, sim in dup_pairs:
        s1 = rows[gi]["source"]
        s2 = rows[nj]["source"]
        pair = tuple(sorted([s1, s2]))
        source_pairs[pair] += 1

        if sim >= 0.999:
            sim_buckets["1.000 (identical)"] += 1
        elif sim >= 0.995:
            sim_buckets["0.995-0.999"] += 1
        else:
            sim_buckets["0.990-0.995"] += 1

    for pair, cnt in source_pairs.most_common():
        print(f"  {pair[0]:20s} <-> {pair[1]:20s}: {cnt:5d}")

    print(f"\n  相似度分布:")
    for bucket, cnt in sorted(sim_buckets.items()):
        print(f"    {bucket}: {cnt}")


def deduplicate_and_assign(rows, feats, dup_pairs):
    """
    Union-Find 聚类重复组，然后按优先级分配。
    返回: 每个 global_idx 的最终 split 分配
    """
    n = len(rows)
    uf = UnionFind(n)

    for gi, nj, sim in dup_pairs:
        uf.union(gi, nj)

    # 收集聚类
    clusters = defaultdict(list)
    for i in range(n):
        root = uf.find(i)
        clusters[root].append(i)

    # 过滤单元素簇
    dup_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    print(f"\n[INFO] 重复簇数: {len(dup_clusters)} "
          f"(涉及 {sum(len(v) for v in dup_clusters.values())} 张图像)")

    # 簇大小分布
    cluster_sizes = Counter(len(v) for v in dup_clusters.values())
    print(f"  簇大小分布: {dict(sorted(cluster_sizes.items()))}")

    # 确定每个簇的目标 split
    # 优先级: train > cross_source > gallery_labeled > query > gallery_distractor
    # cross_source 不应该被去重影响 (它们是独立留出的)
    PRIORITY = {
        "train": 5,
        "cross_source": 4,
        "gallery_labeled": 3,
        "query": 2,
        "gallery_distractor": 1,
        "unassigned": 0,
    }

    # 先初始化每个图像的当前 split (来自 Step 3 的划分)
    # 重新加载 split CSVs 获取当前分配
    split_map = {}  # global_idx -> split_name

    train_rows = load_csv_simple(TRAIN_CSV)
    query_rows = load_csv_simple(QUERY_CSV)
    gallery_rows = load_csv_simple(GALLERY_CSV)
    cross_rows = load_csv_simple(CROSS_SRC_CSV)

    for r in train_rows:
        split_map[int(r["global_idx"])] = "train"
    for r in query_rows:
        split_map[int(r["global_idx"])] = "query"
    for r in gallery_rows:
        idx = int(r["global_idx"])
        if r.get("is_distractor", "False") == "True":
            split_map[idx] = "gallery_distractor"
        else:
            split_map[idx] = "gallery_labeled"
    for r in cross_rows:
        split_map[int(r["global_idx"])] = "cross_source"

    # 对每个重复簇，决定目标 split
    reassign_count = 0
    for root, members in dup_clusters.items():
        # 统计簇中各 split 的成员
        splits_in_cluster = Counter()
        for idx in members:
            s = split_map.get(idx, "unassigned")
            splits_in_cluster[s] += 1

        # 选最高优先级的 split
        best_split = max(splits_in_cluster.keys(),
                         key=lambda s: PRIORITY.get(s, 0))

        # 如果簇中包含 cross_source 成员，cross_source 保留，
        # 其他成员重新分配到 train (不能让cross数据泄漏到其他集合)
        if "cross_source" in splits_in_cluster:
            for idx in members:
                if split_map.get(idx) != "cross_source":
                    old = split_map.get(idx, "unassigned")
                    split_map[idx] = "train"  # 移入train
                    if old != "train":
                        reassign_count += 1
        elif best_split == "train":
            # 所有非distractor成员移入train
            for idx in members:
                old = split_map.get(idx, "unassigned")
                if old == "gallery_distractor":
                    pass  # distractor保持在gallery
                elif old != "train":
                    split_map[idx] = "train"
                    reassign_count += 1
        elif best_split == "gallery_labeled":
            for idx in members:
                old = split_map.get(idx, "unassigned")
                if old == "gallery_distractor":
                    pass
                elif old == "query":
                    # query中的重复图移到gallery_labeled
                    split_map[idx] = "gallery_labeled"
                    reassign_count += 1
        elif best_split == "query":
            # 只有query成员，保留query；gallery_distractor不动
            pass

    print(f"[INFO] 重新分配: {reassign_count} 张图像")

    return split_map, dup_clusters


def load_csv_simple(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def rebuild_splits(rows, split_map, feats):
    """根据去重后的 split_map 重新生成数据集划分"""
    print("\n[INFO] 重建数据集划分...")

    # 按新分配分组
    new_train = []
    new_query = []
    new_gallery_labeled = []
    new_gallery_distractor = []
    new_cross = []

    for r in rows:
        idx = r["global_idx"]
        s = split_map.get(idx, "unassigned")

        if s == "train":
            new_train.append(r)
        elif s == "query":
            new_query.append(r)
        elif s == "gallery_labeled":
            new_gallery_labeled.append(r)
        elif s == "gallery_distractor":
            new_gallery_distractor.append(r)
        elif s == "cross_source":
            new_cross.append(r)
        else:
            # 未分配的: reliable -> 需要重新分层划分
            # uncertain -> distractor
            if r["label_type"] == "uncertain":
                new_gallery_distractor.append(r)
            # 可靠标签但未分配的，后续处理

    print(f"  去重后初步分配:")
    print(f"    Train: {len(new_train)}")
    print(f"    Query: {len(new_query)}")
    print(f"    Gallery(labeled): {len(new_gallery_labeled)}")
    print(f"    Gallery(distractor): {len(new_gallery_distractor)}")
    print(f"    Cross-source: {len(new_cross)}")

    # 对未分配的可靠标签图像，按比例补入各集合
    unassigned_reliable = [r for r in rows
                           if split_map.get(r["global_idx"], "unassigned") == "unassigned"
                           and r["label_type"] != "uncertain"
                           and r["source"] not in CROSS_SOURCE_HOLDOUT]
    if unassigned_reliable:
        print(f"\n[INFO] 未分配可靠标签图像: {len(unassigned_reliable)}，补入划分...")
        rng = np.random.RandomState(RANDOM_SEED)

        # 按 (label, source) 分层
        strata = defaultdict(list)
        for r in unassigned_reliable:
            # SR组整组处理
            gk = extract_sr_group_key(r["path"])
            if gk is not None:
                strata[("sr_group", gk)].append(r)
            else:
                strata[(r["label"], r["source"])].append(r)

        for key, group in sorted(strata.items()):
            n = len(group)
            if n == 0:
                continue
            indices = rng.permutation(n)
            n_train = max(1, int(round(n * TRAIN_RATIO)))
            n_query = max(1, int(round(n * QUERY_RATIO)))
            n_gallery = n - n_train - n_query
            if n_gallery < 1:
                n_gallery = 1
                n_query = max(1, n - n_train - n_gallery)
                if n_train + n_query + n_gallery > n:
                    n_train = n - n_query - n_gallery

            for i, idx in enumerate(indices):
                r = group[idx]
                if i < n_train:
                    new_train.append(r)
                elif i < n_train + n_query:
                    new_query.append(r)
                else:
                    new_gallery_labeled.append(r)

    # 标记字段
    for r in new_train:
        r["split"] = "train"
        r["is_distractor"] = False
    for r in new_query:
        r["split"] = "query"
        r["is_distractor"] = False
    for r in new_gallery_labeled:
        r["split"] = "gallery"
        r["is_distractor"] = False
    for r in new_gallery_distractor:
        r["split"] = "gallery"
        r["is_distractor"] = True
        if not r.get("label") or r["label"] == "":
            r["label"] = "distractor"
            r["label_cn"] = "干扰项"
    for r in new_cross:
        r["split"] = "cross_source_query"
        r["is_distractor"] = False

    all_gallery = new_gallery_labeled + new_gallery_distractor

    return new_train, new_query, all_gallery, new_cross


def verify_no_duplicates(train, query, gallery, feats, path_to_idx):
    """去重后验证"""
    print("\n[VERIFY] 去重后跨集合重复验证...")

    def get_indices(rows):
        return np.array([path_to_idx[r["path"]] for r in rows
                         if r["path"] in path_to_idx], dtype=np.int64)

    train_idx = get_indices(train)
    query_idx = get_indices(query)
    gallery_labeled = [r for r in gallery if not r["is_distractor"]]
    gallery_idx = get_indices(gallery_labeled)

    issues = 0

    # Train vs Query
    if len(train_idx) > 0 and len(query_idx) > 0:
        index = faiss.IndexFlatIP(feats.shape[1])
        index.add(feats[train_idx].astype(np.float32))
        D, _ = index.search(feats[query_idx].astype(np.float32), 1)
        max_sim = D.max()
        n_dupes = (D.flatten() >= SIM_THRESHOLD).sum()
        print(f"  Train-Query max sim: {max_sim:.6f}, "
              f"dupes(>={SIM_THRESHOLD}): {n_dupes}")
        issues += n_dupes

    # Train vs Gallery(labeled)
    if len(train_idx) > 0 and len(gallery_idx) > 0:
        index = faiss.IndexFlatIP(feats.shape[1])
        index.add(feats[train_idx].astype(np.float32))
        D, _ = index.search(feats[gallery_idx].astype(np.float32), 1)
        max_sim = D.max()
        n_dupes = (D.flatten() >= SIM_THRESHOLD).sum()
        print(f"  Train-Gallery(max_sim): {max_sim:.6f}, "
              f"dupes(>={SIM_THRESHOLD}): {n_dupes}")
        issues += n_dupes

    # Query vs Gallery(labeled) - 检查精确匹配
    if len(query_idx) > 0 and len(gallery_idx) > 0:
        index = faiss.IndexFlatIP(feats.shape[1])
        index.add(feats[gallery_idx].astype(np.float32))
        D, _ = index.search(feats[query_idx].astype(np.float32), 1)
        max_sim = D.max()
        n_dupes = (D.flatten() >= SIM_THRESHOLD).sum()
        print(f"  Query-Gallery max sim: {max_sim:.6f}, "
              f"dupes(>={SIM_THRESHOLD}): {n_dupes}")
        issues += n_dupes

    if issues == 0:
        print("  [ALL CLEAN] 无跨集合重复!")
    else:
        print(f"  [WARNING] 仍有 {issues} 处重复")

    return issues


def write_csv(filepath, rows):
    fieldnames = [
        "global_idx", "path", "source",
        "label", "label_cn", "confidence",
        "label_type", "split", "is_distractor", "vote_details"
    ]
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVE] {filepath} ({len(rows)} rows)")


def main():
    print("=" * 60)
    print("Step 3b: 全局去重与重新划分")
    print("=" * 60)

    rows = load_all_labels()
    feats = np.load(FEATURE_NPY)
    with open(FEATURE_INDEX, "r", encoding="utf-8") as f:
        feat_index = json.load(f)
    path_to_idx = {rec["path"]: rec["index"] for rec in feat_index["records"]}

    print(f"[INFO] {len(rows)} images, features {feats.shape}")

    # 1. 搜索重复对
    dup_pairs = find_duplicates(feats, SIM_THRESHOLD)
    analyze_duplicate_sources(dup_pairs, rows)

    if not dup_pairs:
        print("[INFO] 无重复，无需去重")
        return

    # 2. Union-Find 聚类并分配
    split_map, dup_clusters = deduplicate_and_assign(rows, feats, dup_pairs)

    # 3. 重建划分
    train, query, gallery, cross = rebuild_splits(rows, split_map, feats)

    # 4. 验证
    verify_no_duplicates(train, query, gallery, feats, path_to_idx)

    # 5. 保存
    os.makedirs(SPLIT_DIR, exist_ok=True)
    write_csv(TRAIN_CSV, train)
    write_csv(QUERY_CSV, query)
    write_csv(GALLERY_CSV, gallery)
    write_csv(CROSS_SRC_CSV, cross)

    # 6. 统计
    print("\n" + "=" * 60)
    print("去重后最终划分")
    print("=" * 60)
    print(f"  Train:        {len(train):6d}")
    print(f"  Query:        {len(query):6d}")
    n_gallery_labeled = sum(1 for r in gallery if not r["is_distractor"])
    n_dist = sum(1 for r in gallery if r["is_distractor"])
    print(f"  Gallery:      {len(gallery):6d} (labeled: {n_gallery_labeled}, distractor: {n_dist})")
    print(f"  Cross-source: {len(cross):6d}")
    print(f"  Total:        {len(train)+len(query)+len(gallery)+len(cross):6d}")

    print(f"\n--- Train 各来源 ---")
    for src, cnt in Counter(r["source"] for r in train).most_common():
        print(f"  {src:20s}: {cnt:5d}")
    print(f"\n--- Query 各来源 ---")
    for src, cnt in Counter(r["source"] for r in query).most_common():
        print(f"  {src:20s}: {cnt:5d}")
    print(f"\n--- Gallery 各来源 ---")
    for src, cnt in Counter(r["source"] for r in gallery).most_common():
        nl = sum(1 for r in gallery if r["source"]==src and not r["is_distractor"])
        nd = sum(1 for r in gallery if r["source"]==src and r["is_distractor"])
        print(f"  {src:20s}: {cnt:5d} (labeled: {nl}, dist: {nd})")

    # SR组完整性
    sr_splits = defaultdict(set)
    for r in train + query + gallery:
        gk = extract_sr_group_key(r["path"])
        if gk:
            sr_splits[gk].add(r["split"])
    sr_leak = sum(1 for v in sr_splits.values() if len(v) > 1)
    print(f"\n--- SR组完整性 ---")
    print(f"  总组数: {len(sr_splits)}, 跨集合: {sr_leak}")

    # 类别分布
    print(f"\n--- 类别分布 (train/query/gallery/cross) ---")
    for label in CATEGORY_NAMES:
        n_t = sum(1 for r in train if r["label"] == label)
        n_q = sum(1 for r in query if r["label"] == label)
        n_g = sum(1 for r in gallery if r["label"] == label and not r["is_distractor"])
        n_c = sum(1 for r in cross if r["label"] == label)
        cn = CATEGORY_NAMES_CN[label]
        print(f"  {label:20s} ({cn:6s}): "
              f"train={n_t:4d}  query={n_q:4d}  "
              f"gallery={n_g:4d}  cross={n_c:4d}")

    print("\n[DONE] 去重完成。")


if __name__ == "__main__":
    main()
