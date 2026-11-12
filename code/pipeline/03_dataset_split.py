# -*- coding: utf-8 -*-
"""
Step 3: 数据集划分
将标注数据划分为 train / query / gallery / cross_source_test。

划分策略:
  1. 跨源泛化测试集: deepflying + dunhuang_grottoes 全部留出
     - 这两个数据源的可靠标签图像作为 query，在其余源的 gallery 中检索
     - 用于验证模型跨数据源泛化能力
  2. 其余5个数据源 (dh_v1, muraldh_512, muraldh_sr, hf_dunhuang, muraldh_seg)
     的可靠标签图像按 70/15/15 分层划分 train/query/gallery
  3. 分层抽样: 按 label + source 联合分层，保证各类别在各集合中比例一致
  4. muraldh_sr 超分辨率变体处理:
     - 同一原图的 X2/X3/X4/X6 变体必须放在同一集合，避免信息泄漏
  5. uncertain 图像:
     - 不参与 train/query 划分
     - 全部加入 gallery 作为干扰项 (distractors)，增加检索难度
     - 标记为 distractor=True，评估时不计入 relevant

输出:
  splits/train.csv
  splits/query.csv
  splits/gallery.csv
  splits/cross_source_test.csv
  splits/split_statistics.json
"""
import os, sys, json, csv, re
import numpy as np
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    ALL_LABELS, SPLIT_DIR,
    TRAIN_CSV, QUERY_CSV, GALLERY_CSV, CROSS_SRC_CSV,
    TRAIN_RATIO, QUERY_RATIO, GALLERY_RATIO, RANDOM_SEED,
    CROSS_SOURCE_HOLDOUT, CATEGORY_NAMES, CATEGORY_NAMES_CN,
    FEATURE_INDEX
)


def load_all_labels():
    """加载 Step 2 产出的全量标签"""
    rows = []
    with open(ALL_LABELS, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["global_idx"] = int(row["global_idx"])
            row["confidence"] = float(row["confidence"])
            rows.append(row)
    print(f"[INFO] 加载标签: {len(rows)} rows")
    return rows


def extract_sr_group_key(path, source):
    """
    提取 muraldh_sr 的原图组键，确保同一原图的不同倍率变体不跨集合。
    路径模式: .../Mural_SR/Mural_DataSet/0001.png  (X2/X3/X4/X6 在不同子目录)
    实际路径可能是 .../Mural_DataSet/X2/0001.png 等格式
    """
    if source != "muraldh_sr":
        return None
    # 提取文件名中的数字部分作为组键
    fname = os.path.basename(path)
    match = re.search(r'(\d+)', fname)
    if match:
        return f"sr_{match.group(1)}"
    return None


def split_by_stratified_group(records, train_ratio, query_ratio, gallery_ratio,
                              seed=42):
    """兼容旧接口: 对非SR记录分层划分 (SR已在main中预处理)"""
    return split_by_stratified_group_with_sr(
        records, [], train_ratio, query_ratio, gallery_ratio, seed
    )


def split_by_stratified_group_with_sr(records, sr_units, train_ratio,
                                       query_ratio, gallery_ratio, seed=42):
    """
    分层分组划分 (支持SR预分组单元):
      - records: 非SR的单条记录列表, 每条作为独立单元
      - sr_units: SR预分组单元列表, 每个unit是list of records (整组同进退)
      - 分层: 按 (label, source) 联合分层
    """
    rng = np.random.RandomState(seed)
    strata = defaultdict(list)

    for r in records:
        key = (r["label"], r["source"])
        strata[key].append([r])

    for unit in sr_units:
        label = unit[0]["label"]
        key = (label, "muraldh_sr")
        strata[key].append(unit)

    train, query, gallery = [], [], []

    for (label, source), units in sorted(strata.items()):
        n_units = len(units)
        if n_units == 0:
            continue

        if n_units < 3:
            for unit in units:
                train.extend(unit)
            continue

        n_train = max(1, int(round(n_units * train_ratio)))
        n_query = max(1, int(round(n_units * query_ratio)))
        n_gallery = n_units - n_train - n_query

        if n_gallery < 1:
            n_gallery = 1
            n_query = max(1, n_units - n_train - n_gallery)
            if n_train + n_query + n_gallery > n_units:
                n_train = n_units - n_query - n_gallery

        indices = rng.permutation(n_units)
        train_end = n_train
        query_end = n_train + n_query

        for i, idx in enumerate(indices):
            unit = units[idx]
            if i < train_end:
                train.extend(unit)
            elif i < query_end:
                query.extend(unit)
            else:
                gallery.extend(unit)

    return train, query, gallery


def write_csv(filepath, rows, extra_cols=None):
    """写入划分 CSV"""
    if extra_cols is None:
        extra_cols = []
    fieldnames = [
        "global_idx", "path", "source",
        "label", "label_cn", "confidence",
        "label_type", "split", "is_distractor"
    ] + extra_cols

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[SAVE] {filepath} ({len(rows)} rows)")


def compute_label_overlap(train_labels, test_labels):
    """计算训练集和测试集的标签重叠率"""
    train_set = set(train_labels)
    test_set = set(test_labels)
    overlap = train_set & test_set
    return len(overlap), len(test_set)


def main():
    print("=" * 60)
    print("Step 3: 数据集划分")
    print("=" * 60)

    rows = load_all_labels()

    # --- 预处理 muraldh_sr 组: 确保同组5变体标签统一、不跨集合 ---
    sr_groups = defaultdict(list)
    non_sr_rows = []
    for r in rows:
        if r["source"] == "muraldh_sr":
            gk = extract_sr_group_key(r["path"], "muraldh_sr")
            if gk is not None:
                sr_groups[gk].append(r)
            else:
                non_sr_rows.append(r)
        else:
            non_sr_rows.append(r)

    # 对每个SR组: 优先用HR原图(根目录)标签, 否则用组内最高置信度标签
    # 如果组内有可靠标签, 全组统一为该标签并参与划分; 否则全组标为distractor
    sr_reliable_units = []  # list of list (有可靠标签的组)
    sr_distractor_units = []  # list of list (全组uncertain)

    for gk, members in sr_groups.items():
        hr = [m for m in members if "Mural_DataSet_LR" not in m["path"]]
        reliable_members = [m for m in members if m["label_type"] != "uncertain"]

        if reliable_members:
            # 优先HR原图, 其次组内最高置信度
            if hr and hr[0]["label_type"] != "uncertain":
                best = hr[0]
            else:
                best = max(reliable_members, key=lambda x: x["confidence"])
            group_label = best["label"]
            for m in members:
                m["label"] = group_label
                m["label_cn"] = CATEGORY_NAMES_CN.get(group_label, group_label)
                # 保留原始 label_type 和 confidence, 但标签统一
            sr_reliable_units.append(members)
        else:
            for m in members:
                m["label"] = "distractor"
                m["label_cn"] = "干扰项"
            sr_distractor_units.append(members)

    # 分离非SR数据的 reliable vs uncertain
    non_sr_reliable = [r for r in non_sr_rows if r["label_type"] != "uncertain"]
    non_sr_uncertain = [r for r in non_sr_rows if r["label_type"] == "uncertain"]

    # 分离跨源留出
    cross_source = [r for r in non_sr_reliable if r["source"] in CROSS_SOURCE_HOLDOUT]
    main_source = [r for r in non_sr_reliable if r["source"] not in CROSS_SOURCE_HOLDOUT]

    # 将SR可靠组以整体单元注入 main_source 前先临时修改split函数处理逻辑
    # 简单方案: 将SR组作为"已分组记录"传入, 修改split函数支持
    print(f"\n[INFO] 可靠标签: {len(non_sr_reliable) + sum(len(u) for u in sr_reliable_units)}, "
          f"uncertain: {len(non_sr_uncertain) + sum(len(u) for u in sr_distractor_units)}")
    print(f"[INFO] SR可靠组: {len(sr_reliable_units)}组 ({sum(len(u) for u in sr_reliable_units)}张), "
          f"SR干扰组: {len(sr_distractor_units)}组 ({sum(len(u) for u in sr_distractor_units)}张)")
    print(f"[INFO] 跨源留出 ({'+'.join(CROSS_SOURCE_HOLDOUT)}): {len(cross_source)}")
    print(f"[INFO] 主源数据(非SR): {len(main_source)}")

    # 分层分组划分: 非SR + SR可靠组
    print(f"\n[INFO] 按 70/15/15 分层划分主源数据...")
    train, query, gallery = split_by_stratified_group_with_sr(
        main_source, sr_reliable_units,
        TRAIN_RATIO, QUERY_RATIO, GALLERY_RATIO, RANDOM_SEED
    )

    # 标记 split 和 distractor
    for r in train:
        r["split"] = "train"
        r["is_distractor"] = False
    for r in query:
        r["split"] = "query"
        r["is_distractor"] = False
    for r in gallery:
        r["split"] = "gallery"
        r["is_distractor"] = False

    # 跨源数据作为额外 query（从主源 gallery 中检索）
    for r in cross_source:
        r["split"] = "cross_source_query"
        r["is_distractor"] = False

    # uncertain 非SR图像加入 gallery 作为 distractors
    for r in non_sr_uncertain:
        r["split"] = "gallery"
        r["is_distractor"] = True
        r["label"] = "distractor"
        r["label_cn"] = "干扰项"
    gallery.extend(non_sr_uncertain)

    # SR全组uncertain的也加入gallery distractors (整组加入)
    for unit in sr_distractor_units:
        for r in unit:
            r["split"] = "gallery"
            r["is_distractor"] = True
            r["label"] = "distractor"
            r["label_cn"] = "干扰项"
        gallery.extend(unit)

    # 确保目录存在
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # 写入 CSV
    write_csv(TRAIN_CSV, train)
    write_csv(QUERY_CSV, query)
    write_csv(GALLERY_CSV, gallery)
    write_csv(CROSS_SRC_CSV, cross_source)

    # ============ 统计信息 ============
    total_distractors = len(non_sr_uncertain) + sum(len(u) for u in sr_distractor_units)
    stats = {
        "random_seed": RANDOM_SEED,
        "split_ratio": {
            "train": TRAIN_RATIO,
            "query": QUERY_RATIO,
            "gallery": GALLERY_RATIO
        },
        "cross_source_holdout": CROSS_SOURCE_HOLDOUT,
        "total_images": len(rows),
        "reliable_labels": len(non_sr_reliable) + sum(len(u) for u in sr_reliable_units),
        "uncertain_distractors": total_distractors,
    }

    # 各集合统计
    stats["splits"] = {
        "train": {
            "total": len(train),
            "by_source": dict(Counter(r["source"] for r in train)),
            "by_label": dict(Counter(r["label"] for r in train).most_common()),
        },
        "query": {
            "total": len(query),
            "by_source": dict(Counter(r["source"] for r in query)),
            "by_label": dict(Counter(r["label"] for r in query).most_common()),
        },
        "gallery": {
            "total": len(gallery),
            "labeled": sum(1 for r in gallery if not r["is_distractor"]),
            "distractors": sum(1 for r in gallery if r["is_distractor"]),
            "by_source": dict(Counter(r["source"] for r in gallery)),
            "by_label": dict(Counter(r["label"] for r in gallery).most_common()),
        },
        "cross_source_test": {
            "total": len(cross_source),
            "by_source": dict(Counter(r["source"] for r in cross_source)),
            "by_label": dict(Counter(r["label"] for r in cross_source).most_common()),
        },
    }

    # 标签重叠分析
    train_labels = set(r["label"] for r in train)
    query_labels = set(r["label"] for r in query)
    gallery_labels = set(r["label"] for r in gallery if not r["is_distractor"])
    cross_labels = set(r["label"] for r in cross_source)

    overlap_tq = train_labels & query_labels
    overlap_tg = train_labels & gallery_labels
    overlap_tc = train_labels & cross_labels

    stats["label_overlap"] = {
        "train_vs_query": {
            "overlap": len(overlap_tq),
            "train_unique": len(train_labels - query_labels),
            "query_unique": len(query_labels - train_labels),
            "overlap_labels": sorted(list(overlap_tq)),
        },
        "train_vs_gallery": {
            "overlap": len(overlap_tg),
            "train_unique": len(train_labels - gallery_labels),
            "gallery_unique": len(gallery_labels - train_labels),
            "overlap_labels": sorted(list(overlap_tg)),
        },
        "train_vs_cross_source": {
            "overlap": len(overlap_tc),
            "train_unique": len(train_labels - cross_labels),
            "cross_source_unique": len(cross_labels - train_labels),
            "overlap_labels": sorted(list(overlap_tc)),
            "cross_source_only_labels": sorted(list(cross_labels - train_labels)),
        },
    }

    # 每类样本数（用于检查类别不平衡）
    all_labeled = train + query + [r for r in gallery if not r["is_distractor"]]
    label_counts = Counter(r["label"] for r in all_labeled)
    stats["class_distribution"] = {}
    for label in CATEGORY_NAMES:
        count = label_counts.get(label, 0)
        stats["class_distribution"][label] = {
            "cn": CATEGORY_NAMES_CN.get(label, label),
            "total": count,
            "train": sum(1 for r in train if r["label"] == label),
            "query": sum(1 for r in query if r["label"] == label),
            "gallery": sum(1 for r in gallery if r["label"] == label and not r["is_distractor"]),
            "cross_source": sum(1 for r in cross_source if r["label"] == label),
        }

    # muraldh_sr 组完整性检查
    sr_train = set(extract_sr_group_key(r["path"], "muraldh_sr")
                   for r in train if r["source"] == "muraldh_sr")
    sr_query = set(extract_sr_group_key(r["path"], "muraldh_sr")
                   for r in query if r["source"] == "muraldh_sr")
    sr_gallery = set(extract_sr_group_key(r["path"], "muraldh_sr")
                     for r in gallery if r["source"] == "muraldh_sr")
    sr_leak = (sr_train & sr_query) | (sr_train & sr_gallery) | (sr_query & sr_gallery)
    stats["sr_group_check"] = {
        "total_groups": len(sr_train | sr_query | sr_gallery),
        "train_groups": len(sr_train),
        "query_groups": len(sr_query),
        "gallery_groups": len(sr_gallery),
        "cross_set_leakage": len(sr_leak),
        "leaked_groups": list(sr_leak)[:10] if sr_leak else [],
    }

    stats_path = os.path.join(SPLIT_DIR, "split_statistics.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {stats_path}")

    # ============ 打印汇总 ============
    print("\n" + "=" * 60)
    print("Step 3 数据集划分 — 结果汇总")
    print("=" * 60)

    print(f"\n总图像: {len(rows)}")
    print(f"  Train:        {len(train):6d}")
    print(f"  Query:        {len(query):6d}")
    print(f"  Gallery:      {len(gallery):6d} "
          f"(含 {total_distractors} distractors)")
    print(f"  Cross-source: {len(cross_source):6d} "
          f"({'+'.join(CROSS_SOURCE_HOLDOUT)})")

    print(f"\n--- Train 各来源 ---")
    for src, cnt in Counter(r["source"] for r in train).most_common():
        print(f"  {src:20s}: {cnt:5d}")

    print(f"\n--- Query 各来源 ---")
    for src, cnt in Counter(r["source"] for r in query).most_common():
        print(f"  {src:20s}: {cnt:5d}")

    print(f"\n--- Gallery 各来源 ---")
    for src, cnt in Counter(r["source"] for r in gallery).most_common():
        n_labeled = sum(1 for r in gallery if r["source"] == src and not r["is_distractor"])
        n_dist = sum(1 for r in gallery if r["source"] == src and r["is_distractor"])
        print(f"  {src:20s}: {cnt:5d} (labeled: {n_labeled}, distractor: {n_dist})")

    print(f"\n--- 跨源测试各来源 ---")
    for src, cnt in Counter(r["source"] for r in cross_source).most_common():
        print(f"  {src:20s}: {cnt:5d}")

    print(f"\n--- 类别分布 (train/query/gallery/cross) ---")
    for label in CATEGORY_NAMES:
        info = stats["class_distribution"][label]
        cn = info["cn"]
        print(f"  {label:20s} ({cn:6s}): "
              f"train={info['train']:4d}  "
              f"query={info['query']:4d}  "
              f"gallery={info['gallery']:4d}  "
              f"cross={info['cross_source']:4d}  "
              f"total={info['total']:5d}")

    print(f"\n--- 标签重叠 ---")
    lo = stats["label_overlap"]
    print(f"  Train ∩ Query:        {lo['train_vs_query']['overlap']} 类")
    print(f"  Train ∩ Gallery:      {lo['train_vs_gallery']['overlap']} 类")
    print(f"  Train ∩ Cross-source: {lo['train_vs_cross_source']['overlap']} 类")
    if lo['train_vs_cross_source']['cross_source_only_labels']:
        print(f"  [!] 跨源测试中有训练集未见类别: "
              f"{lo['train_vs_cross_source']['cross_source_only_labels']}")

    print(f"\n--- SR组完整性检查 ---")
    sr = stats["sr_group_check"]
    print(f"  总组数: {sr['total_groups']}, "
          f"train={sr['train_groups']}, query={sr['query_groups']}, "
          f"gallery={sr['gallery_groups']}")
    if sr["cross_set_leakage"] == 0:
        print(f"  [OK] 无跨集合泄漏")
    else:
        print(f"  [!] 发现 {sr['cross_set_leakage']} 个泄漏组!")

    print("\n[DONE] Step 3 完成。")


if __name__ == "__main__":
    main()
