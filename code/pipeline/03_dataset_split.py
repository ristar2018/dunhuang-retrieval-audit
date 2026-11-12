# -*- coding: utf-8 -*-
"""
Step 3: dataset splitting
Split the annotated data into train / query / gallery / cross_source_test

split strategy:
  1. cross-source generalisation set: all of deepflying and dunhuang_grottoes held out
     - reliable-label images from these two sources serve as queries, retrieved against the galleries of the remaining sources
     - used to test how well the model generalises across sources
  2. the other five sources (dh_v1, muraldh_512, muraldh_sr, hf_dunhuang, muraldh_seg)
     reliable-label images into train/query/gallery at 70/15/15, stratified
  3. stratified sampling: stratify jointly on label and source so every class keeps the same share in each split
  4. handling of the muraldh_sr super-resolution variants:
     - the X2/X3/X4/X6 variants of one source image must sit in one split, or information leaks
  5. uncertain images:
     - excluded from the train/query split
     - all of them join the gallery as distractors, which makes retrieval harder
     - flagged distractor = True, so it never counts as relevant at evaluation time

output:
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
    """Load the full label table produced by step 2"""
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
    extract the original-image group key of muraldh_sr so that the scale variants of one original image never straddle two sets.
    path pattern: .../Mural_SR/Mural_DataSet/0001.png  (X2/X3/X4/X6 sit in different subdirectories)
    the actual path may look like .../Mural_DataSet/X2/0001.png
    """
    if source != "muraldh_sr":
        return None
    # use the numeric part of the filename as the group key
    fname = os.path.basename(path)
    match = re.search(r'(\d+)', fname)
    if match:
        return f"sr_{match.group(1)}"
    return None


def split_by_stratified_group(records, train_ratio, query_ratio, gallery_ratio,
                              seed=42):
    """Legacy interface: stratified split for non-SR records (SR records are preprocessed in main)"""
    return split_by_stratified_group_with_sr(
        records, [], train_ratio, query_ratio, gallery_ratio, seed
    )


def split_by_stratified_group_with_sr(records, sr_units, train_ratio,
                                       query_ratio, gallery_ratio, seed=42):
    """
    Stratified grouped split (SR pre-grouped units are supported):
      - records: list of non-SR records, each an independent unit
      - sr_units: list of pre-grouped SR units, each a list of records (the whole group moves together)
      - stratification: joint stratification by (label, source)
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
    """Write the split CSVs"""
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
    """Compute the label overlap rate between the training and test sets"""
    train_set = set(train_labels)
    test_set = set(test_labels)
    overlap = train_set & test_set
    return len(overlap), len(test_set)


def main():
    print("=" * 60)
    print("Step 3: 数据集划分")
    print("=" * 60)

    rows = load_all_labels()

    # --- pre-process the muraldh_sr groups: keep the five variants of one group label-consistent and inside one split ---
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

    # per SR group: take the label of the HR original (root directory) if it has one, otherwise the highest-confidence label in the group
    # if the group carries a reliable label, the whole group takes that label and joins the split; otherwise the whole group becomes a distractor
    sr_reliable_units = []  # list of lists (groups with reliable labels)
    sr_distractor_units = []  # list of lists (whole groups uncertain)

    for gk, members in sr_groups.items():
        hr = [m for m in members if "Mural_DataSet_LR" not in m["path"]]
        reliable_members = [m for m in members if m["label_type"] != "uncertain"]

        if reliable_members:
            # prefer the HR original, then the group member with the highest confidence
            if hr and hr[0]["label_type"] != "uncertain":
                best = hr[0]
            else:
                best = max(reliable_members, key=lambda x: x["confidence"])
            group_label = best["label"]
            for m in members:
                m["label"] = group_label
                m["label_cn"] = CATEGORY_NAMES_CN.get(group_label, group_label)
                # keep the original label_type and confidence, but make the label uniform across the group
            sr_reliable_units.append(members)
        else:
            for m in members:
                m["label"] = "distractor"
                m["label_cn"] = "干扰项"
            sr_distractor_units.append(members)

    # separate the non-SR images into reliable and uncertain
    non_sr_reliable = [r for r in non_sr_rows if r["label_type"] != "uncertain"]
    non_sr_uncertain = [r for r in non_sr_rows if r["label_type"] == "uncertain"]

    # separate the cross-source hold-out
    cross_source = [r for r in non_sr_reliable if r["source"] in CROSS_SOURCE_HOLDOUT]
    main_source = [r for r in non_sr_reliable if r["source"] not in CROSS_SOURCE_HOLDOUT]

    # the split function is modified temporarily so the reliable SR groups can be injected into main_source as whole units
    # simple route: pass the SR groups in as pre-grouped records and extend the split function to accept them
    print(f"\n[INFO] 可靠标签: {len(non_sr_reliable) + sum(len(u) for u in sr_reliable_units)}, "
          f"uncertain: {len(non_sr_uncertain) + sum(len(u) for u in sr_distractor_units)}")
    print(f"[INFO] SR可靠组: {len(sr_reliable_units)}组 ({sum(len(u) for u in sr_reliable_units)}张), "
          f"SR干扰组: {len(sr_distractor_units)}组 ({sum(len(u) for u in sr_distractor_units)}张)")
    print(f"[INFO] 跨源留出 ({'+'.join(CROSS_SOURCE_HOLDOUT)}): {len(cross_source)}")
    print(f"[INFO] 主源数据(非SR): {len(main_source)}")

    # stratified group split: non-SR images plus the reliable SR groups
    print(f"\n[INFO] 按 70/15/15 分层划分主源数据...")
    train, query, gallery = split_by_stratified_group_with_sr(
        main_source, sr_reliable_units,
        TRAIN_RATIO, QUERY_RATIO, GALLERY_RATIO, RANDOM_SEED
    )

    # mark split and distractor
    for r in train:
        r["split"] = "train"
        r["is_distractor"] = False
    for r in query:
        r["split"] = "query"
        r["is_distractor"] = False
    for r in gallery:
        r["split"] = "gallery"
        r["is_distractor"] = False

    # cross-source data as extra queries (retrieved against the main-source gallery)
    for r in cross_source:
        r["split"] = "cross_source_query"
        r["is_distractor"] = False

    # uncertain non-SR images join the gallery as distractors
    for r in non_sr_uncertain:
        r["split"] = "gallery"
        r["is_distractor"] = True
        r["label"] = "distractor"
        r["label_cn"] = "干扰项"
    gallery.extend(non_sr_uncertain)

    # whole SR groups that are uncertain join the gallery as distractors (the group moves as a unit)
    for unit in sr_distractor_units:
        for r in unit:
            r["split"] = "gallery"
            r["is_distractor"] = True
            r["label"] = "distractor"
            r["label_cn"] = "干扰项"
        gallery.extend(unit)

    # make sure the directory exists
    os.makedirs(SPLIT_DIR, exist_ok=True)

    # write the CSV
    write_csv(TRAIN_CSV, train)
    write_csv(QUERY_CSV, query)
    write_csv(GALLERY_CSV, gallery)
    write_csv(CROSS_SRC_CSV, cross_source)

    # ============ statistics ============
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

    # per-split statistics
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

    # label-overlap analysis
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

    # samples per class (to expose class imbalance)
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

    # muraldh_sr group-integrity check
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

    # ============ summary printing ============
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
