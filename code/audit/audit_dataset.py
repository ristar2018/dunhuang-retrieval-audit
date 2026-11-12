# -*- coding: utf-8 -*-
"""
数据标注审计脚本
检查项:
1. 数据泄露检查:
   - Train/Query/Gallery 路径重叠
   - muraldh_sr 组完整性
   - 跨源测试集是否混入 train/query
   - muraldh_512 裁切与 muraldh_seg/sr 的路径关联
   - 跨源同图检测 (DINOv2 特征余弦相似度 > 0.99)
2. 数据污染检查:
   - AI 生成图像是否混入
   - 路径是否全部为 real
   - 是否有损坏/异常文件
3. 标注质量评估:
   - 种子标签覆盖率和准确性
   - 传播标签置信度分布
   - 各类别标注质量
   - 高/低置信度样本统计
"""
import os, sys, json, csv, re
import numpy as np
import faiss
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CLEAN_INDEX, FEATURE_NPY, FEATURE_INDEX,
    ALL_LABELS, TRAIN_CSV, QUERY_CSV, GALLERY_CSV, CROSS_SRC_CSV,
    SPLIT_DIR, ANNO_DIR, CATEGORY_NAMES, CATEGORY_NAMES_CN,
    HF_CSV, HF_IMG_DIR, DATA_ROOT
)


def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_features():
    feats = np.load(FEATURE_NPY)
    with open(FEATURE_INDEX, "r", encoding="utf-8") as f:
        idx = json.load(f)
    return feats, idx


def check_path_leakage(train, query, gallery, cross):
    """检查路径级别的数据泄露"""
    print("=" * 60)
    print("[CHECK 1] 路径级别数据泄露检查")
    print("=" * 60)

    def paths(rows):
        return set(r["path"] for r in rows)

    p_train = paths(train)
    p_query = paths(query)
    p_gallery = paths(gallery)
    p_cross = paths(cross)

    print(f"  Train paths:   {len(p_train)}")
    print(f"  Query paths:   {len(p_query)}")
    print(f"  Gallery paths: {len(p_gallery)}")
    print(f"  Cross paths:   {len(p_cross)}")

    issues = []

    # Train-Query 重叠
    tq = p_train & p_query
    if tq:
        issues.append(f"Train-Query 路径重叠: {len(tq)}")
        print(f"  [FAIL] Train-Query overlap: {len(tq)}")
    else:
        print(f"  [OK] Train-Query 无路径重叠")

    # Train-Gallery 重叠
    tg = p_train & p_gallery
    if tg:
        issues.append(f"Train-Gallery 路径重叠: {len(tg)}")
        print(f"  [FAIL] Train-Gallery overlap: {len(tg)}")
    else:
        print(f"  [OK] Train-Gallery 无路径重叠")

    # Train-Cross 重叠
    tc = p_train & p_cross
    if tc:
        issues.append(f"Train-Cross 路径重叠: {len(tc)}")
        print(f"  [FAIL] Train-Cross overlap: {len(tc)}")
    else:
        print(f"  [OK] Train-Cross 无路径重叠")

    # Query-Gallery 重叠 (检索场景中query和gallery不应有同一张图)
    qg = p_query & p_gallery
    if qg:
        issues.append(f"Query-Gallery 路径重叠: {len(qg)}")
        print(f"  [FAIL] Query-Gallery overlap: {len(qg)}")
    else:
        print(f"  [OK] Query-Gallery 无路径重叠")

    # Query-Cross 重叠
    qc = p_query & p_cross
    if qc:
        issues.append(f"Query-Cross 路径重叠: {len(qc)}")
        print(f"  [FAIL] Query-Cross overlap: {len(qc)}")
    else:
        print(f"  [OK] Query-Cross 无路径重叠")

    # 各集合内部路径唯一性
    for name, rows in [("Train", train), ("Query", query),
                        ("Gallery", gallery), ("Cross", cross)]:
        if len(rows) != len(paths(rows)):
            dupes = len(rows) - len(paths(rows))
            issues.append(f"{name} 内部重复路径: {dupes}")
            print(f"  [FAIL] {name} internal duplicates: {dupes}")
        else:
            print(f"  [OK] {name} 内部路径唯一")

    return issues


def check_cross_source_holdout(train, query, cross):
    """检查跨源测试集是否正确留出"""
    print("\n" + "=" * 60)
    print("[CHECK 2] 跨源留出验证")
    print("=" * 60)

    train_sources = set(r["source"] for r in train)
    query_sources = set(r["source"] for r in query)
    cross_sources = set(r["source"] for r in cross)

    print(f"  Train sources: {sorted(train_sources)}")
    print(f"  Query sources: {sorted(query_sources)}")
    print(f"  Cross sources: {sorted(cross_sources)}")

    issues = []
    for src in ["deepflying", "dunhuang_grottoes"]:
        in_train = sum(1 for r in train if r["source"] == src)
        in_query = sum(1 for r in query if r["source"] == src)
        in_cross = sum(1 for r in cross if r["source"] == src)
        print(f"\n  {src}:")
        print(f"    Train: {in_train}, Query: {in_query}, Cross: {in_cross}")
        if in_train > 0:
            issues.append(f"{src} 出现在 Train 中 ({in_train}张)!")
            print(f"    [FAIL] 跨源数据泄漏到 Train!")
        if in_query > 0:
            issues.append(f"{src} 出现在 Query 中 ({in_query}张)!")
            print(f"    [FAIL] 跨源数据泄漏到 Query!")
        if in_train == 0 and in_query == 0 and in_cross > 0:
            print(f"    [OK] 正确留出")

    return issues


def check_sr_groups(train, query, gallery):
    """检查 muraldh_sr 组完整性"""
    print("\n" + "=" * 60)
    print("[CHECK 3] muraldh_sr 组完整性")
    print("=" * 60)

    def get_sr_keys(rows):
        keys = set()
        for r in rows:
            if r["source"] == "muraldh_sr":
                fname = os.path.basename(r["path"])
                m = re.search(r'(\d+)', fname)
                if m:
                    keys.add(f"sr_{m.group(1)}")
        return keys

    sr_train = get_sr_keys(train)
    sr_query = get_sr_keys(query)
    sr_gallery = get_sr_keys(gallery)
    all_sr = sr_train | sr_query | sr_gallery

    print(f"  Total SR groups: {len(all_sr)}")
    print(f"  Train: {len(sr_train)}, Query: {len(sr_query)}, Gallery: {len(sr_gallery)}")

    tq = sr_train & sr_query
    tg = sr_train & sr_gallery
    qg = sr_query & sr_gallery

    issues = []
    if tq:
        issues.append(f"SR组 Train-Query 泄漏: {len(tq)}")
        print(f"  [FAIL] Train-Query SR leak: {len(tq)}")
    else:
        print(f"  [OK] Train-Query SR no leak")

    if tg:
        issues.append(f"SR组 Train-Gallery 泄漏: {len(tg)}")
        print(f"  [FAIL] Train-Gallery SR leak: {len(tg)}")
    else:
        print(f"  [OK] Train-Gallery SR no leak")

    if qg:
        issues.append(f"SR组 Query-Gallery 泄漏: {len(qg)}")
        print(f"  [FAIL] Query-Gallery SR leak: {len(qg)}")
    else:
        print(f"  [OK] Query-Gallery SR no leak")

    # 检查每组5个变体是否在同一集合
    split_by_group = defaultdict(set)
    for r in train + query + gallery:
        if r["source"] == "muraldh_sr":
            fname = os.path.basename(r["path"])
            m = re.search(r'(\d+)', fname)
            if m:
                gk = f"sr_{m.group(1)}"
                if r in train:
                    split_by_group[gk].add("train")
                elif r in query:
                    split_by_group[gk].add("query")
                else:
                    split_by_group[gk].add("gallery")

    cross_split = {k: v for k, v in split_by_group.items() if len(v) > 1}
    if cross_split:
        issues.append(f"SR组跨集合: {len(cross_split)}")
        print(f"  [FAIL] {len(cross_split)} groups span multiple splits!")
    else:
        print(f"  [OK] 所有SR组变体在同一集合")

    # 检查每组变体数量
    group_sizes = defaultdict(int)
    for r in train + query + gallery:
        if r["source"] == "muraldh_sr":
            fname = os.path.basename(r["path"])
            m = re.search(r'(\d+)', fname)
            if m:
                group_sizes[f"sr_{m.group(1)}"] += 1

    size_dist = Counter(group_sizes.values())
    print(f"  组大小分布: {dict(sorted(size_dist.items()))}")

    return issues


def check_content_duplication(feats, feature_index, train, query, gallery):
    """
    检查跨集合内容重复 (DINOv2特征余弦相似度 > 0.99)
    排除muraldh_sr的同源变体（它们设计上就是相似的）
    """
    print("\n" + "=" * 60)
    print("[CHECK 4] 跨集合内容重复检测 (cosine sim > 0.99)")
    print("=" * 60)

    path_to_idx = {rec["path"]: rec["index"] for rec in feature_index["records"]}

    # 获取各集合的特征索引
    def get_feat_indices(rows):
        indices = []
        for r in rows:
            if r["path"] in path_to_idx:
                indices.append(path_to_idx[r["path"]])
        return np.array(indices, dtype=np.int64)

    train_idx = get_feat_indices(train)
    query_idx = get_feat_indices(query)
    gallery_labeled_idx = get_feat_indices(
        [r for r in gallery if r.get("is_distractor", "False") == "False"]
    )

    print(f"  Train features: {len(train_idx)}")
    print(f"  Query features: {len(query_idx)}")
    print(f"  Gallery(labeled) features: {len(gallery_labeled_idx)}")

    issues = []

    # Train vs Query: 搜索最近邻
    if len(train_idx) > 0 and len(query_idx) > 0:
        train_feats = feats[train_idx].astype(np.float32)
        query_feats = feats[query_idx].astype(np.float32)

        index = faiss.IndexFlatIP(feats.shape[1])
        index.add(train_feats)
        D, I = index.search(query_feats, 1)

        duplicates = np.where(D.flatten() > 0.99)[0]
        if len(duplicates) > 0:
            # 排除SR同源
            real_dupes = 0
            dupe_details = []
            for qi in duplicates:
                q_path = query[qi]["path"]
                t_path = train[I[qi][0]]["path"]
                # 如果都是muraldh_sr且数字ID相同，是预期内的
                q_src = query[qi]["source"]
                t_src = train[I[qi][0]]["source"]
                q_m = re.search(r'(\d+)', os.path.basename(q_path))
                t_m = re.search(r'(\d+)', os.path.basename(t_path))
                if q_src == "muraldh_sr" and t_src == "muraldh_sr" and q_m and t_m and q_m.group(1) == t_m.group(1):
                    continue
                real_dupes += 1
                if len(dupe_details) < 5:
                    dupe_details.append(f"    Q: {os.path.basename(q_path)} <-> T: {os.path.basename(t_path)} sim={D[qi][0]:.4f}")

            if real_dupes > 0:
                issues.append(f"Train-Query 内容重复(>0.99): {real_dupes}")
                print(f"  [FAIL] Train-Query near-duplicates: {real_dupes}")
                for d in dupe_details:
                    print(d)
            else:
                print(f"  [OK] Train-Query 无内容重复 (SR同源变体除外)")
        else:
            print(f"  [OK] Train-Query 无内容重复 (max sim = {D.max():.4f})")

    # Train vs Gallery(labeled)
    if len(train_idx) > 0 and len(gallery_labeled_idx) > 0:
        train_feats = feats[train_idx].astype(np.float32)
        gallery_feats = feats[gallery_labeled_idx].astype(np.float32)

        index = faiss.IndexFlatIP(feats.shape[1])
        index.add(train_feats)
        D, I = index.search(gallery_feats, 1)

        duplicates = np.where(D.flatten() > 0.99)[0]
        if len(duplicates) > 0:
            real_dupes = 0
            dupe_details = []
            for gi_pos in duplicates:
                g_path = [r for r in gallery if r.get("is_distractor","False")=="False"][gi_pos]["path"]
                t_path = train[I[gi_pos][0]]["path"]
                g_src = [r for r in gallery if r.get("is_distractor","False")=="False"][gi_pos]["source"]
                t_src = train[I[gi_pos][0]]["source"]
                g_m = re.search(r'(\d+)', os.path.basename(g_path))
                t_m = re.search(r'(\d+)', os.path.basename(t_path))
                if g_src == "muraldh_sr" and t_src == "muraldh_sr" and g_m and t_m and g_m.group(1) == t_m.group(1):
                    continue
                # muraldh_512 裁切可能与 muraldh_seg/sr 同原图
                real_dupes += 1
                if len(dupe_details) < 5:
                    dupe_details.append(f"    G: {os.path.basename(g_path)} <-> T: {os.path.basename(t_path)} sim={D[gi_pos][0]:.4f}")

            if real_dupes > 0:
                issues.append(f"Train-Gallery 内容重复(>0.99): {real_dupes}")
                print(f"  [WARN] Train-Gallery(labeled) near-duplicates: {real_dupes}")
                for d in dupe_details:
                    print(d)
            else:
                print(f"  [OK] Train-Gallery(labeled) 无内容重复")
        else:
            print(f"  [OK] Train-Gallery(labeled) 无内容重复 (max sim = {D.max():.4f})")

    return issues


def check_data_purity(all_rows):
    """检查数据污染"""
    print("\n" + "=" * 60)
    print("[CHECK 5] 数据污染检查")
    print("=" * 60)

    issues = []

    # 检查AI图像混入
    ai_paths = [r for r in all_rows if "ai_generated" in r["path"].lower()
                or r.get("source", "").startswith("ai_")]
    if ai_paths:
        issues.append(f"AI生成图像混入: {len(ai_paths)}")
        print(f"  [FAIL] AI-generated images found: {len(ai_paths)}")
    else:
        print(f"  [OK] 无AI生成图像")

    # 检查路径存在性
    missing = []
    for r in all_rows:
        if not os.path.exists(r["path"]):
            missing.append(r["path"])
    if missing:
        issues.append(f"文件不存在: {len(missing)}")
        print(f"  [FAIL] Missing files: {len(missing)}")
        for p in missing[:5]:
            print(f"    {p}")
    else:
        print(f"  [OK] 所有文件路径存在")

    # 检查模式
    modes = Counter(r.get("mode", "unknown") for r in all_rows)
    print(f"  图像模式: {dict(modes)}")

    # 检查label字段不为空
    empty_labels = [r for r in all_rows if not r.get("label", "").strip()]
    if empty_labels:
        issues.append(f"空标签: {len(empty_labels)}")
        print(f"  [FAIL] Empty labels: {len(empty_labels)}")
    else:
        print(f"  [OK] 无空标签")

    return issues


def assess_annotation_quality(all_rows, seeds_count):
    """评估标注质量"""
    print("\n" + "=" * 60)
    print("[CHECK 6] 标注质量评估")
    print("=" * 60)

    # 分离种子和传播标签
    seed_rows = [r for r in all_rows if r.get("label_type") == "seed"]
    knn_rows = [r for r in all_rows if r.get("label_type") in ("seed_knn", "expanded_knn")]
    uncertain_rows = [r for r in all_rows if r.get("label_type") == "uncertain"]

    print(f"\n  种子标签: {len(seed_rows)}")
    print(f"  传播标签: {len(knn_rows)}")
    print(f"  Uncertain: {len(uncertain_rows)}")

    # 种子标签质量
    print(f"\n  --- 种子标签质量 (hf_dunhuang ground truth) ---")
    seed_by_label = Counter(r["label"] for r in seed_rows)
    print(f"  类别覆盖: {len(seed_by_label)}/16")
    for cat in CATEGORY_NAMES:
        cnt = seed_by_label.get(cat, 0)
        print(f"    {cat:20s}: {cnt:4d}")

    # 传播标签置信度分布
    print(f"\n  --- 传播标签置信度分布 ---")
    if knn_rows:
        confs = np.array([float(r["confidence"]) for r in knn_rows])
        print(f"  样本数: {len(confs)}")
        print(f"  Mean:   {np.mean(confs):.4f}")
        print(f"  Median: {np.median(confs):.4f}")
        print(f"  Std:    {np.std(confs):.4f}")
        print(f"  Min:    {np.min(confs):.4f}")
        print(f"  Max:    {np.max(confs):.4f}")
        print(f"  >=0.9:  {np.mean(confs >= 0.9)*100:.1f}%")
        print(f"  >=0.8:  {np.mean(confs >= 0.8)*100:.1f}%")
        print(f"  >=0.7:  {np.mean(confs >= 0.7)*100:.1f}%")
        print(f"  0.6-0.7: {np.mean((confs >= 0.6) & (confs < 0.7))*100:.1f}%")

    # 按来源统计标注质量
    print(f"\n  --- 各来源标注质量 ---")
    source_stats = defaultdict(lambda: {"total": 0, "seed": 0, "knn": 0,
                                          "uncertain": 0, "conf_sum": 0.0})
    for r in all_rows:
        src = r["source"]
        source_stats[src]["total"] += 1
        lt = r.get("label_type", "")
        if lt == "seed":
            source_stats[src]["seed"] += 1
        elif lt in ("seed_knn", "expanded_knn"):
            source_stats[src]["knn"] += 1
            source_stats[src]["conf_sum"] += float(r["confidence"])
        else:
            source_stats[src]["uncertain"] += 1

    for src in sorted(source_stats.keys()):
        s = source_stats[src]
        rate = (s["seed"] + s["knn"]) / s["total"] * 100
        avg_conf = s["conf_sum"] / s["knn"] if s["knn"] > 0 else 0
        print(f"  {src:20s}: {s['total']:6d} total, "
              f"labeled={s['seed']+s['knn']:5d} ({rate:5.1f}%), "
              f"avg_conf={avg_conf:.3f}")

    # 按类别统计标注质量
    print(f"\n  --- 各类别标注质量 (可靠标签) ---")
    label_stats = defaultdict(lambda: {"count": 0, "conf_sum": 0.0,
                                         "seed": 0, "knn": 0})
    for r in knn_rows + seed_rows:
        lbl = r["label"]
        label_stats[lbl]["count"] += 1
        if r.get("label_type") == "seed":
            label_stats[lbl]["seed"] += 1
        else:
            label_stats[lbl]["knn"] += 1
            label_stats[lbl]["conf_sum"] += float(r["confidence"])

    for cat in CATEGORY_NAMES:
        s = label_stats.get(cat, {"count": 0, "conf_sum": 0, "seed": 0, "knn": 0})
        avg_conf = s["conf_sum"] / s["knn"] if s["knn"] > 0 else 1.0
        print(f"  {cat:20s} ({CATEGORY_NAMES_CN[cat]:6s}): "
              f"{s['count']:5d} total "
              f"(seed={s['seed']:4d}, knn={s['knn']:4d}), "
              f"avg_conf={avg_conf:.3f}")

    # 训练集标签分布平衡性
    print(f"\n  --- Train 类别平衡性 ---")
    train_rows = [r for r in all_rows if r.get("split") == "train"]
    train_labels = Counter(r["label"] for r in train_rows)
    total_train = len(train_rows)
    counts = list(train_labels.values())
    if counts:
        print(f"  总训练样本: {total_train}")
        print(f"  类别数: {len(train_labels)}")
        print(f"  最大类: {max(counts)} ({max(counts)/total_train*100:.1f}%)")
        print(f"  最小类: {min(counts)} ({min(counts)/total_train*100:.1f}%)")
        print(f"  不平衡比: {max(counts)/min(counts):.1f}:1")
        print(f"  Herfindahl index: {sum((c/total_train)**2 for c in counts):.4f}")

    return seed_rows, knn_rows, uncertain_rows


def check_feature_integrity(feats, feature_index):
    """检查特征矩阵完整性"""
    print("\n" + "=" * 60)
    print("[CHECK 7] 特征矩阵完整性")
    print("=" * 60)

    issues = []

    print(f"  Shape: {feats.shape}")
    print(f"  Dtype: {feats.dtype}")

    # 检查NaN/Inf
    nan_count = np.isnan(feats).sum()
    inf_count = np.isinf(feats).sum()
    if nan_count > 0:
        issues.append(f"NaN值: {nan_count}")
        print(f"  [FAIL] NaN: {nan_count}")
    else:
        print(f"  [OK] 无NaN")
    if inf_count > 0:
        issues.append(f"Inf值: {inf_count}")
        print(f"  [FAIL] Inf: {inf_count}")
    else:
        print(f"  [OK] 无Inf")

    # 检查L2归一化
    norms = np.linalg.norm(feats, axis=1)
    norm_ok = np.allclose(norms, 1.0, atol=1e-5)
    if norm_ok:
        print(f"  [OK] L2归一化正确 (norm range: {norms.min():.6f} ~ {norms.max():.6f})")
    else:
        issues.append("L2归一化异常")
        print(f"  [FAIL] L2 norm range: {norms.min():.6f} ~ {norms.max():.6f}")

    # 检查全零行
    zero_rows = np.where(norms < 0.5)[0]
    if len(zero_rows) > 0:
        issues.append(f"全零/异常行: {len(zero_rows)}")
        print(f"  [FAIL] Near-zero rows: {len(zero_rows)}")
    else:
        print(f"  [OK] 无全零行")

    # 索引一致性
    idx_count = feature_index["num_images"]
    if idx_count == feats.shape[0]:
        print(f"  [OK] 索引记录数({idx_count}) == 特征行数({feats.shape[0]})")
    else:
        issues.append(f"索引记录数({idx_count}) != 特征行数({feats.shape[0]})")
        print(f"  [FAIL] index count mismatch!")

    return issues


def main():
    print("=" * 60)
    print("敦煌壁画检索数据集 — 全面审计")
    print("=" * 60)

    # 加载数据
    print("\n[LOAD] 加载数据文件...")
    train = load_csv(TRAIN_CSV)
    query = load_csv(QUERY_CSV)
    gallery = load_csv(GALLERY_CSV)
    cross = load_csv(CROSS_SRC_CSV)
    all_rows = load_csv(ALL_LABELS)
    feats, feature_index = load_features()

    print(f"  Train: {len(train)}, Query: {len(query)}, "
          f"Gallery: {len(gallery)}, Cross: {len(cross)}")
    print(f"  All labels: {len(all_rows)}")
    print(f"  Features: {feats.shape}")

    all_issues = []

    # 执行检查
    all_issues.extend(check_path_leakage(train, query, gallery, cross))
    all_issues.extend(check_cross_source_holdout(train, query, cross))
    all_issues.extend(check_sr_groups(train, query, gallery))
    all_issues.extend(check_content_duplication(
        feats, feature_index, train, query, gallery))

    # 合并所有行用于污染检查
    split_rows = train + query + gallery + cross
    all_issues.extend(check_data_purity(split_rows))
    all_issues.extend(check_feature_integrity(feats, feature_index))

    seed_count = len([r for r in all_rows if r.get("label_type") == "seed"])
    assess_annotation_quality(all_rows, seed_count)

    # 最终总结
    print("\n" + "=" * 60)
    print("审计总结")
    print("=" * 60)

    if not all_issues:
        print("\n  [ALL PASS] 未发现数据泄露或污染问题。")
    else:
        print(f"\n  发现 {len(all_issues)} 个问题:")
        for i, issue in enumerate(all_issues, 1):
            print(f"    {i}. {issue}")

    print()
    return len(all_issues)


if __name__ == "__main__":
    n_issues = main()
    sys.exit(1 if n_issues > 0 else 0)
