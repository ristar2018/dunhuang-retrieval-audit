#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据划分互斥性验证 (F2 - 严谨性审查修订)
验证 Train/Query/Gallery/Cross 四个集合的互斥性:
  Check 1: 索引级互斥 (global_idx无交集)
  Check 2: 文件路径级不重复 (path无交集)
  Check 3: 特征空间泄漏检测 (跨集合cosine sim < 0.99)

数据源: D:\\XiaBao\\AIArtDetect\\retrieval\\splits\\ 下的4个CSV文件
"""

import os, sys, csv, time
import numpy as np

# ── 路径配置 ──
SPLITS_DIR = r"<DATA_ROOT>\retrieval\splits"
FEATURES_PATH = r"<DATA_ROOT>\retrieval\features\dinov3_features.npy"
REPORT_PATH = r"<DATA_ROOT>\retrieval\splits\split_verification_report.txt"
COS_THRESHOLD = 0.99

SPLIT_FILES = {
    "train": "train.csv",
    "query": "query.csv",
    "gallery": "gallery.csv",
    "cross": "cross_source_test.csv",
}


def load_csv(filename):
    """读取CSV, 返回 (global_idx_list, path_list)"""
    fpath = os.path.join(SPLITS_DIR, filename)
    if not os.path.exists(fpath):
        print(f"  [X] 文件不存在: {fpath}")
        return None, None
    idxs, paths = [], []
    with open(fpath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idxs.append(int(row["global_idx"]))
            paths.append(row["path"])
    return idxs, paths


def check_1_index_exclusivity(sets):
    """Check 1: 索引级互斥"""
    print("\n" + "=" * 70)
    print("[Check 1/3] 索引级互斥 (global_idx)")
    print("=" * 70)
    names = list(sets.keys())
    all_pass = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            s1, s2 = sets[n1]["idx_set"], sets[n2]["idx_set"]
            overlap = s1 & s2
            status = "PASS" if len(overlap) == 0 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {n1} vs {n2}: |{n1}|={len(s1)}, |{n2}|={len(s2)}, "
                  f"overlap={len(overlap)} [{status}]")
            if overlap and len(overlap) <= 10:
                for idx in sorted(overlap):
                    print(f"    - global_idx={idx}")
    return all_pass


def check_2_path_exclusivity(sets):
    """Check 2: 文件路径级不重复"""
    print("\n" + "=" * 70)
    print("[Check 2/3] 文件路径级不重复")
    print("=" * 70)
    names = list(sets.keys())
    all_pass = True
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            s1, s2 = sets[n1]["path_set"], sets[n2]["path_set"]
            overlap = s1 & s2
            status = "PASS" if len(overlap) == 0 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {n1} vs {n2}: overlap={len(overlap)} [{status}]")
            if overlap and len(overlap) <= 10:
                for p in sorted(overlap)[:5]:
                    print(f"    - {p}")
    return all_pass


def check_3_feature_leakage(sets, features):
    """Check 3: 特征空间泄漏检测 (批量cosine)"""
    print("\n" + "=" * 70)
    print("[Check 3/3] 特征空间泄漏检测 (cosine sim >= {:.2f})".format(COS_THRESHOLD))
    print("=" * 70)

    # 归一化特征
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    features_norm = features / norms

    names = list(sets.keys())
    all_pass = True
    batch_size = 500

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            idx1 = sorted(sets[n1]["idx_list"])
            idx2 = sorted(sets[n2]["idx_list"])

            f1 = features_norm[idx1]  # (N1, D)
            f2 = features_norm[idx2]  # (N2, D)

            max_sim = -1.0
            leak_pairs = []

            # 分批计算避免OOM
            for start in range(0, len(f1), batch_size):
                end = min(start + batch_size, len(f1))
                batch = f1[start:end]  # (B, D)
                sims = batch @ f2.T  # (B, N2)
                batch_max = sims.max()
                if batch_max > max_sim:
                    max_sim = batch_max
                # 记录泄漏对
                leak_mask = sims >= COS_THRESHOLD
                if leak_mask.any():
                    rows, cols = np.where(leak_mask)
                    for r, c in zip(rows, cols):
                        global_i = idx1[start + r]
                        global_j = idx2[c]
                        leak_pairs.append((global_i, global_j, float(sims[r, c])))

            status = "PASS" if max_sim < COS_THRESHOLD else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {n1}({len(idx1)}) vs {n2}({len(idx2)}): "
                  f"max_cos={max_sim:.4f} [{status}]")
            if leak_pairs:
                print(f"    泄漏对数: {len(leak_pairs)}")
                for gi, gj, s in leak_pairs[:5]:
                    print(f"      idx {gi} vs {gj}: sim={s:.4f}")

    return all_pass


def main():
    t0 = time.time()
    print("=" * 70)
    print("  数据划分互斥性验证 (F2)")
    print("=" * 70)

    # [1] 加载数据
    print("\n[1/4] 加载数据...")
    sets = {}
    total = 0
    for name, fname in SPLIT_FILES.items():
        idxs, paths = load_csv(fname)
        if idxs is None:
            print(f"\n[FATAL] 缺少 {fname}，无法继续验证")
            return 1
        sets[name] = {
            "idx_list": idxs,
            "idx_set": set(idxs),
            "path_list": paths,
            "path_set": set(paths),
        }
        print(f"  {name}: {len(idxs)} 条")
        total += len(idxs)
    print(f"  总计: {total} 条")

    # 总数校验
    if total != 32415:
        print(f"\n  [WARN] 总数 {total} != 预期 32415")
    else:
        print(f"  [OK] 总数 32415 校验通过")

    # [2] Check 1: 索引互斥
    pass1 = check_1_index_exclusivity(sets)

    # [3] Check 2: 路径互斥
    pass2 = check_2_path_exclusivity(sets)

    # [4] Check 3: 特征空间泄漏
    print("\n[4/4] 加载特征矩阵...")
    if not os.path.exists(FEATURES_PATH):
        print(f"  [X] 特征文件不存在: {FEATURES_PATH}")
        print("  跳过 Check 3")
        pass3 = None
    else:
        features = np.load(FEATURES_PATH)
        print(f"  特征矩阵: {features.shape}")
        if features.shape[0] != 32415:
            print(f"  [WARN] 特征数 {features.shape[0]} != 32415")
        pass3 = check_3_feature_leakage(sets, features)
        del features  # 释放内存

    # ── 汇总 ──
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("  验证结果汇总")
    print("=" * 70)
    print(f"  Check 1 (索引互斥):     {'PASS' if pass1 else 'FAIL'}")
    print(f"  Check 2 (路径不重复):   {'PASS' if pass2 else 'FAIL'}")
    if pass3 is not None:
        print(f"  Check 3 (特征空间<0.99): {'PASS' if pass3 else 'FAIL'}")
    else:
        print(f"  Check 3 (特征空间<0.99): SKIPPED (特征文件缺失)")
    overall = pass1 and pass2 and (pass3 if pass3 is not None else True)
    print(f"\n  总体结果: {'ALL PASS - 数据划分通过互斥性验证' if overall else 'FAIL - 存在数据泄漏风险!'}")
    print(f"  耗时: {elapsed:.1f}s")
    print("=" * 70)

    # ── 写报告 ──
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("数据划分互斥性验证报告 (F2)\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总样本: {total}\n\n")
        for name in sets:
            f.write(f"  {name}: {len(sets[name]['idx_list'])} 条\n")
        f.write(f"\nCheck 1 (索引互斥):     {'PASS' if pass1 else 'FAIL'}\n")
        f.write(f"Check 2 (路径不重复):   {'PASS' if pass2 else 'FAIL'}\n")
        f.write(f"Check 3 (特征空间<0.99): {'PASS' if pass3 else 'FAIL' if pass3 is not None else 'SKIPPED'}\n")
        f.write(f"\n总体: {'ALL PASS' if overall else 'FAIL'}\n")
    print(f"\n报告已保存: {REPORT_PATH}")

    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
