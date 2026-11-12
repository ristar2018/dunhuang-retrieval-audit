# -*- coding: utf-8 -*-
"""
Step 3c: 清理跨源测试集中与train重复的dunhuang_grottoes图像
- sim >= 0.99 的142张从cross_source移到gallery labeled
- 保留deepflying全部(max sim 0.9584，无完全重复)
- 保留dunhuang_grottoes中sim<0.99的228张
"""
import csv, json, os
import numpy as np
import faiss

RET = r"<DATA_ROOT>\retrieval"
FEAT = os.path.join(RET, "features", "dinov2_features.npy")
FIDX = os.path.join(RET, "features", "feature_index.json")
SIM_THRESHOLD = 0.99

def load_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

print("=" * 60)
print("Step 3c: 清理跨源测试集重复")
print("=" * 60)

train = load_csv(os.path.join(RET, "splits", "train.csv"))
query = load_csv(os.path.join(RET, "splits", "query.csv"))
gallery = load_csv(os.path.join(RET, "splits", "gallery.csv"))
cross = load_csv(os.path.join(RET, "splits", "cross_source_test.csv"))

with open(FIDX, "r", encoding="utf-8") as f:
    fidx = json.load(f)
path2idx = {rec["path"]: rec["index"] for rec in fidx["records"]}
feats = np.load(FEAT)

# 构建train特征索引
train_idx = np.array([path2idx[r["path"]] for r in train], dtype=np.int64)
index = faiss.IndexFlatIP(768)
index.add(feats[train_idx].astype(np.float32))

# 检查cross中每张图
cross_keep = []
cross_move = []
for r in cross:
    idx = path2idx[r["path"]]
    D, _ = index.search(feats[idx:idx+1].astype(np.float32), 1)
    sim = float(D[0][0])
    if sim >= SIM_THRESHOLD:
        cross_move.append(r)
    else:
        cross_keep.append(r)

print(f"\n跨源测试集: {len(cross)} -> 保留 {len(cross_keep)}, 移出 {len(cross_move)}")

# 按来源细分
from collections import Counter
move_by_src = Counter(r["source"] for r in cross_move)
print(f"移出来源: {dict(move_by_src)}")

# 将移出的图像加入gallery labeled
for r in cross_move:
    r["split"] = "gallery"
    r["is_distractor"] = False
gallery_extended = gallery + cross_move

# 保存
FIELDNAMES = [
    "global_idx", "path", "source",
    "label", "label_cn", "confidence",
    "label_type", "split", "is_distractor", "vote_details"
]

write_csv(os.path.join(RET, "splits", "cross_source_test.csv"), cross_keep, FIELDNAMES)
write_csv(os.path.join(RET, "splits", "gallery.csv"), gallery_extended, FIELDNAMES)

# 最终验证
print("\n" + "=" * 60)
print("最终数据集划分")
print("=" * 60)
n_gallery_labeled = sum(1 for r in gallery_extended if r["is_distractor"] == "False")
n_dist = sum(1 for r in gallery_extended if r["is_distractor"] == "True")
print(f"  Train:        {len(train):6d}")
print(f"  Query:        {len(query):6d}")
print(f"  Gallery:      {len(gallery_extended):6d} (labeled: {n_gallery_labeled}, distractor: {n_dist})")
print(f"  Cross-source: {len(cross_keep):6d}")
print(f"  Total:        {len(train)+len(query)+len(gallery_extended)+len(cross_keep):6d}")

# 跨源来源分布
print(f"\n--- Cross-source 各来源 ---")
for src, cnt in Counter(r["source"] for r in cross_keep).most_common():
    print(f"  {src:20s}: {cnt:5d}")

# 最终重复验证
cross_keep_idx = np.array([path2idx[r["path"]] for r in cross_keep], dtype=np.int64)
D_final, _ = index.search(feats[cross_keep_idx].astype(np.float32), 1)
n99 = int((D_final.flatten() >= 0.99).sum())
n95 = int((D_final.flatten() >= 0.95).sum())
print(f"\n[VERIFY] 清理后跨源测试集 vs Train:")
print(f"  sim>=0.99: {n99}")
print(f"  sim>=0.95: {n95}")
print(f"  max sim:   {D_final.max():.4f}")

# 类别分布
print(f"\n--- 跨源测试集类别分布 ---")
for label, cnt in Counter(r["label"] for r in cross_keep).most_common():
    print(f"  {label:20s}: {cnt:4d}")

# 更新split_statistics.json
stats = {
    "total_images": len(train) + len(query) + len(gallery_extended) + len(cross_keep),
    "splits": {
        "train": {"total": len(train)},
        "query": {"total": len(query)},
        "gallery": {"total": len(gallery_extended),
                    "labeled": n_gallery_labeled,
                    "distractor": n_dist},
        "cross_source_test": {"total": len(cross_keep)},
    },
    "dedup": {
        "sim_threshold": SIM_THRESHOLD,
        "total_dup_pairs_found": 15381,
        "dup_clusters": 6478,
        "images_involved": 16305,
        "reassigned": 1174,
        "cross_source_removed": len(cross_move),
        "final_train_query_max_sim": 0.9837,
        "final_train_gallery_max_sim": 0.9890,
        "final_cross_train_max_sim": round(float(D_final.max()), 4),
    },
    "sr_groups": {"total": 512, "train": 179, "query": 38, "gallery": 295, "leakage": 0},
}
with open(os.path.join(RET, "splits", "split_statistics.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(f"\n[SAVE] split_statistics.json updated")

print("\n[DONE] 跨源测试集清理完成。")
