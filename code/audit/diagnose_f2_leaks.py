#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断F2特征空间泄漏: 检查高相似度对的文件路径/来源/类别"""
import csv, os, numpy as np

SPLITS_DIR = r"<DATA_ROOT>\retrieval\splits"
FEATURES_PATH = r"<DATA_ROOT>\retrieval\features\dinov3_features.npy"
THRESHOLD = 0.99

def load_csv(name):
    path = os.path.join(SPLITS_DIR, name)
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["global_idx"])
            data[idx] = {"path": row["path"], "source": row["source"],
                         "label": row["label"], "confidence": float(row["confidence"]),
                         "label_type": row["label_type"]}
    return data

print("Loading splits...")
train = load_csv("train.csv")
gallery = load_csv("gallery.csv")
cross = load_csv("cross_source_test.csv")

print("Loading features...")
features = np.load(FEATURES_PATH)
norms = np.linalg.norm(features, axis=1, keepdims=True)
norms = np.where(norms == 0, 1, norms)
fn = features / norms

def find_leaks(s1, s2):
    i1, i2 = sorted(s1.keys()), sorted(s2.keys())
    f1, f2 = fn[i1], fn[i2]
    leaks = []
    for st in range(0, len(f1), 500):
        sims = f1[st:min(st+500,len(f1))] @ f2.T
        r, c = np.where(sims >= THRESHOLD)
        for rr, cc in zip(r, c):
            leaks.append((i1[st+rr], i2[cc], float(sims[rr, cc])))
    return leaks

print("\n" + "="*80)
print("TRAIN vs GALLERY leaks")
print("="*80)
ltg = find_leaks(train, gallery)
print(f"Found {len(ltg)} leak pairs")
for gi, gj, sim in ltg[:30]:
    t, g = train[gi], gallery[gj]
    print(f"  sim={sim:.4f} | train[{gi}] {t['source']}/{t['label']} {os.path.basename(t['path'])}")
    print(f"             | gallery[{gj}] {g['source']}/{g['label']} {os.path.basename(g['path'])}")

print("\n" + "="*80)
print("TRAIN vs CROSS leaks")
print("="*80)
ltc = find_leaks(train, cross)
print(f"Found {len(ltc)} leak pairs")
for gi, gj, sim in ltc[:30]:
    t, c = train[gi], cross[gj]
    print(f"  sim={sim:.4f} | train[{gi}] {t['source']}/{t['label']} {os.path.basename(t['path'])}")
    print(f"             | cross[{gj}] {c['source']}/{c['label']} {os.path.basename(c['path'])}")

# Pattern analysis
print("\n" + "="*80)
print("PATTERN: filename prefix match?")
print("="*80)
all_leaks = ltg + ltc
same_prefix = 0
diff_prefix = 0
sources = {}
for gi, gj, sim in all_leaks:
    p1 = train[gi]['path']
    p2 = gallery[gj]['path'] if gj in gallery else cross[gj]['path']
    b1, b2 = os.path.basename(p1), os.path.basename(p2)
    # muraldh pattern: img_XXXcrop_Y_Z.png - check if base image is same
    prefix1 = b1.rsplit('_crop', 1)[0] if '_crop' in b1 else b1.rsplit('_', 2)[0]
    prefix2 = b2.rsplit('_crop', 1)[0] if '_crop' in b2 else b2.rsplit('_', 2)[0]
    match = prefix1 == prefix2
    if match: same_prefix += 1
    else: diff_prefix += 1
    src1 = train[gi]['source']
    src2 = gallery[gj]['source'] if gj in gallery else cross[gj]['source']
    sources[(src1, src2)] = sources.get((src1, src2), 0) + 1
    if diff_prefix <= 5:
        print(f"  DIFF: {b1} vs {b2}")

print(f"\nSame-prefix pairs: {same_prefix}, Diff-prefix pairs: {diff_prefix}")
print(f"Source combinations: {sources}")
