#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查dh_v1和dunhuang_grottoes之间的完整重叠情况"""
import csv, os, re

SPLITS_DIR = r"<DATA_ROOT>\retrieval\splits"

def load_csv(name):
    path = os.path.join(SPLITS_DIR, name)
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            idx = int(row["global_idx"])
            data[idx] = {"path": row["path"], "source": row["source"],
                         "label": row["label"], "confidence": float(row["confidence"])}
    return data

all_data = {}
for fname in ["train.csv", "query.csv", "gallery.csv", "cross_source_test.csv"]:
    d = load_csv(fname)
    for idx, info in d.items():
        info["split"] = fname.replace(".csv", "")
        all_data[idx] = info

# Extract image number from dh_v1 and dunhuang_grottoes
dh_v1 = {}  # number -> [(global_idx, split, path, label)]
dh_grottoes = {}

for idx, info in all_data.items():
    src = info["source"]
    path = info["path"]
    basename = os.path.basename(path)
    
    if src == "dh_v1":
        # Dunhuang_Grottoes_Painting_00018.png -> 18
        m = re.search(r'(\d+)\.png$', basename)
        if m:
            num = int(m.group(1))
            dh_v1.setdefault(num, []).append((idx, info["split"], basename, info["label"]))
    
    elif src == "dunhuang_grottoes":
        # 018.jpg -> 18
        m = re.search(r'(\d+)\.jpg$', basename)
        if m:
            num = int(m.group(1))
            dh_grottoes.setdefault(num, []).append((idx, info["split"], basename, info["label"]))

print(f"dh_v1 unique numbers: {len(dh_v1)}")
print(f"dunhuang_grottoes unique numbers: {len(dh_grottoes)}")

# Count by split
for name, data in [("dh_v1", dh_v1), ("dunhuang_grottoes", dh_grottoes)]:
    splits = {}
    for num, entries in data.items():
        for idx, split, basename, label in entries:
            splits[split] = splits.get(split, 0) + 1
    print(f"  {name} distribution: {splits}")

# Find overlaps (same number exists in both sources)
common_nums = set(dh_v1.keys()) & set(dh_grottoes.keys())
print(f"\nOverlap numbers (same image in both sources): {len(common_nums)}")

# Check how many of these have different splits (i.e., actual leaks)
leak_details = []
safe_details = []  # same split = not a leak
for num in sorted(common_nums):
    for idx1, split1, p1, l1 in dh_v1[num]:
        for idx2, split2, p2, l2 in dh_grottoes[num]:
            if split1 != split2:
                leak_details.append((num, idx1, split1, idx2, split2, l1, l2))
            else:
                safe_details.append((num, idx1, split1, idx2, split2, l1, l2))

print(f"  Same-split pairs (no leak): {len(safe_details)}")
print(f"  Cross-split pairs (LEAK): {len(leak_details)}")

# Show leak distribution
leak_splits = {}
for num, idx1, s1, idx2, s2, l1, l2 in leak_details:
    key = f"{s1} <-> {s2}"
    leak_splits[key] = leak_splits.get(key, 0) + 1
print(f"\nLeak split distribution:")
for k, v in sorted(leak_splits.items()):
    print(f"  {k}: {v}")

# List all cross-split leak pairs
print(f"\nAll cross-split leak pairs ({len(leak_details)}):")
for num, idx1, s1, idx2, s2, l1, l2 in leak_details[:40]:
    label_match = "SAME" if l1 == l2 else f"DIFF({l1} vs {l2})"
    print(f"  #{num:03d}: dh_v1[{idx1}]({s1},{l1}) <-> dh_grottoes[{idx2}]({s2},{l2}) [{label_match}]")
if len(leak_details) > 40:
    print(f"  ... and {len(leak_details)-40} more")
