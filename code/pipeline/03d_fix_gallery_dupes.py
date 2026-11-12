# -*- coding: utf-8 -*-
"""
Step 3d: fix the Train-Gallery labeled near-duplicates the audit found
Turn the labeled gallery images with cosine sim >= 0.99 against train into distractors
these images entered when step 3c moved them from cross-source into the gallery; they do not change the gallery size,
only the is_distractor flag is set to True, so the image never counts as relevant at evaluation time
"""
import os, sys, csv, json
import numpy as np
import faiss
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FEATURE_NPY, FEATURE_INDEX,
    TRAIN_CSV, GALLERY_CSV, SPLIT_DIR
)

SIM_THRESHOLD = 0.99

def load_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    return rows, fieldnames

def save_csv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    print("=" * 60)
    print("Step 3d: 修复 Train-Gallery labeled 近重复")
    print("=" * 60)

    # Load features and index
    feats = np.load(FEATURE_NPY)
    with open(FEATURE_INDEX, "r", encoding="utf-8") as f:
        idx = json.load(f)
    path_to_feat = {r["path"]: feats[r["index"]] for r in idx["records"]}

    # Load splits
    train_rows, tf = load_csv(TRAIN_CSV)
    gallery_rows, gf = load_csv(GALLERY_CSV)
    print(f"Train: {len(train_rows)}, Gallery: {len(gallery_rows)}")

    # Build train feature matrix
    train_feats = np.array([path_to_feat[r["path"]] for r in train_rows], dtype=np.float32)
    print(f"Train features: {train_feats.shape}")

    # Build faiss index for train
    index = faiss.IndexFlatIP(train_feats.shape[1])
    index.add(train_feats)

    # Find gallery labeled items that are near-duplicates of train
    converted = 0
    sim_values = []
    for r in gallery_rows:
        if r.get("is_distractor", "False") in ("True", "true", "1"):
            continue
        qf = path_to_feat[r["path"]].reshape(1, -1).astype(np.float32)
        sims, _ = index.search(qf, 1)
        max_sim = sims[0][0]
        if max_sim >= SIM_THRESHOLD:
            r["is_distractor"] = "True"
            converted += 1
            sim_values.append(max_sim)

    print(f"\nConverted {converted} gallery labeled -> distractor (sim >= {SIM_THRESHOLD})")
    if sim_values:
        print(f"  sim range: {min(sim_values):.4f} - {max(sim_values):.4f}, "
              f"mean: {np.mean(sim_values):.4f}")

    # Verify after fix
    gallery_labeled = [r for r in gallery_rows
                       if r.get("is_distractor", "False") not in ("True", "true", "1")]
    print(f"\nAfter fix: Gallery labeled: {len(gallery_labeled)}, "
          f"distractors: {len(gallery_rows) - len(gallery_labeled)}")

    # Re-verify max sim
    if gallery_labeled:
        gl_feats = np.array([path_to_feat[r["path"]] for r in gallery_labeled], dtype=np.float32)
        sims, _ = index.search(gl_feats, 1)
        max_sims = sims[:, 0]
        n_dup = int(np.sum(max_sims >= SIM_THRESHOLD))
        print(f"Re-verify: Train-Gallery(labeled) max sim = {max_sims.max():.6f}, "
              f"dupes(>=0.99): {n_dup}")

    # Save updated gallery
    save_csv(GALLERY_CSV, gallery_rows, gf)
    print(f"\n[SAVE] {GALLERY_CSV}")

    # Final stats
    print("\nFinal dataset:")
    print(f"  Train:          {len(train_rows)}")
    print(f"  Query:          (unchanged)")
    print(f"  Gallery:        {len(gallery_rows)} "
          f"(labeled: {len(gallery_labeled)}, distractor: {len(gallery_rows)-len(gallery_labeled)})")

    print("\n[DONE] Step 3d fix complete.")

if __name__ == "__main__":
    main()
