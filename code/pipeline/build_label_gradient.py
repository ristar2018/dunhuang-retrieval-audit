#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
D1: build the label-quality gradient training sets (F1, revised)
Generate S0-S3 (the original tiers) plus S0'-S3' (the size-matched controls)

tier definitions:
  L0: label_type = seed, confidence = 1.0 -> about 0% noise
  L1: label_type = seed_knn, confidence = 1.0 -> about 1-3% noise
  L2: label_type = seed_knn, confidence < 1.0 -> about 10-15% noise
  L3: label_type = expanded_knn -> about 20-30% noise

S0 = L0 only
S1 = L0 + L1
S2 = L0 + L1 + L2
S3 = L0 + L1 + L2 + L3 (the whole train split)

control sets S0'-S3': randomly sampled from each tier down to the L0 sample size (2,278)

output: {S0,S1,S2,S3,S0_prime,S1_prime,S2_prime,S3_prime}_indices.json
     + gradient_metadata.json + build_log.txt
"""
import csv, os, json, time, sys
import numpy as np

SPLITS_DIR = r"<DATA_ROOT>\retrieval\splits"
OUTPUT_DIR = r"<DATA_ROOT>\retrieval\data\gradients"
SEED = 42

def load_train_csv():
    path = os.path.join(SPLITS_DIR, "train.csv")
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "global_idx": int(row["global_idx"]),
                "path": row["path"],
                "source": row["source"],
                "label": row["label"],
                "label_cn": row["label_cn"],
                "confidence": float(row["confidence"]),
                "label_type": row["label_type"],
            })
    return entries

def assign_layer(entry):
    """Assign label quality layer based on label_type and confidence"""
    lt = entry["label_type"]
    conf = entry["confidence"]
    if lt == "seed" and conf == 1.0:
        return "L0"
    elif lt == "seed_knn" and conf == 1.0:
        return "L1"
    elif lt == "seed_knn" and conf < 1.0:
        return "L2"
    elif lt == "expanded_knn":
        return "L3"
    else:
        # Fallback: seed with non-1.0 confidence → L1
        if lt == "seed":
            return "L1"
        return "L3"  # default to noisiest

def main():
    t0 = time.time()
    print("=" * 70)
    print("  D1: 构建标签质量梯度训练集 (F1修订版)")
    print("=" * 70)

    # [1] Load training data
    print("\n[1/5] 加载训练集...")
    train_data = load_train_csv()
    print(f"  训练集总数: {len(train_data)}")

    # [2] Assign layers
    print("\n[2/5] 分配标签质量层级...")
    layers = {"L0": [], "L1": [], "L2": [], "L3": []}
    for entry in train_data:
        layer = assign_layer(entry)
        layers[layer].append(entry)
    
    for layer_name in ["L0", "L1", "L2", "L3"]:
        indices = [e["global_idx"] for e in layers[layer_name]]
        print(f"  {layer_name}: {len(indices)} 条")
    
    total_layer = sum(len(v) for v in layers.values())
    assert total_layer == len(train_data), f"Layer sum {total_layer} != train {len(train_data)}"

    # [3] Build cumulative gradient sets
    print("\n[3/5] 构建梯度训练集...")
    L0_idx = [e["global_idx"] for e in layers["L0"]]
    L1_idx = [e["global_idx"] for e in layers["L1"]]
    L2_idx = [e["global_idx"] for e in layers["L2"]]
    L3_idx = [e["global_idx"] for e in layers["L3"]]

    S0 = sorted(L0_idx)
    S1 = sorted(L0_idx + L1_idx)
    S2 = sorted(L0_idx + L1_idx + L2_idx)
    S3 = sorted(L0_idx + L1_idx + L2_idx + L3_idx)

    print(f"  S0 (L0 only):           {len(S0)}")
    print(f"  S1 (L0+L1):             {len(S1)}")
    print(f"  S2 (L0+L1+L2):          {len(S2)}")
    print(f"  S3 (全量):              {len(S3)}")

    # [4] Build controlled sets (equal sample size = |L0|)
    print("\n[4/5] 构建等样本量控制组 (target={})...".format(len(S0)))
    rng = np.random.RandomState(SEED)
    target_size = len(S0)

    # S0' = L0 (exactly target_size)
    S0_prime = sorted(L0_idx)
    assert len(S0_prime) == target_size

    # S1' = sample target_size from L0+L1
    pool_S1 = L0_idx + L1_idx
    S1_prime = sorted(rng.choice(pool_S1, size=target_size, replace=False).tolist())

    # S2' = sample target_size from L0+L1+L2
    pool_S2 = L0_idx + L1_idx + L2_idx
    S2_prime = sorted(rng.choice(pool_S2, size=target_size, replace=False).tolist())

    # S3' = sample target_size from L0+L1+L2+L3
    pool_S3 = L0_idx + L1_idx + L2_idx + L3_idx
    S3_prime = sorted(rng.choice(pool_S3, size=target_size, replace=False).tolist())

    print(f"  S0' (=L0):              {len(S0_prime)}")
    print(f"  S1' (sampled from L0+L1): {len(S1_prime)}")
    print(f"  S2' (sampled from L0-L2): {len(S2_prime)}")
    print(f"  S3' (sampled from L0-L3): {len(S3_prime)}")

    # [5] Save outputs
    print("\n[5/5] 保存输出...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    sets = {
        "S0": S0, "S1": S1, "S2": S2, "S3": S3,
        "S0_prime": S0_prime, "S1_prime": S1_prime,
        "S2_prime": S2_prime, "S3_prime": S3_prime,
    }
    for name, indices in sets.items():
        path = os.path.join(OUTPUT_DIR, f"{name}_indices.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(indices, f)
        print(f"  Saved: {name}_indices.json ({len(indices)} entries)")

    # Metadata
    metadata = {
        "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_train": len(train_data),
        "layers": {k: len(v) for k, v in layers.items()},
        "gradient_sets": {k: len(v) for k, v in sets.items()},
        "control_group": {
            "target_size": target_size,
            "seed": SEED,
            "strategy": "random_sampling_without_replacement",
        },
        "estimated_noise_rates": {
            "S0": "0% (seed, confidence=1.0)",
            "S1": "1-3% (seed_knn unanimous)",
            "S2": "10-15% (seed_knn with disagreement)",
            "S3": "20-30% (expanded_knn)",
        },
        "noise_rate_numeric": {
            "S0": 0.0, "S1": 0.02, "S2": 0.125, "S3": 0.25,
            "S0_prime": 0.0, "S1_prime": 0.02, "S2_prime": 0.125, "S3_prime": 0.25,
        },
    }
    meta_path = os.path.join(OUTPUT_DIR, "gradient_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  Saved: gradient_metadata.json")

    # Build log
    elapsed = time.time() - t0
    log_path = os.path.join(OUTPUT_DIR, "build_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("D1 Build Log\n")
        f.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total train: {len(train_data)}\n\n")
        f.write("Layers:\n")
        for k, v in layers.items():
            f.write(f"  {k}: {len(v)}\n")
        f.write(f"\nGradient sets:\n")
        for k, v in sets.items():
            f.write(f"  {k}: {len(v)}\n")
        f.write(f"\nControl group: target_size={target_size}, seed={SEED}\n")
        f.write(f"\nElapsed: {elapsed:.1f}s\n")
    print(f"  Saved: build_log.txt")

    print(f"\n{'='*70}")
    print(f"  D1 构建完成! 耗时 {elapsed:.1f}s")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'='*70}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
