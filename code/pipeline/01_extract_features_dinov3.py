# -*- coding: utf-8 -*-
"""
Step 1 (DINOv3): feature extraction
Extract 768-dimensional feature vectors for every real Dunhuang mural image with DINOv3 ViT-B/16
- model: local dinov3-vitb16 (AutoModel, 85.7M parameters, LVD-1689M pretrained)
- preprocessing: AutoImageProcessor (resize 256 + center crop 224 + normalize)
- precision: fp16 GPU inference
- output: features/dinov3_features.npy plus features/feature_index.json
- DINOv2 features (dinov2_features.npy) kept as a comparison baseline
"""
import os, sys, json, time
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoImageProcessor

# load the configuration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    CLEAN_INDEX, DINOV3_PATH, FEATURE_NPY, FEATURE_INDEX,
    FEATURE_DIM, INPUT_SIZE, BATCH_SIZE, ensure_dirs, BACKBONE_NAME
)

def load_records():
    """Load every real mural record from clean_index.json"""
    with open(CLEAN_INDEX, "r", encoding="utf-8") as f:
        data = json.load(f)
    real = [r for r in data["records"] if r["class"] == "real"]
    print(f"[INFO] 总记录数: {len(data['records'])}, 真实壁画: {len(real)}")
    return real

def filter_valid(records):
    """Drop files that are missing or corrupt"""
    valid, invalid = [], []
    for i, r in enumerate(records):
        p = r["path"]
        if not os.path.exists(p):
            invalid.append({"path": p, "reason": "file_not_found"})
            continue
        try:
            with Image.open(p) as img:
                img.verify()
            valid.append(r)
        except Exception as e:
            invalid.append({"path": p, "reason": str(e)[:100]})
        if (i+1) % 5000 == 0:
            print(f"  路径检查: {i+1}/{len(records)}, 有效: {len(valid)}, 无效: {len(invalid)}")
    print(f"[INFO] 有效文件: {len(valid)}, 无效/缺失: {len(invalid)}")
    return valid, invalid

@torch.no_grad()
def extract_features(valid_records):
    """Extract DINOv3 features in batches"""
    print(f"[INFO] 加载模型: {DINOV3_PATH}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(DINOV3_PATH)
    model = AutoModel.from_pretrained(DINOV3_PATH, torch_dtype=torch.float16).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 模型: {type(model).__name__}, 参数: {total_params/1e6:.1f}M")
    print(f"[INFO] 设备: {device}, 精度: float16")
    print(f"[INFO] Hidden size: {model.config.hidden_size}, "
          f"Patch: {model.config.patch_size}, "
          f"Registers: {getattr(model.config, 'num_register_tokens', 'N/A')}")

    all_feats = np.zeros((len(valid_records), FEATURE_DIM), dtype=np.float32)
    failed = []
    t0 = time.time()

    for start in range(0, len(valid_records), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(valid_records))
        batch_records = valid_records[start:end]
        images = []
        batch_valid_idx = []

        for local_i, r in enumerate(batch_records):
            try:
                img = Image.open(r["path"]).convert("RGB")
                images.append(img)
                batch_valid_idx.append(local_i)
            except Exception as e:
                failed.append({"path": r["path"], "reason": str(e)[:100]})

        if not images:
            continue

        inputs = processor(images=images, return_tensors="pt").to(device)
        # fp16 input
        inputs["pixel_values"] = inputs["pixel_values"].half()
        outputs = model(**inputs)
        # DINOv3: the CLS token sits at position 0, the four register tokens follow it, then the patch tokens
        # last_hidden_state[:, 0] = CLS token (the global feature)
        feats = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()

        for j, vi in enumerate(batch_valid_idx):
            global_idx = start + vi
            all_feats[global_idx] = feats[j]

        if (start // BATCH_SIZE) % 20 == 0 or end == len(valid_records):
            elapsed = time.time() - t0
            speed = end / elapsed if elapsed > 0 else 0
            eta = (len(valid_records) - end) / speed if speed > 0 else 0
            print(f"  [{end}/{len(valid_records)}] "
                  f"{elapsed:.0f}s elapsed, {speed:.1f} img/s, ETA {eta:.0f}s")

    # drop all-zero rows (samples whose extraction failed completely)
    nonzero_mask = np.linalg.norm(all_feats, axis=1) > 0
    all_feats = all_feats[nonzero_mask]
    valid_records = [r for i, r in enumerate(valid_records) if nonzero_mask[i]]

    # L2 normalisation (so retrieval can use cosine similarity)
    norms = np.linalg.norm(all_feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    all_feats = all_feats / norms

    print(f"[INFO] 特征矩阵: {all_feats.shape}, dtype: {all_feats.dtype}")
    print(f"[INFO] 提取失败: {len(failed)}")

    # GPU cleanup
    del model, outputs, inputs
    torch.cuda.empty_cache()

    return all_feats, valid_records, failed

def main():
    ensure_dirs()
    print("=" * 60)
    print("Step 1: DINOv3 ViT-B/16 特征提取")
    print("=" * 60)

    records = load_records()
    valid, invalid = filter_valid(records)

    all_feats, final_records, failed = extract_features(valid)

    # save the feature matrix
    np.save(FEATURE_NPY, all_feats)
    print(f"[SAVE] 特征矩阵 -> {FEATURE_NPY} "
          f"({os.path.getsize(FEATURE_NPY)/1024/1024:.1f} MB)")

    # save the index
    index = {
        "num_images": len(final_records),
        "feature_dim": FEATURE_DIM,
        "model": BACKBONE_NAME,
        "model_path": DINOV3_PATH,
        "input_size": INPUT_SIZE,
        "normalized": True,
        "records": [
            {
                "index": i,
                "path": r["path"],
                "source": r["source"],
                "width": r["width"],
                "height": r["height"],
            }
            for i, r in enumerate(final_records)
        ],
        "invalid_files": invalid,
        "extraction_failed": failed,
    }
    with open(FEATURE_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] 特征索引 -> {FEATURE_INDEX}")

    # statistics by source
    from collections import Counter
    src_counts = Counter(r["source"] for r in final_records)
    print("\n[STATS] 各来源有效图像数:")
    for k, v in src_counts.most_common():
        print(f"  {k}: {v}")

    print("\n[DONE] Step 1 (DINOv3) 完成。")

if __name__ == "__main__":
    main()
