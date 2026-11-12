# -*- coding: utf-8 -*-
"""
Step 4: Dataset & DataLoader
Dunhuang mural retrieval dataset, with training-augmentation and evaluation modes
"""
import os, csv
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_CSV, QUERY_CSV, GALLERY_CSV, CROSS_SRC_CSV,
    DINOV3_PATH, INPUT_SIZE, TRAIN_BS, EVAL_BS, NUM_WORKERS,
    TRAIN_AUG, COLOR_JITTER, RANDOM_ERASING, CATEGORY_NAMES
)
from transformers import AutoImageProcessor


class DunhuangDataset(Dataset):
    """Dunhuang mural retrieval dataset"""

    def __init__(self, csv_path, transform=None, is_train=False,
                 include_distractors=False):
        self.rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not include_distractors:
                    is_dist = row.get("is_distractor", "False")
                    if is_dist in ("True", "true", "1"):
                        continue
                self.rows.append(row)
        self.transform = transform
        self.is_train = is_train
        self.label_map = {name: i for i, name in enumerate(CATEGORY_NAMES)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.label_map.get(row["label"], -1)
        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long),
            "path": row["path"],
            "source": row["source"],
            "global_idx": int(row["global_idx"]),
        }


def get_transforms(phase="train"):
    """
    using the AutoImageProcessor preprocessing parameters of DINOv3,
    light augmentation is applied during training.
    """
    processor = AutoImageProcessor.from_pretrained(DINOV3_PATH)
    # DINOv3 preprocessing: resize 256 -> center crop 224 -> normalize
    mean = tuple(processor.image_mean)  # (0.485, 0.456, 0.406)
    std = tuple(processor.image_std)     # (0.229, 0.224, 0.225)
    size = processor.size.get("shortest_edge", 256)
    crop = processor.crop_size.get("height", INPUT_SIZE)

    if phase == "train" and TRAIN_AUG:
        return transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(crop),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=COLOR_JITTER, contrast=COLOR_JITTER,
                saturation=COLOR_JITTER, hue=min(COLOR_JITTER / 3, 0.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=RANDOM_ERASING),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def build_dataloaders():
    """Build the training and evaluation dataloaders"""
    train_transform = get_transforms("train")
    eval_transform = get_transforms("eval")

    train_ds = DunhuangDataset(TRAIN_CSV, transform=train_transform, is_train=True)
    query_ds = DunhuangDataset(QUERY_CSV, transform=eval_transform, is_train=False)
    # the gallery contains distractors
    gallery_ds = DunhuangDataset(GALLERY_CSV, transform=eval_transform,
                                 is_train=False, include_distractors=True)
    cross_ds = DunhuangDataset(CROSS_SRC_CSV, transform=eval_transform,
                               is_train=False)

    print(f"[Dataset] Train: {len(train_ds)} (labeled only)")
    print(f"[Dataset] Query: {len(query_ds)}")
    print(f"[Dataset] Gallery: {len(gallery_ds)} (labeled + distractors)")
    print(f"[Dataset] Cross-source: {len(cross_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=TRAIN_BS, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )
    query_loader = DataLoader(
        query_ds, batch_size=EVAL_BS, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    gallery_loader = DataLoader(
        gallery_ds, batch_size=EVAL_BS, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    cross_loader = DataLoader(
        cross_ds, batch_size=EVAL_BS, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    return {
        "train": train_loader,
        "query": query_loader,
        "gallery": gallery_loader,
        "cross": cross_loader,
    }


if __name__ == "__main__":
    loaders = build_dataloaders()
    batch = next(iter(loaders["train"]))
    print(f"\nSample batch:")
    print(f"  image: {batch['image'].shape} {batch['image'].dtype}")
    print(f"  label: {batch['label'].shape}, unique: {batch['label'].unique().tolist()}")
    print(f"  paths: {batch['path'][:2]}")
