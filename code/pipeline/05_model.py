# -*- coding: utf-8 -*-
"""
Step 5: model definition
DINOv3 ViT-B/16 backbone + LoRA adapters + Metric Learning Head.
- backbone: local DINOv3, base weights frozen, only the LoRA parameters trained
- head: SubCenterArcFace / ArcFace / CosFace
- output: 768-dimensional L2-normalised embeddings
"""
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DINOV3_PATH, FEATURE_DIM,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS,
    ARC_SCALE, ARC_MARGIN, SUBCENTER_K, LOSS_NAME, CATEGORY_NAMES
)

NUM_CLASSES = len(CATEGORY_NAMES)  # 16


def build_backbone(use_lora=True):
    """Build the DINOv3 backbone, optionally with LoRA"""
    model = AutoModel.from_pretrained(DINOV3_PATH, torch_dtype=torch.float32)
    # DINOv3 ViT returns last_hidden_state; the CLS token is at position 0
    if use_lora:
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        for p in model.parameters():
            p.requires_grad = False
    return model


class ArcFaceHead(nn.Module):
    """ArcFace: Additive Angular Margin Loss (CVPR 2019)"""
    def __init__(self, in_dim, num_classes, scale=ARC_SCALE, margin=ARC_MARGIN):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))
        self.th = torch.cos(torch.tensor(3.14159265 - margin))
        self.mm = torch.sin(torch.tensor(3.14159265 - margin)) * margin

    def forward(self, embeddings, labels=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None or not self.training:
            return cosine * self.scale
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, self.weight.size(0)).float()
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale


class SubCenterArcFaceHead(nn.Module):
    """SubCenter ArcFace (ECCV 2020), K sub-centers per class"""
    def __init__(self, in_dim, num_classes, k=SUBCENTER_K,
                 scale=ARC_SCALE, margin=ARC_MARGIN):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes * k, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.k = k
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.cos_m = torch.cos(torch.tensor(margin))
        self.sin_m = torch.sin(torch.tensor(margin))
        self.th = torch.cos(torch.tensor(3.14159265 - margin))
        self.mm = torch.sin(torch.tensor(3.14159265 - margin)) * margin

    def forward(self, embeddings, labels=None):
        cosine_all = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        # reshape to (batch, num_classes, k), take max over k
        cosine_all = cosine_all.view(-1, self.num_classes, self.k)
        cosine, _ = cosine_all.max(dim=2)

        if labels is None or not self.training:
            return cosine * self.scale
        sine = torch.sqrt(1.0 - cosine.pow(2).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = F.one_hot(labels, self.num_classes).float()
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output * self.scale


class CosFaceHead(nn.Module):
    """CosFace: Large Margin Cosine Loss (CVPR 2018)"""
    def __init__(self, in_dim, num_classes, scale=ARC_SCALE, margin=0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_dim))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings, labels=None):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None or not self.training:
            return cosine * self.scale
        one_hot = F.one_hot(labels, self.weight.size(0)).float()
        output = cosine - one_hot * self.margin
        return output * self.scale


class DINOv3MetricModel(nn.Module):
    """Full model: DINOv3 backbone + embedding projection + metric head"""

    def __init__(self, use_lora=True, loss_name=LOSS_NAME):
        super().__init__()
        self.backbone = build_backbone(use_lora=use_lora)
        self.embed_dim = FEATURE_DIM

        # Build head
        if loss_name == "subcenter_arcface":
            self.head = SubCenterArcFaceHead(
                self.embed_dim, NUM_CLASSES, k=SUBCENTER_K)
        elif loss_name == "arcface":
            self.head = ArcFaceHead(self.embed_dim, NUM_CLASSES)
        elif loss_name == "cosface":
            self.head = CosFaceHead(self.embed_dim, NUM_CLASSES)
        else:
            raise ValueError(f"Unknown loss: {loss_name}")

        self.loss_name = loss_name

    def forward(self, pixel_values, labels=None):
        outputs = self.backbone(pixel_values=pixel_values)
        # CLS token at position 0 (DINOv3: 0=CLS, 1-4=registers, 5+=patches)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        logits = self.head(embeddings, labels)
        return embeddings, logits

    def get_embeddings(self, pixel_values):
        """At inference time return the embedding only, bypassing the head"""
        outputs = self.backbone(pixel_values=pixel_values)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return F.normalize(embeddings, p=2, dim=1)

    def save_lora(self, path):
        """Save the LoRA weights"""
        self.backbone.save_pretrained(path)
        head_path = os.path.join(path, "metric_head.pt")
        torch.save(self.head.state_dict(), head_path)
        print(f"[SAVE] LoRA + head -> {path}")

    def load_lora(self, path):
        """Load the LoRA weights"""
        from peft import PeftModel
        self.backbone = PeftModel.from_pretrained(self.backbone, path)
        head_path = os.path.join(path, "metric_head.pt")
        if os.path.exists(head_path):
            self.head.load_state_dict(torch.load(head_path, map_location="cpu"))
        print(f"[LOAD] LoRA + head <- {path}")


if __name__ == "__main__":
    # Quick test
    model = DINOv3MetricModel(use_lora=True, loss_name="subcenter_arcface")
    dummy = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])
    emb, logits = model(dummy, labels)
    print(f"Embedding: {emb.shape}, norm: {emb.norm(dim=1)}")
    print(f"Logits: {logits.shape}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total/1e6:.1f}M, Trainable: {trainable/1e6:.2f}M")
