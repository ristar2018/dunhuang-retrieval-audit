# -*- coding: utf-8 -*-
"""
敦煌壁画相似度检索项目 - 全局配置
所有路径基于 D:\\XiaBao\\AIArtDetect\\retrieval\\
"""
import os

# ============ 路径配置 ============
PROJECT_ROOT = r"<DATA_ROOT>\retrieval"
DATA_ROOT    = r"<DATA_ROOT>\data"
MODELS_ROOT  = r"<DATA_ROOT>\models\pretrained"

# 已有元数据
CLEAN_INDEX  = os.path.join(DATA_ROOT, "metadata", "clean_index.json")
HF_CSV       = os.path.join(DATA_ROOT, "raw", "dunhuang_hf", "metadata.csv")
HF_IMG_DIR   = os.path.join(DATA_ROOT, "raw", "dunhuang_hf", "images")

# 项目子目录
ANNO_DIR   = os.path.join(PROJECT_ROOT, "annotations")
FEAT_DIR   = os.path.join(PROJECT_ROOT, "features")
SPLIT_DIR  = os.path.join(PROJECT_ROOT, "splits")
EXP_DIR    = os.path.join(PROJECT_ROOT, "experiments")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

# 输出文件
SEED_LABELS    = os.path.join(ANNO_DIR, "seed_labels.csv")
ALL_LABELS     = os.path.join(ANNO_DIR, "all_labels.csv")
LABEL_STATS    = os.path.join(ANNO_DIR, "label_statistics.json")

FEATURE_NPY    = os.path.join(FEAT_DIR, "dinov3_features.npy")
FEATURE_INDEX  = os.path.join(FEAT_DIR, "feature_index.json")
FEATURE_NPY_V2 = os.path.join(FEAT_DIR, "dinov2_features.npy")  # DINOv2 backup

TRAIN_CSV      = os.path.join(SPLIT_DIR, "train.csv")
QUERY_CSV      = os.path.join(SPLIT_DIR, "query.csv")
GALLERY_CSV    = os.path.join(SPLIT_DIR, "gallery.csv")
CROSS_SRC_CSV  = os.path.join(SPLIT_DIR, "cross_source_test.csv")

# ============ 模型配置 ============
DINOV2_PATH    = os.path.join(MODELS_ROOT, "dinov2-base")
DINOV3_PATH    = os.path.join(MODELS_ROOT, "dinov3-vitb16")
BACKBONE_NAME  = "dinov3-vitb16-pretrain-lvd1689m"
FEATURE_DIM    = 768          # DINOv3 ViT-B/16 输出维度
INPUT_SIZE     = 224          # 输入图像尺寸
BATCH_SIZE     = 64           # 特征提取 batch size（16GB VRAM充裕）
NUM_WORKERS    = 4            # DataLoader workers

# ============ 标签传播配置 ============
KNN_K          = 7            # k-NN 近邻数
CONF_THRESHOLD = 0.60         # 置信度阈值：>=0.6 认为可靠，<0.6 标记 uncertain
N_CLUSTERS     = 16           # K-means 聚类数（与种子类别数一致，用于发现子类）

# ============ 数据集划分配置 ============
TRAIN_RATIO    = 0.70
QUERY_RATIO    = 0.15
GALLERY_RATIO  = 0.15
RANDOM_SEED    = 42

# 跨源泛化测试：留出以下数据源作为测试集
CROSS_SOURCE_HOLDOUT = ["deepflying", "dunhuang_grottoes"]

# ============ 16类语义标签映射 ============
CATEGORY_NAMES = [
    "animals",           # 动物
    "architecture",      # 建筑
    "buddha",            # 佛像
    "buddha_statue",     # 佛塑像
    "cave_exterior",     # 洞窟外观
    "ceiling_pattern",   # 藻井图案
    "dance",             # 舞蹈
    "deity",             # 菩萨/神祇
    "fly_apsara",        # 飞天
    "landscape",         # 山水
    "music",             # 音乐
    "narrative",         # 经变叙事
    "other",             # 其他
    "pattern",           # 纹样
    "story",             # 本生故事
    "wall_mural",        # 壁画(通用)
]

CATEGORY_NAMES_CN = {
    "animals": "动物",
    "architecture": "建筑",
    "buddha": "佛像",
    "buddha_statue": "佛塑像",
    "cave_exterior": "洞窟外观",
    "ceiling_pattern": "藻井图案",
    "dance": "舞蹈",
    "deity": "菩萨神祇",
    "fly_apsara": "飞天",
    "landscape": "山水",
    "music": "音乐",
    "narrative": "经变叙事",
    "other": "其他",
    "pattern": "纹样",
    "story": "本生故事",
    "wall_mural": "壁画通用",
}

# ============ 工具函数 ============
def ensure_dirs():
    """创建所有项目目录"""
    for d in [ANNO_DIR, FEAT_DIR, SPLIT_DIR, EXP_DIR, RESULT_DIR]:
        os.makedirs(d, exist_ok=True)

if __name__ == "__main__":
    ensure_dirs()
    print("项目目录结构已创建:")
    for d in [PROJECT_ROOT, ANNO_DIR, FEAT_DIR, SPLIT_DIR, EXP_DIR, RESULT_DIR]:
        print(f"  {d}")
    print(f"\n特征维度: {FEATURE_DIM}")
    print(f"种子类别数: {len(CATEGORY_NAMES)}")
    print(f"KNN k={KNN_K}, 置信度阈值={CONF_THRESHOLD}")


# ============ 训练配置 (DINOv3 + LoRA + Metric Learning) ============
# 实验目录
EXP_NAME       = "dinov3_lora_subarc"   # 实验名称
CKPT_DIR       = os.path.join(EXP_DIR, EXP_NAME, "checkpoints")
LOG_DIR        = os.path.join(EXP_DIR, EXP_NAME, "logs")

# LoRA
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj"]  # DINOv3 ViT attention (分离式QKV)

# 训练超参
TRAIN_BS       = 32                      # 训练 batch size
EVAL_BS        = 64                      # 评估 batch size
EPOCHS         = 40
LR_BACKBONE    = 1e-4                    # LoRA参数学习率
LR_HEAD        = 5e-4                    # 分类头学习率
WEIGHT_DECAY   = 0.01
WARMUP_EPOCHS  = 3
SCHEDULER      = "cosine"                # cosine annealing

# 损失函数
LOSS_NAME      = "subcenter_arcface"     # subcenter_arcface | arcface | cosface
ARC_SCALE      = 30.0                    # ArcFace scale (s)
ARC_MARGIN     = 0.5                     # ArcFace margin (m)
SUBCENTER_K    = 3                       # SubCenterArcFace K=3

# Miner (可选)
USE_MINER      = True
MINER_NAME     = "multi_similarity"      # multi_similarity | batch_hard
MINER_EPSILON  = 0.1

# 数据增强
TRAIN_AUG      = True
COLOR_JITTER   = 0.3
RANDOM_ERASING = 0.1

# 评估
EVAL_EVERY     = 5                       # 每N个epoch评估一次
SAVE_EVERY     = 10                      # 每N个epoch保存checkpoint
TOP_K          = [1, 5, 10]              # Recall@K
