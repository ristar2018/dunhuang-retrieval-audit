# -*- coding: utf-8 -*-
"""
Dunhuang mural similarity retrieval - global configuration
All paths are relative to the retrieval data root
"""
import os

# ============ paths ============
PROJECT_ROOT = r"<DATA_ROOT>\retrieval"
DATA_ROOT    = r"<DATA_ROOT>\data"
MODELS_ROOT  = r"<DATA_ROOT>\models\pretrained"

# existing metadata
CLEAN_INDEX  = os.path.join(DATA_ROOT, "metadata", "clean_index.json")
HF_CSV       = os.path.join(DATA_ROOT, "raw", "dunhuang_hf", "metadata.csv")
HF_IMG_DIR   = os.path.join(DATA_ROOT, "raw", "dunhuang_hf", "images")

# project sub-directory
ANNO_DIR   = os.path.join(PROJECT_ROOT, "annotations")
FEAT_DIR   = os.path.join(PROJECT_ROOT, "features")
SPLIT_DIR  = os.path.join(PROJECT_ROOT, "splits")
EXP_DIR    = os.path.join(PROJECT_ROOT, "experiments")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

# output file
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

# ============ model configuration ============
DINOV2_PATH    = os.path.join(MODELS_ROOT, "dinov2-base")
DINOV3_PATH    = os.path.join(MODELS_ROOT, "dinov3-vitb16")
BACKBONE_NAME  = "dinov3-vitb16-pretrain-lvd1689m"
FEATURE_DIM    = 768          # output dimension of DINOv3 ViT-B/16
INPUT_SIZE     = 224          # input image size
BATCH_SIZE     = 64           # feature-extraction batch size (16 GB of VRAM is ample)
NUM_WORKERS    = 4            # DataLoader workers

# ============ label-propagation configuration ============
KNN_K          = 7            # number of k-NN neighbours
CONF_THRESHOLD = 0.60         # confidence threshold: >= 0.6 counts as reliable, < 0.6 is marked uncertain
N_CLUSTERS     = 16           # number of K-means clusters (equal to the number of seed classes; used to expose sub-classes)

# ============ split configuration ============
TRAIN_RATIO    = 0.70
QUERY_RATIO    = 0.15
GALLERY_RATIO  = 0.15
RANDOM_SEED    = 42

# cross-source generalisation test: hold out the following sources as the test set
CROSS_SOURCE_HOLDOUT = ["deepflying", "dunhuang_grottoes"]

# ============ 16-class semantic label map ============
CATEGORY_NAMES = [
    "animals",           # animal
    "architecture",      # architecture
    "buddha",            # Buddha statue
    "buddha_statue",     # Buddha sculpture
    "cave_exterior",     # cave interior
    "ceiling_pattern",   # caisson ceiling pattern
    "dance",             # dance
    "deity",             # bodhisattva / deity
    "fly_apsara",        # flying apsara
    "landscape",         # landscape
    "music",             # music
    "narrative",         # sutra-illustration narrative
    "other",             # other
    "pattern",           # pattern
    "story",             # jataka narrative
    "wall_mural",        # mural (generic)
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

# ============ utilities ============
def ensure_dirs():
    """Create all project directories"""
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


# ============ training configuration (DINOv3 + LoRA + metric learning) ============
# experiment directory
EXP_NAME       = "dinov3_lora_subarc"   # experiment name
CKPT_DIR       = os.path.join(EXP_DIR, EXP_NAME, "checkpoints")
LOG_DIR        = os.path.join(EXP_DIR, EXP_NAME, "logs")

# LoRA
LORA_R         = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj"]  # DINOv3 ViT attention (separate QKV)

# training hyperparameters
TRAIN_BS       = 32                      # training batch size
EVAL_BS        = 64                      # evaluation batch size
EPOCHS         = 40
LR_BACKBONE    = 1e-4                    # learning rate for the LoRA parameters
LR_HEAD        = 5e-4                    # learning rate for the classification head
WEIGHT_DECAY   = 0.01
WARMUP_EPOCHS  = 3
SCHEDULER      = "cosine"                # cosine annealing

# loss function
LOSS_NAME      = "subcenter_arcface"     # subcenter_arcface | arcface | cosface
ARC_SCALE      = 30.0                    # ArcFace scale (s)
ARC_MARGIN     = 0.5                     # ArcFace margin (m)
SUBCENTER_K    = 3                       # SubCenterArcFace K=3

# miner (optional)
USE_MINER      = True
MINER_NAME     = "multi_similarity"      # multi_similarity | batch_hard
MINER_EPSILON  = 0.1

# data augmentation
TRAIN_AUG      = True
COLOR_JITTER   = 0.3
RANDOM_ERASING = 0.1

# evaluation
EVAL_EVERY     = 5                       # evaluate every N epochs
SAVE_EVERY     = 10                      # save a checkpoint every N epochs
TOP_K          = [1, 5, 10]              # Recall@K
