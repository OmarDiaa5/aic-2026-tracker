# =============================================================================
# config.py — Central configuration for OSTrackSmall UAV Tracker
# AIC 2026 Competition Submission
# =============================================================================

# ── Model Architecture Constants ──────────────────────────────────────────────
Z_SIZE       = 128    # Template crop size (pixels)
X_SIZE       = 256    # Search crop size (pixels)
PATCH_SIZE   = 16     # ViT patch size
EMBED_DIM    = 384    # Transformer embedding dimension (ViT-Small)
NUM_HEADS    = 6      # Multi-head attention heads
DEPTH        = 12     # Transformer encoder depth

# Derived token counts (do not change unless patch size changes)
NUM_Z_TOKENS = (Z_SIZE  // PATCH_SIZE) ** 2   # 64  template tokens
NUM_X_TOKENS = (X_SIZE  // PATCH_SIZE) ** 2   # 256 search tokens
TOTAL_TOKENS = NUM_Z_TOKENS + NUM_X_TOKENS     # 320 joint tokens

# ── Tracker Inference Hyperparameters ─────────────────────────────────────────
CONFIDENCE_THRESHOLD  = 0.35   # Below → target declared absent
UPDATE_THRESHOLD      = 0.60   # Above → update running template
UPDATE_INTERVAL       = 5      # Frames between template updates
TEMPLATE_ANCHOR_W     = 0.25   # Weight of initial template in running mix
HANN_WEIGHT           = 0.30   # Score-map Hann window blend factor
SEARCH_SCALE          = 2.0    # Search region size = base_size × this
TEMPLATE_SCALE        = 1.0    # Template region size = base_size × this

# ── Training Hyperparameters ───────────────────────────────────────────────────
N_EPOCHS             = 80
BATCH_SIZE           = 32
GRAD_CLIP            = 1.0
EARLY_STOP_PATIENCE  = 20
EARLY_STOP_MIN_DELTA = 0.002
PHASE2_EPOCH         = 20       # Epoch at which curriculum phase 2 begins

PAIRS_PER_SEQ_TRAIN  = 25       # Training pairs sampled per sequence per epoch
PAIRS_PER_SEQ_VAL    = 3        # Validation pairs per sequence
INIT_FRAME_GAP       = 100      # Max template-search frame gap in phase 1
PHASE2_FRAME_GAP     = 150      # Max frame gap in phase 2
ABSENCE_PROB_P1      = 0.15     # Fraction of absent samples injected in phase 1
ABSENCE_PROB_P2      = 0.20     # Fraction of absent samples injected in phase 2
MIN_VISIBLE_FRAMES   = 2        # Sequences with fewer visible frames are skipped

# ── Loss Weights ──────────────────────────────────────────────────────────────
W_FOCAL = 1.0   # Focal loss on corner heatmap
W_L1    = 5.0   # L1 loss on predicted bbox
W_GIOU  = 2.0   # GIoU loss on predicted bbox
W_CONF  = 1.0   # Focal BCE loss on confidence head

# ── Optimizer ────────────────────────────────────────────────────────────────
LR_PRETRAINED   = 1e-5   # Patch embed + x-branch transformer weights
LR_NEW_BACKBONE = 5e-5   # z-branch transformer weights (new, no pretrain)
LR_HEADS        = 2e-4   # CornerHead + ConfidenceHead + pos_embed
WEIGHT_DECAY    = 0.05
LR_MIN          = 1e-6   # CosineAnnealingLR eta_min

# ── Checkpoint / Saving ───────────────────────────────────────────────────────
CKPT_EVERY_STEPS = 100
CKPT_PATH        = "weights/tracker_ckpt_latest.pt"
BEST_CKPT_PATH   = "weights/tracker_ckpt_best.pt"

# ── Session Time Budget (Kaggle) ──────────────────────────────────────────────
MAX_SESSION_HOURS = 11.5
