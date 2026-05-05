# Technical Report: OSTrackSmall — A Lightweight Mixed-Attention Tracker for Aerial Single-Object Tracking

**AIC 2026 Competition Submission**
**Team: A-Eye**

---

## 1. Introduction

This report presents **OSTrackSmall**, a custom single-object tracker designed for the AIC 2026 UAV (Unmanned Aerial Vehicle) tracking challenge. The tracker is purpose-built for aerial video sequences captured by drones, where targets are typically small, undergo rapid scale changes due to altitude variation, and frequently experience full or partial occlusion.

Our approach builds on the One-Stream Tracking (OSTrack) paradigm — processing template and search tokens jointly through a shared Vision Transformer (ViT) encoder — but introduces several architectural innovations tailored to the aerial domain:

1. **Mixed Attention Module (MAM):** A custom attention mechanism that prevents template contamination by restricting template tokens to self-attention while allowing search tokens full cross-attention over the joint sequence.
2. **CornerHead Localization:** A corner-based bounding box regression head that predicts top-left and bottom-right heatmaps with sub-pixel offset regression, providing superior precision for small aerial targets.
3. **ConfidenceHead Absence Detection:** A lightweight MLP head that explicitly models target absence, enabling robust handling of out-of-view and fully occluded scenarios common in aerial footage.
4. **Curriculum-Based Training:** A two-phase training strategy with progressive difficulty scaling and controlled absence injection.

The resulting model achieves real-time inference at approximately 15–22 ms per frame on an NVIDIA T4 GPU while maintaining a parameter count of 46.45M and a disk footprint of only 88.75 MB.

---

## 2. Architecture

### 2.1 Overview

OSTrackSmall follows a one-stream joint feature extraction paradigm. A template image (128×128 pixels) and a search image (256×256 pixels) are independently tokenized via a shared patch embedding layer, concatenated along the token dimension, and processed through a 12-layer Mixed Attention Encoder. The encoder output is then routed to two task-specific heads: a CornerHead for bounding box localization and a ConfidenceHead for target presence estimation.

```
Template (128×128) ──→ PatchEmbed (16×16) ──→ 64 tokens  ─┐
                                                            ├──→ [z_tokens ; x_tokens] ──→ MixedAttentionEncoder (12 layers)
Search  (256×256) ──→ PatchEmbed (16×16) ──→ 256 tokens ─┘                                        │
                                                                                                    ├──→ CornerHead ──→ bbox (x, y, w, h)
                                                                                                    ├──→ ConfidenceHead ──→ confidence score
                                                                                                    └──→ Score Map ──→ spatial response map
```

### 2.2 Patch Embedding

Both template and search images are tokenized using a single `PatchEmbed16x16` module consisting of:
- A 16×16 convolution with stride 16 projecting each patch to the 384-dimensional embedding space.
- Layer normalization applied to the output tokens.

This produces 64 template tokens (8×8 grid) and 256 search tokens (16×16 grid), for a total of 320 joint tokens per forward pass.

### 2.3 Mixed Attention Module (MAM)

The core innovation of our architecture is the **Mixed Attention Module**, which replaces the standard self-attention mechanism in each transformer layer. In a standard one-stream tracker, all tokens (template + search) attend to all other tokens equally. This creates a symmetry problem: search-region noise can contaminate the template representation over successive layers, causing drift.

Our MAM breaks this symmetry with an asymmetric attention pattern:

| Token Type | Attention Scope | Rationale |
|---|---|---|
| **Template tokens (z)** | Self-attention over template tokens only | Preserves the clean target appearance model; prevents search noise from corrupting the template |
| **Search tokens (x)** | Full cross-attention over all tokens (z + x) | Allows the search region to attend to the template for target matching while also performing spatial self-attention |

Each MAM layer contains:
- **Template branch:** Multi-head self-attention (`attn_z`), layer norm (`norm_z1`, `norm_z2`), and a feed-forward network (`ffn_z`).
- **Search branch:** Multi-head attention (`attn_x`) that receives the full concatenated [z; x] sequence as keys/values but only search tokens as queries, followed by layer norm (`norm_x1`, `norm_x2`), context normalization (`norm_ctx`), and a feed-forward network (`ffn_x`).

This asymmetric design acts as a lightweight structural prior: the template remains a stable reference signal throughout the network depth, while the search tokens progressively attend to it to localize the target.

### 2.4 CornerHead

We use a corner-based regression head rather than a center-based head. The CornerHead predicts:

1. **Top-Left (TL) Heatmap:** A spatial probability map over the 16×16 search grid indicating the likely position of the bounding box's top-left corner.
2. **Bottom-Right (BR) Heatmap:** A corresponding probability map for the bottom-right corner.
3. **TL Offset:** Sub-pixel (Δx, Δy) offset refinement for the top-left corner.
4. **BR Offset:** Sub-pixel (Δx, Δy) offset refinement for the bottom-right corner.

The CornerHead architecture consists of:
- A shared MLP trunk (384 → 384 → 384 with GELU activation and layer normalization).
- Four parallel output branches, each a single linear projection from 384 dimensions to the appropriate output size.

The final bounding box is computed by:
1. Extracting the argmax positions from the TL and BR heatmaps.
2. Adding the predicted sub-pixel offsets.
3. Converting from corner coordinates to center-width-height (cxcywh) format, normalized to [0, 1].

**Why corners over centers?** For small aerial targets (often <20×20 pixels in the original frame), predicting two corner points provides tighter bounding box estimates than predicting a single center point plus width/height. The corner formulation implicitly captures aspect ratio information and is more robust to the extreme scale variations seen in drone footage.

### 2.5 ConfidenceHead

Target absence is a critical challenge in aerial tracking — the target may fly out of the camera's field of view, become fully occluded by structures, or simply be too small to detect. We handle this with a dedicated ConfidenceHead that produces a scalar confidence score in [0, 1].

Architecture:
- **Dual-pooling:** Both mean-pooling and max-pooling are applied to the search tokens, then concatenated (producing a 768-dimensional vector).
- **MLP classifier:** 768 → 384 → 1, with GELU activation and a sigmoid output.

During inference, if the confidence score falls below a threshold (0.35), the target is declared absent and the tracker outputs the last known valid bounding box without updating the template.

### 2.6 Score Map

In addition to the CornerHead output, we compute a spatial score map by projecting each of the 256 search tokens to a scalar logit via a learned linear layer. This score map serves two purposes:

1. **Cross-validation:** During inference, the score map peak is compared against the CornerHead's predicted bounding box center. If they disagree by more than 15% of the search window, the tracker falls back to the score map position with the previous frame's bounding box size. This catches catastrophic CornerHead failures.
2. **Training signal:** A Gaussian focal loss is applied to the score map, centered on the ground-truth top-left corner, providing an auxiliary gradient signal to the encoder.

---

## 3. Training

### 3.1 Data

Training uses the E-Eye aerial tracking dataset provided by the competition organizers. Each sequence consists of a video file and a per-frame bounding box annotation file. Sequences with fewer than 2 visible frames are excluded.

### 3.2 Pair Sampling and Augmentation

For each training step, we sample a (template, search) frame pair from a random sequence:
- The **template frame** is always a frame where the target is visible.
- The **search frame** is sampled within a configurable temporal gap from the template.
- **Absence injection:** A configurable fraction of training pairs have the search-frame annotation zeroed out, simulating target absence.

The search crop undergoes extensive augmentation:
- Horizontal flip (50% probability)
- HSV color jitter (80% probability)
- Gaussian blur (25% probability, kernel size 3 or 5)
- Altitude scale simulation (40% probability, scale 0.75–1.35×)
- Directional motion blur (30% probability, simulating drone movement)
- Perspective jitter (25% probability, simulating camera angle changes)

Template crops are **not** augmented to preserve appearance fidelity.

### 3.3 Curriculum Learning

Training proceeds in two phases:

| Parameter | Phase 1 (Epochs 1–20) | Phase 2 (Epochs 21–80) |
|---|---|---|
| Max frame gap | 100 frames | 150 frames |
| Absence injection rate | 15% | 20% |

Phase 1 uses shorter temporal gaps to establish stable feature representations. Phase 2 increases the gap and absence rate to improve robustness to drift and disappearance.

### 3.4 Loss Function

The total loss is a weighted combination of four terms:

$$\mathcal{L}_{total} = w_{focal} \cdot \mathcal{L}_{focal} + w_{L1} \cdot \mathcal{L}_{L1} + w_{GIoU} \cdot \mathcal{L}_{GIoU} + w_{conf} \cdot \mathcal{L}_{conf}$$

| Loss | Weight | Description |
|---|---|---|
| **Focal Loss** | 1.0 | Applied to the score map against a Gaussian target centered on the ground-truth TL corner (σ=1.5) |
| **L1 Loss** | 5.0 | Smooth L1 regression on the predicted bounding box (cxcywh) vs. ground truth |
| **GIoU Loss** | 2.0 | Generalized Intersection-over-Union loss for scale-invariant box regression |
| **Confidence Loss** | 1.0 | Focal BCE loss on the ConfidenceHead output (γ=2.5, α=0.75) |

The L1, GIoU, and Focal losses are computed only on visible (non-absent) samples. The confidence loss is computed on all samples.

### 3.5 Optimizer

We use a three-group AdamW optimizer with differential learning rates to account for the different initialization states of model components:

| Parameter Group | Learning Rate | Rationale |
|---|---|---|
| Pretrained ViT weights (patch embed + x-branch attention) | 1×10⁻⁵ | These weights are initialized from ViT-Small pretrained on ImageNet-21K (via timm). Fine-tune gently. |
| New backbone weights (z-branch attention in MAM) | 5×10⁻⁵ | The z-branch attention layers are new (no pretrained weights exist for the asymmetric MAM). Train faster. |
| Task heads + positional embedding | 2×10⁻⁴ | CornerHead, ConfidenceHead, and the learned positional embedding are initialized randomly. Train aggressively. |

Additional training details:
- **Weight decay:** 0.05
- **Scheduler:** Cosine annealing with η_min = 1×10⁻⁶
- **Gradient clipping:** Max norm 1.0
- **Mixed precision:** FP16 via PyTorch AMP
- **Batch size:** 32
- **Early stopping:** Patience of 20 epochs (activated after epoch 25), minimum improvement δ = 0.002

### 3.6 Pretrained Weight Initialization

The patch embedding layer and the search-branch (x-branch) attention layers in each MAM layer are initialized from a ViT-Small model pretrained on ImageNet-21K, loaded via the `timm` library (`vit_small_patch16_224`). Positional embeddings are interpolated from the pretrained 14×14 grid to our 8×8 (template) + 16×16 (search) token layout using bicubic interpolation.

The z-branch attention layers, CornerHead, ConfidenceHead, and score map projection are initialized with PyTorch defaults (Xavier/Kaiming).

---

## 4. Inference

### 4.1 Tracker Initialization

On the first frame:
1. The ground-truth bounding box is used to extract a template crop (128×128).
2. The template is stored as the **anchor template** (never modified) and as the **running template** (updated online).

### 4.2 Per-Frame Tracking

For each subsequent frame:

1. **Multi-Scale Search:** Three search crops are extracted at scales [0.95, 1.0, 1.10] relative to the base crop size. The scale with the highest confidence score is selected.
2. **Score Map with Hann Window:** A cosine (Hann) window penalty is applied to the score map to suppress responses far from the previous target position, with a blend weight of 0.30.
3. **Bounding Box Decoding:** The CornerHead produces a bounding box prediction. The score map peak is also computed.
4. **Cross-Validation:** If the CornerHead center and score map peak disagree by more than 15% of the search window, the tracker falls back to the score map position with the previous bounding box dimensions.
5. **Confidence Gating:**
   - If confidence ≥ 0.60: Update the running template via exponential moving average (EMA).
   - If confidence < 0.35: Declare the target absent. Output the last valid bounding box.
   - If 0.35 ≤ confidence < 0.60: Accept the prediction but do not update the template.

### 4.3 Template Update Strategy

The running template is updated every 5 frames (when confidence exceeds the update threshold) using an EMA:

```
running_template = anchor_weight × anchor_template + (1 - anchor_weight) × new_template
```

where `anchor_weight = 0.25`. This ensures that 25% of the template always comes from the pristine first-frame appearance, preventing catastrophic drift.

### 4.4 Robustness Guarantees

The inference pipeline includes several safety mechanisms:
- **Last-valid-box fallback:** If any frame fails (exception, corrupt video frame), the last successfully tracked bounding box is used.
- **Minimum dimension clamping:** Predicted width and height are clamped to a minimum of 1 pixel.
- **Full CSV coverage:** After tracking all sequences, the output is cross-checked against the sample submission to guarantee 100% row coverage.

---

## 5. Competition Constraints Compliance

| Metric | Maximum Allowed | Our Model | Margin |
|---|---|---|---|
| Number of Parameters | 50 Million | 46.45 Million | 7.1% under |
| Model Size (Disk) | 500 MB | 88.75 MB | 82.3% under |
| Inference Latency | 30 ms | ~15–22 ms (T4 GPU) | 27–50% under |
| Computational Complexity | 30 GFLOPs | ~12.5 GFLOPs | 58.3% under |

---

## 6. Reproducibility

### 6.1 Environment

- **Framework:** PyTorch 2.2.0
- **CUDA:** 12.1
- **Key dependencies:** timm 0.9.12, opencv-python-headless 4.9.0.80, numpy 1.26.4, pandas 2.2.1

### 6.2 Repository Structure

```
aic-2026-tracker/
├── README.md              # Project overview and usage instructions
├── Dockerfile             # Builds inference container with automatic model download
├── requirements.txt       # Pinned Python dependencies
├── config.py              # All constants and hyperparameters
├── model.py               # Full architecture (OSTrackSmall, MAM, CornerHead, ConfidenceHead)
├── data_pipeline.py       # Data loading, augmentation, dataset construction
├── loss.py                # Focal, L1, GIoU, and confidence loss functions
├── train.py               # Complete training loop with curriculum learning
├── inference.py           # Generates submission CSV via CLI
└── docs/
    └── technical_report.md
```

### 6.3 Docker Inference

```bash
docker build -t aic-tracker .
docker run --gpus all -v /path/to/dataset:/app/data -v /path/to/output:/app/output aic-tracker
```

The Dockerfile automatically downloads the trained model checkpoint from Kaggle Models using `kagglehub`.

### 6.4 Training Reproduction

```bash
pip install -r requirements.txt
python train.py --data_dir /path/to/dataset
```

Training completes within the 12-hour Kaggle notebook runtime budget.

---

## 7. Key Design Decisions Summary

1. **One-stream joint encoding** over Siamese networks: Eliminates the need for a separate correlation step and allows early template-search interaction.
2. **Asymmetric MAM** over symmetric self-attention: Prevents template contamination from noisy search regions across 12 transformer layers.
3. **CornerHead** over CenterHead: Provides tighter bounding boxes for small aerial targets by directly predicting corner positions with sub-pixel refinement.
4. **Explicit absence modeling** via ConfidenceHead: Critical for aerial tracking where targets frequently leave the field of view.
5. **Multi-scale search with Hann penalty**: Handles the rapid scale changes caused by drone altitude variation while suppressing false positives far from the expected target location.
6. **EMA template update with anchor weight**: Balances adaptation to appearance changes with resistance to drift by always mixing in the pristine first-frame template.

---

## 8. Model Checkpoint

The trained model checkpoint is publicly available on Kaggle Models:
**https://www.kaggle.com/models/omardiaa05/a-eye-model/**

- Format: PyTorch state dictionary (`.pt`)
- Size: 88.75 MB
- Training epoch: 45
- Best validation IoU: 0.4697
