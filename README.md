# AIC 2026: UAV Single-Object Tracker

Custom **OSTrackSmall** — Mixed Attention Module (MAM) backbone +
CornerHead localization + ConfidenceHead absence detection.
Fine-tuned from ViT-Small ImageNet-21K weights (timm).

## Model Checkpoint

**[INSERT GOOGLE DRIVE / KAGGLE DATASET DIRECT LINK HERE]**

Place the downloaded file at `weights/best_model.pt`.

## Architecture Summary

| Component | Detail |
|---|---|
| Backbone | ViT-Small (384-dim, 12 layers, 6 heads) |
| Patch Size | 16×16 |
| Template | 128×128 → 64 tokens |
| Search | 256×256 → 256 tokens |
| Encoder | Mixed Attention Module (MAM) — template self-attention, search full cross-attention |
| Localization | CornerHead (TL + BR heatmaps with sub-pixel offset regression) |
| Absence | ConfidenceHead (mean+max pooling MLP) |
| Params | ~46.5M |
| Checkpoint | ~89 MB |

## Inference via Docker

```bash
# Build the inference Docker image
docker build --build-arg MODEL_URL="YOUR_DIRECT_LINK" -t aic-tracker .

# Run inference (mount dataset and output directory)
docker run --gpus all \
  -v /path/to/dataset:/app/data \
  -v /path/to/output:/app/output \
  aic-tracker
```

Output: `/path/to/output/submission.csv`

## Inference without Docker

```bash
pip install -r requirements.txt

python inference.py \
    --data_dir /path/to/dataset \
    --checkpoint weights/best_model.pt \
    --output submission.csv
```

## Training (optional reproduction)

```bash
pip install -r requirements.txt

python train.py --data_dir /path/to/dataset
```

Training uses a curriculum learning strategy:
- **Phase 1** (epochs 1-20): max frame gap 100, 15% absence injection
- **Phase 2** (epochs 21-80): max frame gap 150, 20% absence injection

Three-group optimizer with differential learning rates:
- Pretrained ViT weights: 1e-5
- New z-branch backbone: 5e-5
- Heads + positional embedding: 2e-4

## Repository Structure

```
aic-2026-tracker/
├── README.md              # This file
├── Dockerfile             # Inference environment builder
├── requirements.txt       # Python dependencies
├── config.py              # All constants and hyperparameters
├── model.py               # OSTrackSmall architecture + AerialTracker
├── data_pipeline.py       # Annotation parsing, video reading, augmentation, dataset
├── loss.py                # Focal, L1, GIoU, confidence losses
├── train.py               # Full training loop with curriculum learning
├── inference.py           # Generates submission.csv from test sequences
├── weights/               # Model checkpoints (not tracked in git)
└── docs/
    └── technical_report.pdf
```

## Key Design Decisions

1. **No external architecture dependencies** — entire model is self-contained
2. **CornerHead** over CenterHead — predicts TL/BR corners with sub-pixel offset for precise small-object localization
3. **MAM encoding** — template tokens use self-attention only (prevents search contamination), search tokens attend to full joint sequence
4. **Multi-scale inference** — 3 scales (0.95, 1.0, 1.10) with Hann-penalized score map selection
5. **Score-map cross-check** — if CornerHead bbox disagrees with score-map peak (>15% of search window), falls back to score-map position with prior size
6. **EMA template update** — exponential moving average with anchor weight to prevent template drift
