# =============================================================================
# inference.py — AIC 2026 UAV Tracker Inference Pipeline
#
# Generates submission.csv from competition test sequences.
# Designed to run inside Docker with CLI arguments.
#
# Usage:
#   python inference.py \
#       --data_dir /app/data \
#       --checkpoint /app/weights/best_model.pt \
#       --output /app/output/submission.csv
# =============================================================================

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from pathlib import Path

from config import CONFIDENCE_THRESHOLD, UPDATE_THRESHOLD, UPDATE_INTERVAL
from model import OSTrackSmall, AerialTracker
from data_pipeline import (
    parse_annotation_file,
    is_box_visible,
    VideoReader,
    image_to_tensor,
    _find_video_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description="AIC 2026 UAV Tracker Inference")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory of the competition dataset")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Output path for submission CSV")
    parser.add_argument("--split", type=str, default="public_lb",
                        help="Manifest split to run inference on")
    parser.add_argument("--confidence_threshold", type=float,
                        default=CONFIDENCE_THRESHOLD,
                        help="Confidence threshold for absence detection")
    parser.add_argument("--update_threshold", type=float,
                        default=UPDATE_THRESHOLD,
                        help="Confidence threshold for template update")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device,
               confidence_threshold: float = 0.35) -> OSTrackSmall:
    """Load OSTrackSmall from a competition checkpoint."""
    model = OSTrackSmall(confidence_threshold=confidence_threshold).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("net", ckpt.get("model", ckpt))
    else:
        state_dict = ckpt

    # Strip DataParallel prefix if present
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    model.load_state_dict(state_dict)
    model.eval()

    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    best_iou = ckpt.get("best_val_iou", "?") if isinstance(ckpt, dict) else "?"
    print(f"Model loaded from epoch {epoch} | best_val_iou={best_iou}")
    return model


def generate_submission(args):
    """Run tracking on all sequences in the specified split and write CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device, args.confidence_threshold)

    tracker = AerialTracker(
        model, device,
        search_size=256, template_size=128,
        update_interval=UPDATE_INTERVAL,
        update_threshold=args.update_threshold,
        confidence_threshold=args.confidence_threshold,
    )

    # ── Load manifest ─────────────────────────────────────────────────────────
    root = Path(args.data_dir)
    manifest_path = root / "metadata" / "contestant_manifest.json"
    if not manifest_path.exists():
        # Try alternate location
        manifest_path = root / "contestant_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    split_seqs = manifest.get(args.split, {})
    if not split_seqs:
        print(f"ERROR: Split '{args.split}' not found. Available: {list(manifest.keys())}")
        sys.exit(1)

    print(f"Running inference on {len(split_seqs)} sequences (split={args.split})")

    # ── Track all sequences ───────────────────────────────────────────────────
    pred_dict = {}     # "seq_id_frameIdx" → (x, y, w, h)
    init_boxes = {}    # seq_id → initialization box
    t_start = time.time()

    for seq_idx, (seq_id, meta) in enumerate(split_seqs.items()):
        n_frames = meta["n_frames"]
        ann_path = root / meta["annotation_path"]
        video_path = root / meta["video_path"]
        actual_vid = _find_video_path(video_path.parent, video_path)

        if actual_vid is None:
            print(f"  [SKIP] {seq_id} — video not found")
            continue

        boxes = parse_annotation_file(ann_path)
        if not boxes or not is_box_visible(boxes[0]):
            print(f"  [SKIP] {seq_id} — invalid frame-0 annotation")
            continue

        # Frame 0 — initialize tracker
        x0, y0, w0, h0 = [float(v) for v in boxes[0]]
        init_box = (int(x0), int(y0), max(1, int(w0)), max(1, int(h0)))
        init_boxes[seq_id] = init_box

        reader = VideoReader(actual_vid, cache_size=5)
        frame0 = image_to_tensor(reader.get_frame_rgb(0))
        tracker.initialise(frame0, (x0, y0, w0, h0))
        pred_dict[f"{seq_id}_0"] = init_box

        last_valid_box = init_box
        fallback_count = 0

        # Frames 1…N
        for fi in range(1, n_frames):
            try:
                frame_t = image_to_tensor(reader.get_frame_rgb(fi))
                result = tracker.track(frame_t)
                x, y, w, h = result["bbox_xywh"]
                w = max(1, w)
                h = max(1, h)
                box = (max(0, x), max(0, y), w, h)
                pred_dict[f"{seq_id}_{fi}"] = box
                last_valid_box = box
            except Exception as e:
                pred_dict[f"{seq_id}_{fi}"] = last_valid_box
                fallback_count += 1
                if fallback_count <= 3:
                    print(f"    [FALLBACK] {seq_id} frame {fi}: {type(e).__name__}: {e}")

        reader.release()
        absent_pct = 100 * tracker.absent_count / max(1, n_frames - 1)
        flag = f"  ⚠ {fallback_count} fallbacks" if fallback_count > 0 else ""
        print(f"  [{seq_idx+1}/{len(split_seqs)}] {seq_id:<35s} "
              f"{n_frames:5d} frames  absent={absent_pct:.1f}%{flag}")

    elapsed = time.time() - t_start
    print(f"\nTracking complete in {elapsed:.1f}s")

    # ── Coverage check against sample_submission ──────────────────────────────
    sample_sub_path = root / "metadata" / "sample_submission.csv"
    if not sample_sub_path.exists():
        sample_sub_path = root / "sample_submission.csv"

    if sample_sub_path.exists():
        sub = pd.read_csv(sample_sub_path)
        missing_count = 0
        for row_id in sub["id"]:
            if row_id not in pred_dict:
                missing_count += 1
                seq_id = "_".join(row_id.rsplit("_", 1)[:-1])
                if seq_id in init_boxes:
                    pred_dict[row_id] = init_boxes[seq_id]
                else:
                    pred_dict[row_id] = (0, 0, 1, 1)
        if missing_count > 0:
            print(f"⚠ Filled {missing_count} missing predictions from init boxes")
        else:
            print(f"✓ 100% coverage — all {len(sub)} predictions present")

        # Build final CSV
        sub["x"] = sub["id"].map(lambda i: pred_dict.get(i, (0, 0, 1, 1))[0])
        sub["y"] = sub["id"].map(lambda i: pred_dict.get(i, (0, 0, 1, 1))[1])
        sub["w"] = sub["id"].map(lambda i: pred_dict.get(i, (0, 0, 1, 1))[2])
        sub["h"] = sub["id"].map(lambda i: pred_dict.get(i, (0, 0, 1, 1))[3])
    else:
        print("sample_submission.csv not found — building CSV from predictions only")
        rows = []
        for key, (x, y, w, h) in sorted(pred_dict.items()):
            rows.append({"id": key, "x": x, "y": y, "w": w, "h": h})
        sub = pd.DataFrame(rows)

    # ── Write output ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    sub.to_csv(args.output, index=False)

    n_total = len(sub)
    n_zero = ((sub["w"] == 0) | (sub["h"] == 0)).sum()
    print(f"\nsubmission.csv saved — {n_total:,} rows")
    print(f"  non-zero w: {(sub['w'] > 0).sum():,} ({100 * (sub['w'] > 0).sum() / n_total:.1f}%)")
    print(f"  zero w or h: {n_zero:,} — should be 0%")
    print(f"Output: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    args = parse_args()
    generate_submission(args)
