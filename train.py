# =============================================================================
# train.py — Training script for OSTrackSmall UAV Tracker
# =============================================================================

import argparse
import os
import time
import warnings

import numpy as np
import torch

from config import (
    N_EPOCHS, BATCH_SIZE, GRAD_CLIP, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA,
    PHASE2_EPOCH, CKPT_EVERY_STEPS, CKPT_PATH, BEST_CKPT_PATH,
    PAIRS_PER_SEQ_TRAIN, PAIRS_PER_SEQ_VAL,
    INIT_FRAME_GAP, PHASE2_FRAME_GAP,
    ABSENCE_PROB_P1, ABSENCE_PROB_P2,
    W_FOCAL, W_L1, W_GIOU, W_CONF,
    LR_PRETRAINED, LR_NEW_BACKBONE, LR_HEADS, WEIGHT_DECAY, LR_MIN,
    MAX_SESSION_HOURS, MIN_VISIBLE_FRAMES,
)
from model import OSTrackSmall, AerialTracker, load_pretrained_weights
from data_pipeline import (
    build_dataset, tracking_collate_fn, image_to_tensor, FastFrameReader,
)
from loss import compute_loss

warnings.filterwarnings('ignore', message='enable_nested_tensor')


def parse_args():
    p = argparse.ArgumentParser(description="Train OSTrackSmall")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--max_hours", type=float, default=MAX_SESSION_HOURS)
    return p.parse_args()


def quick_validate(model, sequences, device):
    model.eval()
    tracker_val = AerialTracker(model, device, search_size=256, template_size=128,
                                update_interval=5, update_threshold=0.55,
                                confidence_threshold=0.35)
    all_ious = []
    for seq in sequences:
        reader = FastFrameReader(seq.seq_id, seq.video_path, seq.n_frames)
        if not seq.visible_idxs:
            reader.release()
            continue
        init_idx = 0 if seq.visible[0] else seq.visible_idxs[0]
        frame0 = image_to_tensor(reader.get_frame_rgb(init_idx))
        bx, by, bw, bh = seq.boxes[init_idx].tolist()
        tracker_val.initialise(frame0, (bx, by, bw, bh))
        for fi in range(init_idx + 1, seq.n_frames):
            frame_t = image_to_tensor(reader.get_frame_rgb(fi))
            result = tracker_val.track(frame_t)
            px, py, pw, ph = result['bbox_xywh']
            gx, gy, gw, gh = seq.boxes[fi].tolist()
            if seq.visible[fi]:
                ix1 = max(px, gx); iy1 = max(py, gy)
                ix2 = min(px + pw, gx + gw); iy2 = min(py + ph, gy + gh)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = pw * ph + gw * gh - inter
                iou = inter / (union + 1e-7) if union > 0 else 0.0
                all_ious.append(iou)
        reader.release()
    model.train()
    return float(np.mean(all_ious)) if all_ious else 0.0


def main():
    args = parse_args()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    session_start = time.time()

    root = args.data_dir
    manifest_path = os.path.join(root, "metadata", "contestant_manifest.json")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = build_dataset(root=root, manifest_path=manifest_path, split="train",
                                  pairs_per_seq=PAIRS_PER_SEQ_TRAIN,
                                  max_frame_gap=INIT_FRAME_GAP)
    train_dataset.use_augmentation = True
    train_dataset.absence_prob = ABSENCE_PROB_P1

    val_dataset = build_dataset(root=root, manifest_path=manifest_path, split="val",
                                pairs_per_seq=PAIRS_PER_SEQ_VAL, max_frame_gap=10)
    val_sequences = val_dataset.sequences

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=tracking_collate_fn,
        drop_last=True, prefetch_factor=6, persistent_workers=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = OSTrackSmall(confidence_threshold=0.35).to(device)
    load_pretrained_weights(model)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    # ── Optimizer ─────────────────────────────────────────────────────────────
    pretrained_params = (
        list(raw_model.patch_embed.parameters()) +
        [p for name, p in raw_model.transformer.named_parameters()
         if any(k in name for k in ['attn_x', 'ffn_x', 'norm_x1', 'norm_x2'])]
    )
    new_backbone_params = [
        p for name, p in raw_model.transformer.named_parameters()
        if any(k in name for k in ['attn_z', 'ffn_z', 'norm_z', 'norm_ctx'])
    ]
    head_params = (
        list(raw_model.confidence_head.parameters()) +
        list(raw_model.bbox_head.parameters()) +
        [raw_model.pos_embed]
    )

    optimizer = torch.optim.AdamW([
        {'params': pretrained_params,   'lr': LR_PRETRAINED},
        {'params': new_backbone_params, 'lr': LR_NEW_BACKBONE},
        {'params': head_params,         'lr': LR_HEADS},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=LR_MIN)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_val_iou = 0.0
    no_improve = 0
    global_step = 0
    history = []

    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt.get('global_step', 0)
        best_val_iou = ckpt.get('best_val_iou', 0.0)
        no_improve = ckpt.get('no_improve', 0)
        history = ckpt.get('history', [])
        print(f"Resumed at epoch {start_epoch} | best_val_iou={best_val_iou:.4f}")

    # ── Training Loop ─────────────────────────────────────────────────────────
    os.makedirs("weights", exist_ok=True)
    print(f"\nStarting training loop (epochs {start_epoch}-{args.epochs})...")

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == PHASE2_EPOCH + 1:
            train_dataset.max_frame_gap = PHASE2_FRAME_GAP
            train_dataset.absence_prob = ABSENCE_PROB_P2
            print(f">> Phase 2: gap={PHASE2_FRAME_GAP} | absence={ABSENCE_PROB_P2}")

        model.train()
        ep_loss = {'total': 0., 'giou': 0., 'l1': 0., 'focal': 0., 'conf': 0.}
        n_steps = 0
        t_start = time.time()

        for step, batch in enumerate(train_loader):
            template = batch['template'].to(device, non_blocking=True)
            search = batch['search'].to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(template, search)
            losses = compute_loss(out, batch, w_focal=W_FOCAL, w_l1=W_L1,
                                  w_giou=W_GIOU, w_conf=W_CONF)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            for k in ep_loss:
                ep_loss[k] += losses[k].item()
            n_steps += 1
            global_step += 1

            if global_step % CKPT_EVERY_STEPS == 0:
                torch.save({'epoch': epoch, 'global_step': global_step,
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'scaler': scaler.state_dict(),
                            'history': history, 'best_val_iou': best_val_iou,
                            'no_improve': no_improve}, CKPT_PATH)

        avg = {k: v / max(n_steps, 1) for k, v in ep_loss.items()}
        elapsed = time.time() - t_start

        with torch.no_grad():
            val_iou = quick_validate(raw_model, val_sequences, device)
        scheduler.step()

        history.append({**avg, 'epoch': epoch, 'val_iou': val_iou})
        print(f"Ep {epoch} | loss={avg['total']:.4f} | val_iou={val_iou:.4f} | "
              f"time={elapsed / 60:.1f}min")

        # Early stopping (disabled for first 25 epochs)
        if val_iou > best_val_iou + EARLY_STOP_MIN_DELTA:
            best_val_iou = val_iou
            no_improve = 0
            torch.save({'epoch': epoch, 'global_step': global_step,
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'scaler': scaler.state_dict(),
                        'history': history, 'best_val_iou': best_val_iou,
                        'no_improve': no_improve}, BEST_CKPT_PATH)
            print(f"  *** NEW BEST IOU {best_val_iou:.4f} ***")
        elif epoch > 25:
            no_improve += 1
            print(f"  --- No improve {no_improve}/{EARLY_STOP_PATIENCE}")
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        torch.save({'epoch': epoch, 'global_step': global_step,
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'history': history, 'best_val_iou': best_val_iou,
                    'no_improve': no_improve}, CKPT_PATH)

        if (time.time() - session_start) / 3600 >= args.max_hours:
            print(f"\n⏱ Session budget ({args.max_hours}h) exhausted after epoch {epoch}.")
            break

    train_dataset.release_readers()
    print(f"\nTraining done. Best val_iou: {best_val_iou:.4f}")


if __name__ == "__main__":
    main()
