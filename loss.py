# =============================================================================
# loss.py — Loss functions for OSTrackSmall (CornerHead version)
# =============================================================================

import torch
import torch.nn.functional as F


def weighted_focal_loss(pred_logits, target, gamma=2.0, alpha=0.25):
    """Focal loss on raw logits. Target is a Gaussian heatmap."""
    pred_logits = pred_logits.float()
    target = target.float()
    pred_prob = torch.sigmoid(pred_logits)
    ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
    pt = torch.where(target >= 0.5, pred_prob, 1 - pred_prob)
    focal_w = alpha * (1 - pt) ** gamma
    return (focal_w * ce).mean()


def score_map_focal_loss_corner(score_map, gt_corner_norm,
                                search_size=256, patch_size=16, sigma=1.5):
    """Focal loss on score map with Gaussian target centered on TL corner."""
    grid = search_size // patch_size
    B, device = gt_corner_norm.shape[0], gt_corner_norm.device
    cx = gt_corner_norm[:, 0] * grid
    cy = gt_corner_norm[:, 1] * grid
    gx = torch.arange(grid, device=device).float()
    gy = torch.arange(grid, device=device).float()
    grid_y_2d, grid_x_2d = torch.meshgrid(gy, gx, indexing='ij')
    dist_sq = ((grid_x_2d.unsqueeze(0) - cx.view(B, 1, 1)) ** 2 +
               (grid_y_2d.unsqueeze(0) - cy.view(B, 1, 1)) ** 2)
    gauss = torch.exp(-dist_sq / (2 * sigma ** 2)).reshape(B, -1)
    return weighted_focal_loss(score_map, gauss)


def bbox_l1_loss(pred_box, gt_box):
    return F.l1_loss(pred_box, gt_box)


def giou_loss(pred_box, gt_box):
    """GIoU loss. Inputs are (B, 4) in cxcywh normalized format."""
    def to_xyxy(b):
        cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

    p, g = to_xyxy(pred_box), to_xyxy(gt_box)
    inter_x1 = torch.max(p[..., 0], g[..., 0])
    inter_y1 = torch.max(p[..., 1], g[..., 1])
    inter_x2 = torch.min(p[..., 2], g[..., 2])
    inter_y2 = torch.min(p[..., 3], g[..., 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    pred_area = pred_box[..., 2] * pred_box[..., 3]
    gt_area = gt_box[..., 2] * gt_box[..., 3]
    union = pred_area + gt_area - inter
    iou = inter / (union + 1e-7)

    enc_x1 = torch.min(p[..., 0], g[..., 0])
    enc_y1 = torch.min(p[..., 1], g[..., 1])
    enc_x2 = torch.max(p[..., 2], g[..., 2])
    enc_y2 = torch.max(p[..., 3], g[..., 3])
    enc_area = ((enc_x2 - enc_x1) * (enc_y2 - enc_y1)).clamp(1e-7)

    giou_val = iou - (enc_area - union) / enc_area
    return (1 - giou_val).mean()


def focal_bce_loss(pred, target, gamma=2.5, alpha=0.75):
    pred = pred.float()
    target = target.float()
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.where(target == 1, pred, 1 - pred)
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def compute_loss(out, batch, w_focal=1.0, w_l1=5.0, w_giou=2.0, w_conf=1.0):
    """Combined loss — CornerHead version."""
    device = out['confidence'].device
    is_vis = batch['is_visible'].to(device)
    gt_norm = batch['gt_box_norm'].to(device)

    loss_conf = focal_bce_loss(out['confidence'], is_vis)
    vis_mask = is_vis.squeeze(1).bool()

    if vis_mask.sum() == 0:
        return {'total': w_conf * loss_conf,
                'conf': loss_conf,
                'l1': torch.tensor(0., device=device),
                'giou': torch.tensor(0., device=device),
                'focal': torch.tensor(0., device=device)}

    pred_vis = out['bbox'][vis_mask]
    gt_vis = gt_norm[vis_mask]
    score_vis = out['score_map'][vis_mask]

    gt_tl_norm = torch.stack([
        gt_vis[:, 0] - gt_vis[:, 2] / 2,
        gt_vis[:, 1] - gt_vis[:, 3] / 2,
    ], dim=-1)

    loss_focal = score_map_focal_loss_corner(score_vis, gt_tl_norm)
    loss_l1 = bbox_l1_loss(pred_vis, gt_vis)
    loss_giou = giou_loss(pred_vis, gt_vis)

    loss_total = (w_focal * loss_focal +
                  w_l1 * loss_l1 +
                  w_giou * loss_giou +
                  w_conf * loss_conf)

    return {'total': loss_total, 'conf': loss_conf,
            'l1': loss_l1, 'giou': loss_giou, 'focal': loss_focal}
