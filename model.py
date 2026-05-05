# =============================================================================
# model.py — OSTrackSmall Architecture
#
# Components:
#   PatchEmbed16x16       — 16×16 patch projection, ViT-style
#   MixedAttentionLayer   — MAM: template self-attn, search full cross-attn
#   MixedAttentionEncoder — Stack of MAM layers
#   CornerHead            — TL + BR heatmaps with sub-pixel offset regression
#   ConfidenceHead        — Target presence/absence prediction
#   OSTrackSmall          — Full tracker model
#   AerialTracker         — Online inference wrapper with template update
#   load_pretrained_weights — Maps timm ViT-Small weights into OSTrackSmall
# =============================================================================

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from config import (
    DEPTH, EMBED_DIM, NUM_HEADS, NUM_X_TOKENS, NUM_Z_TOKENS,
    PATCH_SIZE, TOTAL_TOKENS, X_SIZE, Z_SIZE,
    CONFIDENCE_THRESHOLD, UPDATE_THRESHOLD, UPDATE_INTERVAL,
    TEMPLATE_ANCHOR_W, HANN_WEIGHT,
)


# ── 1. Patch Embedding ────────────────────────────────────────────────────────

class PatchEmbed16x16(nn.Module):
    """Project image into patch tokens via a single Conv2d with stride=patch_size."""

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) → (B, N_tokens, embed_dim)
        x = self.proj(x)                      # (B, D, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)      # (B, N, D)
        return self.norm(x)


# ── 2. Confidence Head ────────────────────────────────────────────────────────

class ConfidenceHead(nn.Module):
    """
    Predict target presence probability in [0, 1].
    Uses mean + max pooling over search tokens to capture both
    global context and peak activations.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, hidden_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, search_tokens: torch.Tensor) -> torch.Tensor:
        # search_tokens: (B, N_x, D)
        mean_feat = search_tokens.mean(dim=1)            # (B, D)
        max_feat  = search_tokens.max(dim=1).values      # (B, D)
        x = torch.cat([mean_feat, max_feat], dim=-1)     # (B, 2D)
        return torch.sigmoid(self.mlp(x))                # (B, 1)


# ── 3. Corner Head ────────────────────────────────────────────────────────────

class CornerHead(nn.Module):
    """
    Predict bounding box corners (top-left, bottom-right) as heatmaps
    with sub-pixel offset refinement.

    Output: bbox in (cx, cy, w, h) normalized [0,1] format for loss
            score_map (TL logits) for focal loss supervision.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        hidden_dim: int = 256,
        grid: int = X_SIZE // PATCH_SIZE,
    ) -> None:
        super().__init__()
        self.grid = grid

        def _heatmap_branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(embed_dim, hidden_dim, 3, padding=1),
                nn.GroupNorm(32, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
                nn.GroupNorm(16, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, 1, 1),
            )

        def _offset_branch() -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(embed_dim, hidden_dim // 2, 3, padding=1),
                nn.GroupNorm(16, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim // 2, 2, 1),   # (dx, dy) sub-pixel offset
                nn.Sigmoid(),                        # offset in [0, 1] within a cell
            )

        self.tl_heatmap = _heatmap_branch()
        self.br_heatmap = _heatmap_branch()
        self.tl_offset  = _offset_branch()
        self.br_offset  = _offset_branch()

        # Coordinate grid — fixed, moved to device with model
        gx = (torch.arange(grid).float() + 0.5) / grid
        gy = (torch.arange(grid).float() + 0.5) / grid
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")   # (G, G)
        self.register_buffer("grid_x", grid_x.reshape(-1))        # (G*G,)
        self.register_buffer("grid_y", grid_y.reshape(-1))

    def _decode_corner(
        self,
        heatmap_logits: torch.Tensor,   # (B, 1, G, G)
        offset_map: torch.Tensor,        # (B, 2, G, G) in [0,1]
    ) -> torch.Tensor:
        """Soft-argmax + sub-pixel offset → (B, 2) corner in [0,1]."""
        B, _, G, _ = heatmap_logits.shape
        flat_logits = heatmap_logits.reshape(B, -1)           # (B, G*G)
        prob = torch.softmax(flat_logits, dim=-1)              # (B, G*G)

        cx = (prob * self.grid_x.unsqueeze(0)).sum(dim=-1)     # (B,)
        cy = (prob * self.grid_y.unsqueeze(0)).sum(dim=-1)

        off_flat = offset_map.reshape(B, 2, -1)                # (B, 2, G*G)
        dx = (prob * off_flat[:, 0]).sum(dim=-1) / G
        dy = (prob * off_flat[:, 1]).sum(dim=-1) / G

        x = (cx + dx).clamp(0, 1)
        y = (cy + dy).clamp(0, 1)
        return torch.stack([x, y], dim=-1)                     # (B, 2)

    def forward(
        self, search_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            search_tokens: (B, N_x, D)
        Returns:
            bbox:      (B, 4) in (cx, cy, w, h) normalized
            score_map: (B, G*G) TL heatmap logits for focal loss
        """
        B, N, C = search_tokens.shape
        G = self.grid
        feat = search_tokens.transpose(1, 2).reshape(B, C, G, G)

        tl_hm  = self.tl_heatmap(feat)        # (B, 1, G, G)
        br_hm  = self.br_heatmap(feat)
        tl_off = self.tl_offset(feat)          # (B, 2, G, G)
        br_off = self.br_offset(feat)

        tl = self._decode_corner(tl_hm, tl_off)   # (B, 2): x1, y1
        br = self._decode_corner(br_hm, br_off)   # (B, 2): x2, y2

        # Enforce ordering: tl < br
        x1 = torch.min(tl[:, 0], br[:, 0])
        y1 = torch.min(tl[:, 1], br[:, 1])
        x2 = torch.max(tl[:, 0], br[:, 0])
        y2 = torch.max(tl[:, 1], br[:, 1])

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = (x2 - x1).clamp(min=1e-4)
        h  = (y2 - y1).clamp(min=1e-4)

        bbox      = torch.stack([cx, cy, w, h], dim=-1)   # (B, 4)
        score_map = tl_hm.reshape(B, -1)                  # (B, G*G)
        return bbox, score_map


# ── 4. Mixed Attention Module (MAM) ──────────────────────────────────────────

class MixedAttentionLayer(nn.Module):
    """
    MAM: Template tokens use self-attention only.
         Search tokens use full cross-attention over the joint (z+x) sequence.

    This structural prior keeps the template representation clean while allowing
    the search branch to be conditioned on template features at every layer.
    Near-zero overhead vs. standard joint encoding.
    """

    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)

        # Template branch
        self.attn_z  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_z1 = nn.LayerNorm(embed_dim)
        self.norm_z2 = nn.LayerNorm(embed_dim)
        self.ffn_z   = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim), nn.Dropout(dropout),
        )

        # Search branch
        self.attn_x  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_x1 = nn.LayerNorm(embed_dim)
        self.norm_x2 = nn.LayerNorm(embed_dim)
        self.ffn_x   = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, joint: torch.Tensor, n_z: int = NUM_Z_TOKENS) -> torch.Tensor:
        z = joint[:, :n_z, :]    # (B, 64, D)
        x = joint[:, n_z:, :]    # (B, 256, D)

        # Template: self-attention only
        z_attn, _ = self.attn_z(self.norm_z1(z), self.norm_z1(z), self.norm_z1(z))
        z = z + z_attn
        z = z + self.ffn_z(self.norm_z2(z))

        # Search: full attention over joint sequence (template conditions search)
        joint_normed = self.norm_x1(torch.cat([z, x], dim=1))
        x_attn, _ = self.attn_x(self.norm_x1(x), joint_normed, joint_normed)
        x = x + x_attn
        x = x + self.ffn_x(self.norm_x2(x))

        return torch.cat([z, x], dim=1)


class MixedAttentionEncoder(nn.Module):
    """Stack of MixedAttentionLayer blocks."""

    def __init__(
        self,
        num_layers: int = DEPTH,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            MixedAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, joint: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            joint = layer(joint)
        return joint


# ── 5. OSTrackSmall — Full Model ──────────────────────────────────────────────

class OSTrackSmall(nn.Module):
    """
    Full tracker model.

    Forward pass:
        template (B, 3, 128, 128) + search (B, 3, 256, 256)
        → bbox (B, 4), score_map (B, 256), confidence (B, 1), absent (B,)
    """

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD) -> None:
        super().__init__()
        self.confidence_threshold = confidence_threshold

        self.patch_embed = PatchEmbed16x16()
        self.pos_embed   = nn.Parameter(torch.zeros(1, TOTAL_TOKENS, EMBED_DIM))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer    = MixedAttentionEncoder()
        self.confidence_head = ConfidenceHead()
        self.bbox_head       = CornerHead()

        # Persistent template memory for online update
        self.register_buffer("template_memory", torch.zeros(1, NUM_Z_TOKENS, EMBED_DIM))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_template(self, template: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(template)

    def forward(
        self, template: torch.Tensor, search: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        t_tokens = self.patch_embed(template)                             # (B, 64, D)
        s_tokens = self.patch_embed(search)                               # (B, 256, D)
        joint    = torch.cat([t_tokens, s_tokens], dim=1) + self.pos_embed  # (B, 320, D)
        joint    = self.transformer(joint)
        search_out = joint[:, NUM_Z_TOKENS:, :]                           # (B, 256, D)

        confidence         = self.confidence_head(search_out)             # (B, 1)
        bbox, score_map    = self.bbox_head(search_out)                   # (B,4), (B,256)
        absent             = confidence.squeeze(-1) < self.confidence_threshold

        return dict(
            bbox=bbox,
            score_map=score_map,
            confidence=confidence,
            absent=absent,
            search_tokens=search_out,
        )

    def update_template(
        self,
        new_template: torch.Tensor,
        confidence: torch.Tensor,
        update_threshold: float = UPDATE_THRESHOLD,
    ) -> bool:
        if confidence.item() >= update_threshold:
            with torch.no_grad():
                self.template_memory.copy_(self.encode_template(new_template))
            return True
        return False


# ── 6. Pretrained Weight Loader ───────────────────────────────────────────────

def load_pretrained_weights(
    model: OSTrackSmall,
    checkpoint_path: Optional[str] = None,
) -> None:
    """
    Load weights from either:
      - A saved competition checkpoint (tracker_ckpt_*.pt)
      - timm vit_small_patch16_224.augreg_in21k ImageNet-21K pretrained weights

    Positional embeddings are bicubically interpolated from 14×14 (224px)
    to 8×8 (z) and 16×16 (x) grids.
    """

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        raw = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(raw, dict):
            src = raw.get("net", raw.get("model", raw))
        else:
            src = raw
        src = {(k[7:] if k.startswith("module.") else k): v for k, v in src.items()}
    else:
        print("Loading timm vit_small_patch16_224.augreg_in21k (ImageNet-21K pretrain)")
        import timm
        vit = timm.create_model("vit_small_patch16_224.augreg_in21k", pretrained=True)
        src = vit.state_dict()

    dst    = model.state_dict()
    mapped: Dict[str, torch.Tensor] = {}
    skipped = []

    # Patch embed projection
    for suffix in ("weight", "bias"):
        k = f"patch_embed.proj.{suffix}"
        if k in src and k in dst and src[k].shape == dst[k].shape:
            mapped[k] = src[k]
        else:
            skipped.append(k)

    # Transformer layers — map ViT blocks to MAM branches
    for i in range(DEPTH):
        layer_map = [
            (f"blocks.{i}.attn.qkv.weight",
             [f"transformer.layers.{i}.attn_x.in_proj_weight",
              f"transformer.layers.{i}.attn_z.in_proj_weight"]),
            (f"blocks.{i}.attn.qkv.bias",
             [f"transformer.layers.{i}.attn_x.in_proj_bias",
              f"transformer.layers.{i}.attn_z.in_proj_bias"]),
            (f"blocks.{i}.attn.proj.weight",
             [f"transformer.layers.{i}.attn_x.out_proj.weight",
              f"transformer.layers.{i}.attn_z.out_proj.weight"]),
            (f"blocks.{i}.attn.proj.bias",
             [f"transformer.layers.{i}.attn_x.out_proj.bias",
              f"transformer.layers.{i}.attn_z.out_proj.bias"]),
            (f"blocks.{i}.norm1.weight",
             [f"transformer.layers.{i}.norm_x1.weight",
              f"transformer.layers.{i}.norm_z1.weight"]),
            (f"blocks.{i}.norm1.bias",
             [f"transformer.layers.{i}.norm_x1.bias",
              f"transformer.layers.{i}.norm_z1.bias"]),
            (f"blocks.{i}.norm2.weight",
             [f"transformer.layers.{i}.norm_x2.weight",
              f"transformer.layers.{i}.norm_z2.weight"]),
            (f"blocks.{i}.norm2.bias",
             [f"transformer.layers.{i}.norm_x2.bias",
              f"transformer.layers.{i}.norm_z2.bias"]),
            (f"blocks.{i}.mlp.fc1.weight",
             [f"transformer.layers.{i}.ffn_x.0.weight",
              f"transformer.layers.{i}.ffn_z.0.weight"]),
            (f"blocks.{i}.mlp.fc1.bias",
             [f"transformer.layers.{i}.ffn_x.0.bias",
              f"transformer.layers.{i}.ffn_z.0.bias"]),
            (f"blocks.{i}.mlp.fc2.weight",
             [f"transformer.layers.{i}.ffn_x.3.weight",
              f"transformer.layers.{i}.ffn_z.3.weight"]),
            (f"blocks.{i}.mlp.fc2.bias",
             [f"transformer.layers.{i}.ffn_x.3.bias",
              f"transformer.layers.{i}.ffn_z.3.bias"]),
        ]
        for src_key, dst_keys in layer_map:
            if src_key not in src:
                skipped.extend(dst_keys)
                continue
            for dst_key in dst_keys:
                if dst_key in dst and dst[dst_key].shape == src[src_key].shape:
                    mapped[dst_key] = src[src_key].clone()
                else:
                    skipped.append(dst_key)

    # Positional embedding — interpolate from 14×14 to (8×8) + (16×16)
    if "pos_embed" in src:
        pe_src = src["pos_embed"]
        if pe_src.shape[1] == 197:          # strip CLS token
            pe_src = pe_src[:, 1:, :]
        pe_2d = pe_src.reshape(1, 14, 14, EMBED_DIM).permute(0, 3, 1, 2).float()

        pe_z = F.interpolate(
            pe_2d, size=(Z_SIZE // PATCH_SIZE, Z_SIZE // PATCH_SIZE),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1).reshape(1, NUM_Z_TOKENS, EMBED_DIM)

        pe_x = F.interpolate(
            pe_2d, size=(X_SIZE // PATCH_SIZE, X_SIZE // PATCH_SIZE),
            mode="bilinear", align_corners=False,
        ).permute(0, 2, 3, 1).reshape(1, NUM_X_TOKENS, EMBED_DIM)

        mapped["pos_embed"] = torch.cat([pe_z, pe_x], dim=1)
    else:
        skipped.append("pos_embed")

    dst.update(mapped)
    model.load_state_dict(dst)
    print(f"Weight loading complete: mapped={len(mapped)}/{len(dst)}, skipped={len(skipped)}")
    print("  Heads (CornerHead, ConfidenceHead) — random init, trained from scratch")


# ── 7. AerialTracker — Online Inference Wrapper ───────────────────────────────

class AerialTracker:
    """
    Online single-object tracker for aerial/UAV sequences.

    Implements:
      - Multi-scale search (3 scales) with Hann-penalized score map
      - Exponential moving average template update
      - Absence detection with progressive search region expansion
      - Score-map cross-check for robust bbox decoding
    """

    def __init__(
        self,
        model: OSTrackSmall,
        device: torch.device,
        search_size: int = X_SIZE,
        template_size: int = Z_SIZE,
        update_interval: int = UPDATE_INTERVAL,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        update_threshold: float = UPDATE_THRESHOLD,
    ) -> None:
        self.model               = model.to(device).eval()
        self.device              = device
        self.search_size         = search_size
        self.template_size       = template_size
        self.update_interval     = update_interval
        self.update_threshold    = update_threshold
        self.confidence_threshold = confidence_threshold

        # State reset on initialise()
        self.template:       Optional[torch.Tensor] = None
        self.template_init:  Optional[torch.Tensor] = None
        self.last_center:    Optional[Tuple[float, float]] = None
        self.target_wh:      Optional[Tuple[float, float]] = None
        self.frame_count:    int = 0
        self.absent_count:   int = 0
        self.search_scale_multiplier: float = 1.0
        self._anchor_w = TEMPLATE_ANCHOR_W
        self.hann_weight = HANN_WEIGHT

        # Hann window for score map penalty
        grid = search_size // PATCH_SIZE
        h1d  = torch.hann_window(grid, periodic=False)
        self.hann = (h1d.unsqueeze(1) * h1d.unsqueeze(0)).reshape(1, -1)

    def initialise(
        self,
        frame: torch.Tensor,
        gt_bbox_xywh: Tuple[float, float, float, float],
    ) -> None:
        """Set up tracker state from ground-truth box on frame 0."""
        x, y, w, h = [float(v) for v in gt_bbox_xywh]
        self.last_center  = (x + w / 2, y + h / 2)
        self.target_wh    = (max(w, 10.0), max(h, 10.0))
        self.frame_count  = 0
        self.absent_count = 0
        self.search_scale_multiplier = 1.0

        patch, _ = self._get_patch(frame, self.last_center, self.target_wh, self.template_size)
        self.template      = patch.unsqueeze(0).to(self.device)
        self.template_init = self.template.clone()

    @torch.no_grad()
    def track(self, frame: torch.Tensor) -> dict:
        """
        Process one frame.
        Returns dict with keys: bbox_xywh (int tuple), confidence (float), absent (bool).
        """
        self.frame_count += 1

        # Blend running template with anchored initial template for stability
        template_mixed = (
            (1 - self._anchor_w) * self.template +
            self._anchor_w * self.template_init
        )

        hann     = self.hann.to(self.device)
        hann_w   = self.hann_weight * hann + (1.0 - self.hann_weight)

        # Multi-scale search: pick scale with best confidence + penalized score
        best_metric = -1e9
        best_out = best_region = None

        for scale in (0.95, 1.0, 1.10):
            scaled_wh = (
                self.target_wh[0] * scale * self.search_scale_multiplier,
                self.target_wh[1] * scale * self.search_scale_multiplier,
            )
            patch, region = self._get_patch(frame, self.last_center, scaled_wh, self.search_size)
            search_s      = patch.unsqueeze(0).to(self.device)
            out_s         = self.model(template_mixed, search_s)
            penalized     = (out_s["score_map"] * hann_w).max().item()
            metric        = out_s["confidence"].item() + 0.3 * penalized
            if metric > best_metric:
                best_metric = metric
                best_out    = out_s
                best_region = region

        confidence = best_out["confidence"].item()
        is_absent  = best_out["absent"].item()

        def _fallback_xywh() -> Tuple[int, int, int, int]:
            cx, cy = self.last_center
            tw, th = self.target_wh
            return (max(0, int(cx - tw / 2)), max(0, int(cy - th / 2)),
                    max(1, int(tw)), max(1, int(th)))

        if is_absent:
            self.absent_count += 1
            # Progressively expand search region during prolonged absence
            if self.absent_count > 30:
                self.search_scale_multiplier = 2.5
            elif self.absent_count > 15:
                self.search_scale_multiplier = 2.0
            elif self.absent_count > 5:
                self.search_scale_multiplier = 1.5
            return {"bbox_xywh": _fallback_xywh(), "confidence": confidence, "absent": True}

        self.search_scale_multiplier = 1.0
        sx1, sy1, sx2, sy2 = best_region
        sw, sh = sx2 - sx1, sy2 - sy1

        hann_cpu = self.hann.squeeze(0).cpu()
        pred_cx, pred_cy, pred_w_raw, pred_h_raw = self._decode_with_scoremap_crosscheck(
            best_out["bbox"].squeeze(0).cpu(),
            best_out["score_map"].squeeze(0).cpu(),
            best_region,
            hann_cpu,
        )

        ref_w, ref_h = self.target_wh
        pred_w = float(np.clip(pred_w_raw, max(4.0, 0.20 * ref_w), 5.0 * ref_w))
        pred_h = float(np.clip(pred_h_raw, max(4.0, 0.20 * ref_h), 5.0 * ref_h))

        # Discard degenerate predictions — fall back to prior size
        if pred_w_raw < 6.0 or pred_h_raw < 6.0:
            pred_w, pred_h = ref_w, ref_h

        # Smooth center position
        cx0, cy0 = self.last_center
        self.last_center = (0.75 * pred_cx + 0.25 * cx0, 0.75 * pred_cy + 0.25 * cy0)

        # Smooth size
        self.target_wh = (
            max(4.0, 0.65 * ref_w + 0.35 * pred_w),
            max(4.0, 0.65 * ref_h + 0.35 * pred_h),
        )
        self.absent_count = 0

        # Template update via EMA
        if self.frame_count % self.update_interval == 0 and confidence > 0.5:
            new_patch, _ = self._get_patch(
                frame, self.last_center, self.target_wh, self.template_size
            )
            alpha = 0.05
            self.template = (
                (1 - alpha) * self.template +
                alpha * new_patch.unsqueeze(0).to(self.device)
            )

        cx, cy = self.last_center
        return {
            "bbox_xywh": (
                max(0, int(cx - pred_w / 2)),
                max(0, int(cy - pred_h / 2)),
                max(4, int(pred_w)),
                max(4, int(pred_h)),
            ),
            "confidence": confidence,
            "absent": False,
        }

    def _decode_with_scoremap_crosscheck(
        self,
        bbox_pred: torch.Tensor,     # (4,) cx cy w h normalized
        score_map: torch.Tensor,     # (G*G,) raw logits
        search_region: tuple,        # (sx1, sy1, sx2, sy2)
        hann_window: torch.Tensor,   # (G*G,)
    ) -> Tuple[float, float, float, float]:
        """
        Cross-check CornerHead bbox against score-map peak.
        If they agree (distance < 15% of search window), trust CornerHead.
        Otherwise fall back to score-map peak position with prior size.
        """
        sx1, sy1, sx2, sy2 = search_region
        sw, sh = sx2 - sx1, sy2 - sy1
        G = self.search_size // PATCH_SIZE

        penalized = torch.softmax(score_map * hann_window, dim=-1)
        peak_idx  = penalized.argmax().item()
        peak_gx   = peak_idx % G
        peak_gy   = peak_idx // G
        peak_cx_img = sx1 + ((peak_gx + 0.5) / G) * sw
        peak_cy_img = sy1 + ((peak_gy + 0.5) / G) * sh

        box_cx_img = sx1 + bbox_pred[0].item() * sw
        box_cy_img = sy1 + bbox_pred[1].item() * sh

        dist      = np.sqrt((peak_cx_img - box_cx_img) ** 2 + (peak_cy_img - box_cy_img) ** 2)
        threshold = 0.15 * max(sw, sh)

        if dist < threshold:
            return (box_cx_img, box_cy_img,
                    bbox_pred[2].item() * sw,
                    bbox_pred[3].item() * sh)
        else:
            return (peak_cx_img, peak_cy_img, self.target_wh[0], self.target_wh[1])

    def _get_patch(
        self,
        frame: torch.Tensor,
        center: Tuple[float, float],
        target_wh: Tuple[float, float],
        patch_size: int,
    ) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        cx, cy = center
        tw, th = target_wh
        H, W   = frame.shape[1], frame.shape[2]
        p      = (tw + th) / 2
        crop_sz = float(np.sqrt((tw + p) * (th + p))) * 2.0
        half    = crop_sz / 2
        sx1 = max(0, int(cx - half));  sy1 = max(0, int(cy - half))
        sx2 = min(W, int(cx + half));  sy2 = min(H, int(cy + half))
        crop = frame[:, sy1:sy2, sx1:sx2]
        if crop.shape[1] == 0 or crop.shape[2] == 0:
            crop = torch.zeros(3, patch_size, patch_size)
        patch = TF.resize(crop, [patch_size, patch_size], antialias=True)
        return patch, (sx1, sy1, sx2, sy2)
