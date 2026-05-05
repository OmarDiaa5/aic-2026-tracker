# =============================================================================
# data_pipeline.py — Data loading, annotation parsing, crop extraction,
#                    augmentation, and dataset construction.
# =============================================================================

import cv2
import json
import os
import random
import numpy as np
import torch
import torch.utils.data

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Annotation Parsing ─────────────────────────────────────────────────────────

def parse_annotation_file(ann_path: Path) -> List[np.ndarray]:
    """Parse annotation.txt → list of [x, y, w, h] float32 arrays."""
    boxes: List[np.ndarray] = []
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            normalized = line.replace("\t", ",").replace(" ", ",")
            parts = [p for p in normalized.split(",") if p]
            try:
                if len(parts) < 4:
                    raise ValueError
                xywh = np.array([float(p) for p in parts[:4]], dtype=np.float32)
            except (ValueError, IndexError):
                xywh = np.zeros(4, dtype=np.float32)
            boxes.append(xywh)
    return boxes


def is_box_visible(box: np.ndarray, min_dim: float = 1.0) -> bool:
    """True if target is visible. Absent = w<=1 OR h<=1."""
    return float(box[2]) > min_dim and float(box[3]) > min_dim


# ── VideoReader with LRU Cache ─────────────────────────────────────────────────

class VideoReader:
    def __init__(self, video_path: Path, cache_size: int = 50) -> None:
        self.video_path = Path(video_path)
        self._cache_size = cache_size
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cap: Optional[cv2.VideoCapture] = None
        self._decoder_pos = -1
        self._n_frames = 0
        self._height = 0
        self._width = 0
        self._open()

    def _open(self) -> None:
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open: {self.video_path}")
        self._n_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._decoder_pos = -1

    def release(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
        self._cap = None

    def __del__(self) -> None:
        self.release()

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def get_frame(self, frame_idx: int) -> np.ndarray:
        if not (0 <= frame_idx < self._n_frames):
            raise IndexError(f"Frame {frame_idx} out of range [0, {self._n_frames})")
        if frame_idx in self._cache:
            self._cache.move_to_end(frame_idx)
            return self._cache[frame_idx]
        if self._cap is None or not self._cap.isOpened():
            self._open()
        if frame_idx != self._decoder_pos + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
            self._decoder_pos = frame_idx - 1
        ret, frame = self._cap.read()
        if not ret or frame is None:
            frame = np.zeros((self._height or 720, self._width or 1280, 3), dtype=np.uint8)
        self._decoder_pos = frame_idx
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[frame_idx] = frame
        return frame

    def get_frame_rgb(self, frame_idx: int) -> np.ndarray:
        return cv2.cvtColor(self.get_frame(frame_idx), cv2.COLOR_BGR2RGB)


# ── FastFrameReader (JPEG-based, training only) ───────────────────────────────

def seq_id_to_dir_name(seq_id):
    return seq_id.replace("/", "__")


class FastFrameReader:
    def __init__(self, seq_id: str, video_path: Path, n_frames: int,
                 cache_size: int = 3, frames_dirs: Optional[List[Path]] = None):
        self.seq_id = seq_id
        self.video_path = Path(video_path)
        safe_seq_id = seq_id_to_dir_name(seq_id)

        if frames_dirs is None:
            frames_dirs = [
                Path("/kaggle/input/datasets/youssefzzein/e-eye-dataset"),
                Path("/kaggle/input/frames-dataset/_output_/frames"),
                Path("/kaggle/input/datasets/omardiaa05/frames-dataset-fix"),
                Path("/kaggle/input/datasets/diaaeldien/frames-dataset-fix3/frames_fix3/frames"),
            ]

        def _valid_dir(base: Path) -> bool:
            d = base / safe_seq_id
            return d.exists() and len(list(d.glob("*.jpg"))) == n_frames

        self._use_jpegs = False
        for fdir in frames_dirs:
            if _valid_dir(fdir):
                self._use_jpegs = True
                self.jpeg_dir = fdir / safe_seq_id
                break

        if self._use_jpegs:
            self._frames = sorted(self.jpeg_dir.glob("*.jpg"))
            self._n_frames = len(self._frames)
        else:
            self.cap = cv2.VideoCapture(str(video_path))
            self._n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def n_frames(self) -> int:
        return self._n_frames

    def get_frame_rgb(self, idx: int) -> np.ndarray:
        if self._use_jpegs:
            bgr = cv2.imread(str(self._frames[idx]))
            if bgr is None:
                return np.zeros((720, 1280, 3), dtype=np.uint8)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, bgr = self.cap.read()
            if not ret or bgr is None:
                return np.zeros((720, 1280, 3), dtype=np.uint8)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        if not self._use_jpegs and hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()


# ── Crop Utilities ─────────────────────────────────────────────────────────────

def compute_crop_size(w: float, h: float, crop_scale: float = 1.0) -> float:
    p = (w + h) / 2.0
    return float(np.sqrt((w + p) * (h + p))) * crop_scale


def extract_crop(image: np.ndarray, cx: float, cy: float,
                 crop_sz: float, output_sz: int) -> Tuple[np.ndarray, float]:
    H, W = image.shape[:2]
    half = crop_sz / 2.0
    x1, y1 = round(cx - half), round(cy - half)
    x2, y2 = x1 + round(crop_sz), y1 + round(crop_sz)

    left_pad = int(max(0, -x1)); top_pad = int(max(0, -y1))
    right_pad = int(max(0, x2 - W)); bot_pad = int(max(0, y2 - H))
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    visible = image[y1c:y2c, x1c:x2c]

    if left_pad > 0 or top_pad > 0 or right_pad > 0 or bot_pad > 0:
        mean_color = visible.reshape(-1, 3).mean(axis=0).tolist() if visible.size > 0 else [128.0] * 3
        visible = cv2.copyMakeBorder(visible, top_pad, bot_pad, left_pad, right_pad,
                                     borderType=cv2.BORDER_CONSTANT, value=mean_color)

    if visible.shape[0] != output_sz or visible.shape[1] != output_sz:
        interp = cv2.INTER_AREA if output_sz < crop_sz else cv2.INTER_LINEAR
        visible = cv2.resize(visible, (output_sz, output_sz), interpolation=interp)

    return visible, output_sz / crop_sz


def box_to_crop_coords(box_xywh, cx, cy, crop_sz, output_sz):
    scale = output_sz / crop_sz
    origin_x = cx - crop_sz / 2.0
    origin_y = cy - crop_sz / 2.0
    x, y, w, h = box_xywh.astype(np.float64)
    return np.array([(x - origin_x) * scale, (y - origin_y) * scale,
                     w * scale, h * scale], dtype=np.float32)


def _clamp_box(box, size):
    x, y, w, h = box.astype(np.float64)
    x = float(np.clip(x, 0.0, size - 1.0)); y = float(np.clip(y, 0.0, size - 1.0))
    w = float(np.clip(w, 0.0, size - x));   h = float(np.clip(h, 0.0, size - y))
    return np.array([x, y, w, h], dtype=np.float32)


def crop_box_to_normalized_center(box_crop, crop_size):
    x, y, w, h = box_crop.astype(np.float32)
    return np.array([(x + w / 2) / crop_size, (y + h / 2) / crop_size,
                     w / crop_size, h / crop_size], dtype=np.float32)


def extract_template_and_search(frame_z, frame_x, box_z, box_x,
                                z_size=128, x_size=256,
                                template_scale=1.0, search_scale=2.0):
    x_z, y_z, w_z, h_z = box_z.astype(np.float64)
    cx_z = x_z + w_z / 2.0; cy_z = y_z + h_z / 2.0

    base_sz = compute_crop_size(w_z, h_z, crop_scale=1.0)
    templ_crop_sz = base_sz * template_scale
    search_crop_sz = base_sz * search_scale

    template, _ = extract_crop(frame_z, cx_z, cy_z, templ_crop_sz, z_size)
    search, _ = extract_crop(frame_x, cx_z, cy_z, search_crop_sz, x_size)

    vis = is_box_visible(box_x)
    if vis:
        gt_crop = box_to_crop_coords(box_x, cx_z, cy_z, search_crop_sz, x_size)
        gt_crop = _clamp_box(gt_crop, x_size)
        gt_norm = crop_box_to_normalized_center(gt_crop, x_size)
    else:
        gt_crop = np.zeros(4, dtype=np.float32)
        gt_norm = np.zeros(4, dtype=np.float32)

    return {"template": template, "search": search,
            "gt_box_crop": gt_crop, "gt_box_norm": gt_norm, "is_visible": vis}


# ── Search Crop Augmentation ──────────────────────────────────────────────────

def augment_search_crop(search_img, gt_box_crop, is_vis, crop_size=256):
    img = search_img.copy()
    box = gt_box_crop.copy() if is_vis else np.zeros(4, dtype=np.float32)
    W = crop_size

    # 1. Horizontal flip
    if random.random() < 0.5:
        img = np.ascontiguousarray(np.fliplr(img))
        if is_vis and box[2] > 0:
            box[0] = W - box[0] - box[2]
            box = _clamp_box(box, W)
            if box[2] < 2 or box[3] < 2:
                is_vis = False
                box = np.zeros(4, dtype=np.float32)

    # 2. Color jitter
    if random.random() < 0.8:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= random.uniform(0.6, 1.5)
        hsv[:, :, 2] *= random.uniform(0.6, 1.5)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 3. Gaussian blur
    if random.random() < 0.25:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # 4. Altitude scale simulation
    if random.random() < 0.4:
        scale_factor = random.uniform(0.75, 1.35)
        new_size = int(W * scale_factor)
        img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        if new_size > W:
            start = (new_size - W) // 2
            img = img[start:start + W, start:start + W]
        else:
            pad = (W - new_size) // 2
            pad_r = W - new_size - pad
            mean_color = img.reshape(-1, 3).mean(axis=0).tolist() if img.size > 0 else [128.0] * 3
            img = cv2.copyMakeBorder(img, pad, pad_r, pad, pad_r,
                                     cv2.BORDER_CONSTANT, value=mean_color)
        if is_vis and box[2] > 0:
            box = box * scale_factor
            if new_size > W:
                start = (new_size - W) // 2
                box[0] -= start; box[1] -= start
            else:
                pad = (W - new_size) // 2
                box[0] += pad; box[1] += pad
            box = _clamp_box(box, W)
            if box[2] < 2 or box[3] < 2:
                is_vis = False
                box = np.zeros(4, dtype=np.float32)

    # 5. Directional motion blur
    if random.random() < 0.3:
        kernel_size = random.randint(7, 19)
        if kernel_size % 2 == 0: kernel_size += 1
        angle = random.uniform(0, 180)
        M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0
        kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
        kernel = kernel / np.sum(kernel)
        img = cv2.filter2D(img, -1, kernel)

    # 6. Perspective jitter
    if random.random() < 0.25:
        distortion = 0.05 * W
        pts1 = np.float32([[0, 0], [W, 0], [0, W], [W, W]])
        pts2 = np.float32([
            [random.uniform(-distortion, distortion), random.uniform(-distortion, distortion)],
            [W + random.uniform(-distortion, distortion), random.uniform(-distortion, distortion)],
            [random.uniform(-distortion, distortion), W + random.uniform(-distortion, distortion)],
            [W + random.uniform(-distortion, distortion), W + random.uniform(-distortion, distortion)]
        ])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (W, W), borderMode=cv2.BORDER_REPLICATE)
        if is_vis and box[2] > 0:
            box_pts = np.float32([
                [box[0], box[1]], [box[0] + box[2], box[1]],
                [box[0], box[1] + box[3]], [box[0] + box[2], box[1] + box[3]]
            ]).reshape(-1, 1, 2)
            trans_pts = cv2.perspectiveTransform(box_pts, M).reshape(-1, 2)
            x_min, y_min = np.min(trans_pts, axis=0)
            x_max, y_max = np.max(trans_pts, axis=0)
            box = np.array([x_min, y_min, x_max - x_min, y_max - y_min], dtype=np.float32)
            box = _clamp_box(box, W)
            if box[2] < 2 or box[3] < 2:
                is_vis = False
                box = np.zeros(4, dtype=np.float32)

    # Recompute normalized box
    if is_vis and box[2] > 0 and box[3] > 0:
        box = _clamp_box(box, W)
        norm = crop_box_to_normalized_center(box, W)
    else:
        box = np.zeros(4, dtype=np.float32)
        norm = np.zeros(4, dtype=np.float32)
        is_vis = False

    return img, box, norm, is_vis


# ── Tensor Conversion ──────────────────────────────────────────────────────────

def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    chw = np.ascontiguousarray(image.transpose(2, 0, 1))
    return torch.from_numpy(chw.astype(np.float32) / 255.0)


# ── SequenceInfo Dataclass ─────────────────────────────────────────────────────

@dataclass
class SequenceInfo:
    seq_id:       str
    video_path:   Path
    ann_path:     Path
    boxes:        List[np.ndarray] = field(default_factory=list)
    visible:      List[bool]       = field(default_factory=list)
    visible_idxs: List[int]        = field(default_factory=list)

    @property
    def n_frames(self): return len(self.boxes)

    @property
    def absence_ratio(self):
        if not self.visible: return 0.0
        return (len(self.visible) - sum(self.visible)) / len(self.visible)


# ── TrackingDataset ────────────────────────────────────────────────────────────

class TrackingDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, pairs_per_seq=200, max_frame_gap=100,
                 z_size=128, x_size=256, template_scale=1.0, search_scale=2.0,
                 use_augmentation=True, absence_prob=0.15,
                 video_cache_size=3, max_open_readers=8):
        super().__init__()
        self.sequences = sequences
        self.pairs_per_seq = pairs_per_seq
        self.max_frame_gap = max_frame_gap
        self.z_size = z_size
        self.x_size = x_size
        self.template_scale = template_scale
        self.search_scale = search_scale
        self.use_augmentation = use_augmentation
        self.absence_prob = absence_prob
        self.video_cache_size = video_cache_size
        self._max_open_readers = max_open_readers
        self._total_len = len(sequences) * pairs_per_seq
        self._readers: OrderedDict[str, FastFrameReader] = OrderedDict()

    def __len__(self): return self._total_len

    def __getitem__(self, idx):
        seq = self.sequences[(idx // self.pairs_per_seq) % len(self.sequences)]
        z_idx, x_idx, force_absent = self._sample_pair(seq)
        reader = self._get_reader(seq)

        frame_z = reader.get_frame_rgb(z_idx)
        frame_x = reader.get_frame_rgb(x_idx)
        box_z = seq.boxes[z_idx].copy()
        box_x = seq.boxes[x_idx].copy() if not force_absent else np.zeros(4, dtype=np.float32)
        is_vis = seq.visible[x_idx] and not force_absent

        crops = extract_template_and_search(
            frame_z, frame_x, box_z, box_x,
            z_size=self.z_size, x_size=self.x_size,
            template_scale=self.template_scale, search_scale=self.search_scale)
        is_visible = is_vis and crops["is_visible"]

        search_img = crops["search"]
        gt_box_crop = crops["gt_box_crop"]
        gt_box_norm = crops["gt_box_norm"]

        if self.use_augmentation:
            search_img, gt_box_crop, gt_box_norm, is_visible = augment_search_crop(
                search_img, gt_box_crop, is_visible, crop_size=self.x_size)

        return {
            "template":    image_to_tensor(crops["template"]),
            "search":      image_to_tensor(search_img),
            "gt_box_norm": torch.from_numpy(gt_box_norm),
            "gt_box_crop": torch.from_numpy(gt_box_crop),
            "is_visible":  torch.tensor([1.0 if is_visible else 0.0]),
            "seq_id":      seq.seq_id,
            "frame_z_idx": z_idx,
            "frame_x_idx": x_idx,
        }

    def _sample_pair(self, seq):
        if not seq.visible_idxs:
            return 0, min(1, seq.n_frames - 1), False
        z_idx = random.choice(seq.visible_idxs)
        if random.random() < self.absence_prob:
            absent_idxs = [i for i in range(seq.n_frames) if not seq.visible[i]]
            if absent_idxs:
                return z_idx, random.choice(absent_idxs), False
            else:
                x_idx = random.randint(0, seq.n_frames - 1)
                return z_idx, x_idx, True
        x_idx = random.randint(z_idx, min(z_idx + self.max_frame_gap, seq.n_frames - 1))
        return z_idx, x_idx, False

    def _get_reader(self, seq) -> FastFrameReader:
        if seq.seq_id in self._readers:
            self._readers.move_to_end(seq.seq_id)
            return self._readers[seq.seq_id]
        if len(self._readers) >= self._max_open_readers:
            _, evicted = self._readers.popitem(last=False)
            evicted.release()
        self._readers[seq.seq_id] = FastFrameReader(
            seq.seq_id, seq.video_path, seq.n_frames, cache_size=self.video_cache_size)
        return self._readers[seq.seq_id]

    def release_readers(self):
        for r in self._readers.values(): r.release()
        self._readers.clear()

    def get_absence_stats(self):
        ratios = [s.absence_ratio for s in self.sequences]
        seqs_with = sum(1 for r in ratios if r > 0)
        return {"n_sequences": len(self.sequences), "seqs_with_absence": seqs_with,
                "fraction_with_absence": seqs_with / max(1, len(self.sequences)),
                "mean_absence_ratio": float(np.mean(ratios)),
                "max_absence_ratio": float(np.max(ratios)) if ratios else 0.0}


def tracking_collate_fn(batch):
    return {
        "template":    torch.stack([b["template"]    for b in batch]),
        "search":      torch.stack([b["search"]      for b in batch]),
        "gt_box_norm": torch.stack([b["gt_box_norm"] for b in batch]),
        "gt_box_crop": torch.stack([b["gt_box_crop"] for b in batch]),
        "is_visible":  torch.stack([b["is_visible"]  for b in batch]),
        "seq_id":      [b["seq_id"]      for b in batch],
        "frame_z_idx": [b["frame_z_idx"] for b in batch],
        "frame_x_idx": [b["frame_x_idx"] for b in batch],
    }


# ── Video Path Discovery ──────────────────────────────────────────────────────

def _validate_video(path: Path) -> bool:
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            return False
        ret, _ = cap.read()
        cap.release()
        return ret
    except Exception:
        return False


def _find_video_path(seq_dir: Path, manifest_video_path: Path) -> Optional[Path]:
    if manifest_video_path.exists():
        return manifest_video_path
    mp4_files = list(seq_dir.glob("*.mp4"))
    if len(mp4_files) == 1:
        return mp4_files[0]
    elif len(mp4_files) > 1:
        return max(mp4_files, key=lambda p: p.stat().st_size)
    return None


def build_dataset(root, manifest_path, split="train",
                  pairs_per_seq=200, max_frame_gap=100,
                  z_size=128, x_size=256,
                  template_scale=1.0, search_scale=2.0,
                  min_visible_frames=2, video_cache_size=50) -> TrackingDataset:
    root = Path(root)
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    if split == "val":
        split_seqs = dict(list(manifest.get("train", {}).items())[-20:])
    else:
        split_seqs = manifest.get(split, {})
        if split == "train":
            split_seqs = dict(list(split_seqs.items())[:-20])
    if not split_seqs:
        raise ValueError(f"Split '{split}' not in manifest.")
    print(f"Manifest: {len(split_seqs)} sequences in split='{split}'")
    sequences = []
    skipped_missing = skipped_short = skipped_corrupt = 0
    for seq_id, meta in split_seqs.items():
        manifest_video_path = root / meta["video_path"]
        ann_path = root / meta["annotation_path"]
        seq_dir = manifest_video_path.parent
        video_path = _find_video_path(seq_dir, manifest_video_path)
        if video_path is None or not ann_path.exists():
            skipped_missing += 1
            continue
        if not _validate_video(video_path):
            skipped_corrupt += 1
            continue
        boxes = parse_annotation_file(ann_path)
        visible = [is_box_visible(b) for b in boxes]
        visible_idxs = [i for i, v in enumerate(visible) if v]
        if len(visible_idxs) < min_visible_frames:
            skipped_short += 1
            continue
        sequences.append(SequenceInfo(
            seq_id=seq_id, video_path=video_path, ann_path=ann_path,
            boxes=boxes, visible=visible, visible_idxs=visible_idxs))
    print(f"Loaded {len(sequences)} sequences "
          f"(skipped {skipped_missing} missing, "
          f"{skipped_corrupt} corrupt, {skipped_short} too-short)")
    if not sequences:
        raise RuntimeError("No usable sequences found.")
    return TrackingDataset(
        sequences=sequences, pairs_per_seq=pairs_per_seq,
        max_frame_gap=max_frame_gap, z_size=z_size, x_size=x_size,
        template_scale=template_scale, search_scale=search_scale,
        video_cache_size=video_cache_size)
