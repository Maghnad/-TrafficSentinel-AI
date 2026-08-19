"""Helmet classification.

The original HSV skin-tone heuristic has two failure modes that matter for
Indian footage specifically: fixed skin-tone hue ranges are unreliable across
darker skin tones, and a black helmet against black hair is nearly identical
in both skin ratio and saturation uniformity. Since NO_HELMET is the headline
violation, resting it on the weakest component in the system is the wrong
trade.

The fix is a small fine-tuned classifier. YOLOv8n-cls on ~2000 cropped head
images trains in about 15 minutes on a free Colab GPU and runs in ~3 ms per
crop on CPU. See README for the training script.

If no model file is present the class falls back to the improved multi-cue
HSV heuristic below, which is better than returning UNCERTAIN for every rider.
Uncertain calls still route to the human review queue; the HSV fallback simply
raises the probability of actually surfacing genuine violations there.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def enhance_crop(crop: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel. Applied HERE, on a small crop, rather than on
    the full frame before detection - this is where it actually pays off."""
    if crop.size == 0:
        return crop
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _hsv_classify(crop: np.ndarray) -> Tuple[str, float]:
    """Multi-cue HSV fallback when no trained model is available.

    Three complementary cues are combined so that any single one failing
    (e.g. dark skin matching hair in S channel) does not flip the result:

    1. Skin-tone pixel ratio  - skin visible = likely no helmet
    2. Edge density           - helmets are smooth; hair is textured
    3. Colour uniformity      - helmets are single-colour; hair varies

    Confidence is intentionally capped at 0.72 so results always land in
    the review queue rather than being auto-issued as a challan.
    """
    if crop.size == 0:
        return "UNCERTAIN", 0.0

    crop_e = enhance_crop(crop)
    hsv = cv2.cvtColor(crop_e, cv2.COLOR_BGR2HSV)

    # --- Cue 1: skin-tone ratio (HSV) ---
    # Broad range covering South-Asian skin tones
    mask_lo = cv2.inRange(hsv, (0, 18, 60),  (22, 220, 255))
    mask_hi = cv2.inRange(hsv, (165, 18, 60), (180, 220, 255))
    skin_mask = cv2.bitwise_or(mask_lo, mask_hi)
    skin_ratio = float(np.count_nonzero(skin_mask)) / max(skin_mask.size, 1)

    # --- Cue 2: edge density (Canny on gray) ---
    gray = cv2.cvtColor(crop_e, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edge_density = float(np.count_nonzero(edges)) / max(edges.size, 1)

    # --- Cue 3: HSV saturation uniformity ---
    sat = hsv[:, :, 1].astype(float)
    sat_std = float(np.std(sat))

    # --- Decision ---
    # High skin exposure -> bare head -> NO_HELMET
    if skin_ratio > 0.20:
        conf = min(0.72, 0.48 + skin_ratio * 0.80)
        return "NO_HELMET", round(conf, 2)

    # High edge density -> hair texture -> likely no helmet
    if edge_density > 0.18 and skin_ratio > 0.08:
        conf = min(0.65, 0.42 + edge_density * 1.2)
        return "NO_HELMET", round(conf, 2)

    # Low skin + low edge density + low saturation std -> uniform object = HELMET
    if skin_ratio < 0.10 and edge_density < 0.10 and sat_std < 38:
        conf = min(0.72, 0.50 + (38 - sat_std) / 80.0)
        return "HELMET", round(conf, 2)

    return "UNCERTAIN", 0.40


class HelmetClassifier:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.names = {}
        path = cfg.model_path
        if path and Path(path).exists():
            try:
                from ultralytics import YOLO
                self.model = YOLO(path, task="classify")
                self.names = self.model.names
                print(f"[helmet] loaded classifier: {path}")
            except Exception as exc:
                print(f"[helmet] failed to load {path}: {exc}")
        else:
            print("[helmet] no classifier found - using HSV multi-cue fallback. "
                  "All calls are capped at conf<0.75 and route to review. "
                  "Train models/helmet_cls.pt for production accuracy.")

    @property
    def available(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------------ #

    def head_crop(self, frame: np.ndarray, person_bbox) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = [int(round(v)) for v in person_bbox]
        h = y2 - y1
        if h < self.cfg.min_head_px * 2:  # lowered from *3: rider at 80px needs 36px not 54px
            return None
        hh = max(self.cfg.min_head_px, int(h * self.cfg.head_crop_ratio))
        # Widen slightly - helmets are wider than the head.
        pad = int(0.12 * (x2 - x1))
        cx1 = max(0, x1 - pad)
        cx2 = min(frame.shape[1], x2 + pad)
        cy1 = max(0, y1)
        cy2 = min(frame.shape[0], y1 + hh)
        if cx2 <= cx1 or cy2 <= cy1:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < self.cfg.min_head_px:
            return None
        return crop

    def classify(self, frame: np.ndarray, person_bbox) -> Tuple[str, float]:
        """Returns (status, confidence) where status is one of
        HELMET / NO_HELMET / UNCERTAIN.

        If the trained ML model is loaded, it is used exclusively.
        Otherwise the multi-cue HSV fallback runs, keeping the system
        operational without a GPU or labelled dataset.
        """
        crop = self.head_crop(frame, person_bbox)
        if crop is None:
            return "UNCERTAIN", 0.0

        # --- ML model path ---
        if self.model is not None:
            crop_e = enhance_crop(crop)
            crop_r = cv2.resize(crop_e, (96, 96), interpolation=cv2.INTER_CUBIC)
            try:
                res = self.model.predict(crop_r, verbose=False, imgsz=96)
            except Exception:
                return "UNCERTAIN", 0.0
            if not res or res[0].probs is None:
                return "UNCERTAIN", 0.0

            probs = res[0].probs
            idx = int(probs.top1)
            conf = float(probs.top1conf)
            name = str(self.names.get(idx, "")).lower()

            if conf < 0.55:
                return "UNCERTAIN", conf
            if "no" in name or "without" in name or "bare" in name:
                return "NO_HELMET", conf
            if "helmet" in name:
                return "HELMET", conf
            return "UNCERTAIN", conf

        # --- HSV fallback path ---
        return _hsv_classify(crop)

