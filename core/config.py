"""Central configuration. Everything tunable lives here, nothing is hardcoded
in the hot loop. Per-camera geometry is loaded from a calibration JSON produced
by calibrate.py."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class DetectorConfig:
    weights: str = "yolov8n.pt"
    # 'auto' picks openvino on Intel CPU, tensorrt on Jetson, else pytorch.
    backend: str = "auto"
    imgsz: int = 480            # 640 -> 480 is ~1.7x faster, minimal recall loss
    conf: float = 0.35
    iou: float = 0.5
    device: str = "cpu"
    half: bool = False          # only meaningful on CUDA
    tracker: str = "bytetrack.yaml"
    # Only run inference on every Nth decoded frame. Tracks are interpolated
    # between, so IDs stay stable.
    frame_stride: int = 1


@dataclass
class OCRConfig:
    enabled: bool = True
    gpu: bool = False
    # Hard cap on OCR attempts per track. Once we have a regex-valid plate we
    # stop entirely for that vehicle.
    max_attempts_per_track: int = 12
    min_crop_height: int = 40   # skip vehicles too small for OCR to ever work
    queue_size: int = 8         # drop-oldest beyond this; never blocks the loop
    plate_regex: str = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$"


@dataclass
class HelmetConfig:
    # Path to a fine-tuned YOLOv8 classification model (helmet / no_helmet).
    # If absent, the system reports UNCERTAIN and routes to human review
    # instead of auto-issuing a challan. See README for the training script.
    model_path: Optional[str] = "models/helmet_cls.pt"
    conf_auto_issue: float = 0.80
    head_crop_ratio: float = 0.30
    min_head_px: int = 18


@dataclass
class RuleConfig:
    speed_limit_kmh: float = 60.0
    speed_min_track_frames: int = 6
    triple_riding_min: int = 3
    wrongway_min_displacement_m: float = 4.0
    wrongway_dot_threshold: float = -0.55
    nearmiss_ttc_s: float = 1.5
    nearmiss_min_speed_kmh: float = 12.0
    # A violation is auto-issued only above this composite confidence.
    # Below it, the record is written with status='review'.
    auto_issue_confidence: float = 0.65  # lowered so HSV helmet fallback (cap 0.72) can auto-surface


@dataclass
class EvidenceConfig:
    root: str = "evidence"
    clip_seconds_before: float = 2.0
    clip_seconds_after: float = 2.0
    save_clips: bool = True
    crop_margin_px: int = 24
    jpeg_quality: int = 88


@dataclass
class CameraGeometry:
    """Per-camera spatial calibration. Produced by calibrate.py."""
    camera_id: str = "CAM-01"
    latitude: float = 22.5726
    longitude: float = 88.3639
    # 4 image points (px) -> 4 world points (metres) on the road plane.
    homography_src: list = field(default_factory=list)
    homography_dst: list = field(default_factory=list)
    # [[x1,y1],[x2,y2]] - crossing this while the light is red = violation.
    stop_line: list = field(default_factory=list)
    # Unit vector of legal travel direction in world coords.
    lane_direction: list = field(default_factory=lambda: [0.0, -1.0])
    # List of polygons (each a list of [x,y] image points).
    no_parking_zones: list = field(default_factory=list)
    parking_dwell_seconds: float = 20.0


@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    helmet: HelmetConfig = field(default_factory=HelmetConfig)
    rules: RuleConfig = field(default_factory=RuleConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    geometry: CameraGeometry = field(default_factory=CameraGeometry)
    db_path: str = "trafficsentinel.db"

    @staticmethod
    def load(path: str | Path) -> "AppConfig":
        path = Path(path)
        if not path.exists():
            return AppConfig()
        raw = json.loads(path.read_text(encoding='utf-8-sig'))
        cfg = AppConfig()
        for key, sub in raw.items():
            if hasattr(cfg, key) and isinstance(sub, dict):
                target = getattr(cfg, key)
                for k, v in sub.items():
                    if hasattr(target, k):
                        setattr(target, k, v)
            elif hasattr(cfg, key):
                setattr(cfg, key, sub)
        return cfg

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


# Fine amounts (INR) and severity, kept separate from logic so they can be
# edited without touching the rule engine.
VIOLATION_META = {
    "NO_HELMET":      {"fine": 1000, "severity": "HIGH"},
    "TRIPLE_RIDING":  {"fine": 1000, "severity": "HIGH"},
    "RED_LIGHT":      {"fine": 5000, "severity": "CRITICAL"},
    "OVERSPEEDING":   {"fine": 2000, "severity": "CRITICAL"},
    "WRONG_WAY":      {"fine": 5000, "severity": "CRITICAL"},
    "ILLEGAL_PARKING": {"fine": 500, "severity": "LOW"},
    "NEAR_MISS":      {"fine": 0,    "severity": "INFO"},
}


