"""Asynchronous ANPR — adopts the exact logic from
automatic-number-plate-recognition-python-yolov8 (main.py + util.py).

Key pipeline (from reference project):
  1. Run YOLOv8 license_plate_detector on the FULL FRAME
  2. For each detected plate bbox, crop from frame
  3. Grayscale → cv2.threshold(gray, 64, 255, THRESH_BINARY_INV)
  4. EasyOCR readtext() on the thresholded crop
  5. Positional format_license() mapping (chars 0,1,4,5,6 = letters; 2,3 = digits)
  6. Assign plate to vehicle via get_car() containment check
"""

from __future__ import annotations

import queue
import re
import string
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .helmet import enhance_crop


@dataclass
class OCRJob:
    track_id: int
    crop: np.ndarray


@dataclass
class OCRResult:
    track_id: int
    plate: Optional[str]
    confidence: float


# ---------------------------------------------------------------------------
# Character mapping dictionaries — verbatim from reference util.py
# ---------------------------------------------------------------------------

dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}

# Extended mappings for Indian plates (additional confusions)
dict_char_to_int_ext = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1', 'T': '1',
    'Z': '2', 'J': '3', 'A': '4',
    'S': '5', 'G': '6', 'B': '8'
}

dict_int_to_char_ext = {
    '0': 'O', '1': 'I', '2': 'Z',
    '3': 'J', '4': 'A', '5': 'S',
    '6': 'G', '7': 'U', '8': 'B'
}


# ---------------------------------------------------------------------------
# Functions — verbatim from reference util.py
# ---------------------------------------------------------------------------

def license_complies_format(text: str) -> bool:
    """Check if the license plate text complies with the required format.
    Reference format: 7 chars — LL DD LLL (e.g. AB12CDE)
    """
    if len(text) != 7:
        return False

    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text: str) -> str:
    """Format the license plate text by converting characters using the mapping
    dictionaries — verbatim from reference util.py."""
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char,
               4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_


def read_license_plate_ref(reader, license_plate_crop) -> Tuple[Optional[str], float]:
    """Read the license plate text from the given cropped image — verbatim
    logic from reference util.py read_license_plate()."""
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(' ', '')

        if license_complies_format(text):
            return format_license(text), score

    return None, 0.0


def get_car(license_plate_bbox, vehicle_bboxes_with_ids):
    """Retrieve the vehicle coordinates and ID based on the license plate
    coordinates — verbatim from reference util.py."""
    x1, y1, x2, y2 = license_plate_bbox

    for vbox in vehicle_bboxes_with_ids:
        xcar1, ycar1, xcar2, ycar2, car_id = vbox
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id

    return -1, -1, -1, -1, -1


# ---------------------------------------------------------------------------
# Extended Indian plate disambiguation (handles 9-10 char Indian plates
# that the 7-char reference format doesn't cover)
# ---------------------------------------------------------------------------

def disambiguate_indian_plate(text: str) -> str:
    """Format and disambiguate license plate text for Indian RTO syntax."""
    t = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(t) < 6 or len(t) > 11:
        return t

    chars = list(t)
    n = len(chars)

    # 1. State code (first 2 always letters)
    for i in (0, 1):
        if chars[i] in dict_int_to_char_ext:
            chars[i] = dict_int_to_char_ext[chars[i]]

    # 2. Position 2 always a digit
    if chars[2] in dict_char_to_int_ext:
        chars[2] = dict_char_to_int_ext[chars[2]]

    # 3. Last 4 always digits
    for i in range(max(3, n - 4), n):
        if chars[i] in dict_char_to_int_ext:
            chars[i] = dict_char_to_int_ext[chars[i]]

    # 4. Standard 10-char (UP12AA7855): pos 3 = digit, pos 4,5 = letters
    if n == 10:
        if chars[3] in dict_char_to_int_ext:
            chars[3] = dict_char_to_int_ext[chars[3]]
        for i in (4, 5):
            if chars[i] in dict_int_to_char_ext:
                chars[i] = dict_int_to_char_ext[chars[i]]
    elif n == 9:
        for i in range(max(3, n - 4), n):
            if chars[i] in dict_char_to_int_ext:
                chars[i] = dict_char_to_int_ext[chars[i]]

    return ''.join(chars)


# ---------------------------------------------------------------------------
# PlateReader — uses the reference project's exact pipeline
# ---------------------------------------------------------------------------

class PlateReader:
    """YOLOv8 plate localizer + EasyOCR reader using the exact pipeline from
    the reference automatic-number-plate-recognition-python-yolov8 project."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.regex = re.compile(cfg.plate_regex)
        self._reader = None
        self._plate_detector = None
        self._init_detector()

    def _init_detector(self):
        model_path = getattr(self.cfg, "plate_model_path", "models/license_plate_detector.pt")
        from pathlib import Path
        if model_path and Path(model_path).exists():
            try:
                from ultralytics import YOLO
                self._plate_detector = YOLO(model_path)
                print(f"[anpr] loaded YOLOv8 license plate detector: {model_path}")
            except Exception as exc:
                print(f"[anpr] warning: could not load plate detector {model_path}: {exc}")

    def _lazy(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=self.cfg.gpu, verbose=False)
        return self._reader

    # ------------------------------------------------------------------ #
    # FULL-FRAME plate detection — the key logic from reference main.py
    # ------------------------------------------------------------------ #

    def detect_plates_in_frame(self, frame: np.ndarray) -> List[dict]:
        """Run YOLOv8 license_plate_detector on the FULL FRAME (exactly like
        reference main.py line: license_plates = license_plate_detector(frame)[0]).

        Returns list of dicts: {bbox: [x1,y1,x2,y2], bbox_score: float,
                                text: str|None, text_score: float}
        """
        if self._plate_detector is None or frame is None or frame.size == 0:
            return []

        try:
            conf_thresh = getattr(self.cfg, "plate_detector_conf", 0.20)
            results = self._plate_detector.predict(frame, conf=conf_thresh, verbose=False)
            if not results or len(results[0].boxes) == 0:
                return []
        except Exception:
            return []

        reader = self._lazy()
        plates = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].cpu().numpy()]
            score = float(box.conf)

            # Crop license plate from frame — exactly like reference main.py:
            # license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            h_img, w_img = frame.shape[:2]
            cx1 = max(0, x1)
            cy1 = max(0, y1)
            cx2 = min(w_img, x2)
            cy2 = min(h_img, y2)
            license_plate_crop = frame[cy1:cy2, cx1:cx2]

            if license_plate_crop.size == 0:
                continue

            # Process license plate — exactly like reference main.py:
            # license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(
                license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # Read license plate number — exactly like reference main.py:
            # license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
            license_plate_text, license_plate_text_score = read_license_plate_ref(
                reader, license_plate_crop_thresh)

            # If reference format didn't match, try extended Indian pipeline
            if license_plate_text is None:
                license_plate_text, license_plate_text_score = self._read_indian(
                    license_plate_crop)

            plates.append({
                "bbox": [cx1, cy1, cx2, cy2],
                "bbox_score": score,
                "text": license_plate_text,
                "text_score": license_plate_text_score or 0.0,
            })

        return plates

    # ------------------------------------------------------------------ #
    # Per-crop OCR fallback (for async worker when full-frame isn't available)
    # ------------------------------------------------------------------ #

    def isolate_plate(self, vehicle_crop: np.ndarray) -> np.ndarray:
        """Runs YOLOv8-plate on the vehicle crop to get a tight bounding box."""
        if self._plate_detector is None or vehicle_crop is None or vehicle_crop.size == 0:
            return vehicle_crop

        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 20:
            return vehicle_crop

        try:
            conf_thresh = getattr(self.cfg, "plate_detector_conf", 0.20)
            results = self._plate_detector.predict(vehicle_crop, imgsz=320,
                                                    conf=conf_thresh, verbose=False)
            if not results or len(results[0].boxes) == 0:
                return vehicle_crop

            boxes = results[0].boxes
            best_idx = int(boxes.conf.cpu().numpy().argmax())
            x1, y1, x2, y2 = [int(round(v)) for v in boxes.xyxy[best_idx].cpu().numpy()]

            pad_x = max(2, int((x2 - x1) * 0.06))
            pad_y = max(2, int((y2 - y1) * 0.08))
            px1, py1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            px2, py2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            if px2 > px1 and py2 > py1:
                plate_sub = vehicle_crop[py1:py2, px1:px2]
                if plate_sub.size > 0:
                    return plate_sub
        except Exception:
            pass
        return vehicle_crop

    def _read_indian(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        """Extended multi-pass OCR for Indian plates (9-10 chars)."""
        if crop is None or crop.size == 0:
            return None, 0.0

        reader = self._lazy()
        best_text, best_conf = None, 0.0

        # Prepare multiple image variants for OCR
        h, w = crop.shape[:2]
        target_h = max(80, min(140, int(h * 3.0)))
        scale = target_h / max(1, h)
        upscaled = cv2.resize(crop, (int(w * scale), target_h),
                               interpolation=cv2.INTER_CUBIC)
        upscaled = enhance_crop(upscaled)

        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 7, 55, 55)
        _, otsu = cv2.threshold(denoised, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bin_inv = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)

        for image in (upscaled, denoised, otsu, bin_inv):
            try:
                results = reader.readtext(
                    image,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                    detail=1, paragraph=False,
                    text_threshold=0.18, low_text=0.12)
            except Exception:
                continue
            if not results:
                continue

            # Sort top-to-bottom, left-to-right for multi-line plates
            results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
            raw = re.sub(r'[^A-Z0-9]', '',
                         ''.join(r[1] for r in results).upper())
            conf = float(np.mean([r[2] for r in results]))

            if not raw or len(raw) < 5:
                continue

            corrected = disambiguate_indian_plate(raw)

            if self.regex.match(corrected):
                return corrected, max(conf, 0.88)
            if self.regex.match(raw):
                return raw, max(conf, 0.82)

            if conf > best_conf and len(corrected) >= 6:
                best_text, best_conf = corrected, conf

        if best_text and len(best_text) >= 6:
            fixed = disambiguate_indian_plate(best_text)
            if self.regex.match(fixed):
                return fixed, max(best_conf, 0.82)
            return fixed, max(best_conf, 0.70)

        return None, 0.0

    def read(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        """Read plate from a vehicle crop (used by async OCR worker).
        Tries reference pipeline first, then extended Indian pipeline."""
        if crop is None or crop.size == 0:
            return None, 0.0

        # Step 1: Isolate plate region using YOLOv8
        plate_crop_img = self.isolate_plate(crop)

        # Step 2: Try reference pipeline (grayscale → THRESH_BINARY_INV → readtext)
        if plate_crop_img is not crop:
            gray = cv2.cvtColor(plate_crop_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 64, 255, cv2.THRESH_BINARY_INV)
            text, score = read_license_plate_ref(self._lazy(), thresh)
            if text is not None:
                return text, score

        # Step 3: Try extended Indian plate pipeline on isolated crop
        text, score = self._read_indian(plate_crop_img)
        if text is not None:
            return text, score

        # Step 4: If plate isolation failed, try on entire vehicle crop
        if plate_crop_img is not crop:
            text, score = self._read_indian(crop)
            if text is not None:
                return text, score

        return None, 0.0

    def is_valid(self, plate: Optional[str]) -> bool:
        return bool(plate and self.regex.match(plate))


# ---------------------------------------------------------------------------
# Async OCR worker thread
# ---------------------------------------------------------------------------

class OCRWorker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.reader = PlateReader(cfg)
        self._q: queue.Queue[OCRJob] = queue.Queue(maxsize=cfg.queue_size)
        self._results: Dict[int, OCRResult] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.processed = 0

    def start(self) -> OCRWorker:
        if not self.cfg.enabled:
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                plate, conf = self.reader.read(job.crop)
            except Exception:
                plate, conf = None, 0.0
            self.processed += 1
            with self._lock:
                prev = self._results.get(job.track_id)
                if plate and (prev is None or conf > prev.confidence):
                    self._results[job.track_id] = OCRResult(job.track_id, plate, conf)
                elif prev is None:
                    self._results[job.track_id] = OCRResult(job.track_id, None, 0.0)

    # ------------------------------------------------------------------ #

    def submit(self, track_id: int, crop: np.ndarray) -> bool:
        """Non-blocking. Drops the oldest job if the queue is saturated."""
        if not self.cfg.enabled or crop is None or crop.size == 0:
            return False
        job = OCRJob(track_id, crop.copy())
        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
            try:
                self._q.get_nowait()
                self._q.put_nowait(job)
                return True
            except Exception:
                return False

    def result(self, track_id: int) -> Optional[OCRResult]:
        with self._lock:
            return self._results.get(track_id)

    @property
    def backlog(self) -> int:
        return self._q.qsize()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Utility crop function used by engine.py
# ---------------------------------------------------------------------------

def plate_crop(frame: np.ndarray, bbox, min_h: int) -> Optional[np.ndarray]:
    """Crop the bumper/plate-bearing region of a vehicle for plate detection."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if w < 24 or h < 24:
        return None

    # For small vehicles/motorcycles (h < 90), use the full vehicle region
    if h < 90:
        y1_crop = max(0, int(y1 - 0.05 * h))
        y2_crop = min(h_img, int(y2 + 0.05 * h))
    else:
        # Focus on lower 55% where front/rear plates are located
        y1_crop = max(0, int(y1 + 0.45 * h))
        y2_crop = min(h_img, int(y2 + 0.08 * h))

    x1_crop = max(0, int(x1 - 0.06 * w))
    x2_crop = min(w_img, int(x2 + 0.06 * w))

    if y2_crop <= y1_crop or x2_crop <= x1_crop:
        return None

    crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
    return crop if crop.size > 0 else None
