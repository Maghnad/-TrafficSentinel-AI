"""Asynchronous ANPR with Indian Plate Disambiguation and Bumper Focusing."""

from __future__ import annotations

import queue
import re
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

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


# Character disambiguation maps for Indian license plate syntax:
CHAR_TO_LETTER = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}
CHAR_TO_DIGIT = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}


def disambiguate_indian_plate(text: str) -> str:
    """Corrects typical OCR optical confusions based on Indian RTO registration rules."""
    t = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(t) < 7 or len(t) > 11:
        return t

    chars = list(t)
    n = len(chars)

    # 1. First 2 characters are ALWAYS state code letters (DL, MH, KA, WB, etc.)
    for i in (0, 1):
        if chars[i] in CHAR_TO_LETTER:
            chars[i] = CHAR_TO_LETTER[chars[i]]

    # 2. Position 2 is ALWAYS a digit (e.g. DL 3, KA 0)
    if chars[2] in CHAR_TO_DIGIT:
        chars[2] = CHAR_TO_DIGIT[chars[2]]

    # 3. Last 4 characters are ALWAYS registration digits (e.g. 6535, 1234)
    for i in range(max(3, n - 4), n):
        if chars[i] in CHAR_TO_DIGIT:
            chars[i] = CHAR_TO_DIGIT[chars[i]]

    return ''.join(chars)


class PlateReader:
    """Owns the EasyOCR reader and the pre/post-processing."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.regex = re.compile(cfg.plate_regex)
        self._reader = None

    def _lazy(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=self.cfg.gpu, verbose=False)
        return self._reader

    # ------------------------------------------------------------------ #

    @staticmethod
    def _prep(crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = crop.shape[:2]
        scale = min(3.0, max(1.0, 320.0 / max(1, w)))
        if scale > 1.05:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        crop = enhance_crop(crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 7, 55, 55)
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return denoised, otsu

    def _normalise(self, text: str) -> str:
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def read(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        if crop is None or crop.size == 0:
            return None, 0.0
        reader = self._lazy()
        gray, otsu = self._prep(crop)

        best_raw, best_conf = None, 0.0
        
        for image in (gray, otsu):
            try:
                results = reader.readtext(
                    image,
                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -',
                    detail=1,
                    paragraph=False
                )
            except Exception:
                continue
            if not results:
                continue

            results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
            raw_text = self._normalise(''.join(r[1] for r in results))
            conf = float(np.mean([r[2] for r in results])) if results else 0.0

            if not raw_text:
                continue

            corrected = disambiguate_indian_plate(raw_text)

            if self.regex.match(corrected):
                return corrected, max(conf, 0.85)
            if self.regex.match(raw_text):
                return raw_text, max(conf, 0.80)

            if conf > best_conf and len(corrected) >= 5:
                best_raw, best_conf = corrected, conf

            if conf > 0.70:
                break

        if best_raw and len(best_raw) >= 5:
            return best_raw, best_conf * 0.75
            
        return None, 0.0

    def is_valid(self, plate: Optional[str]) -> bool:
        return bool(plate and self.regex.match(plate))


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


def plate_crop(frame: np.ndarray, bbox, min_h: int) -> Optional[np.ndarray]:
    """Crop the bumper/plate-bearing region of a vehicle accurately without cutting off text."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if w < 30 or h < 30:
        return None

    # Focus on the lower 42% of the vehicle (where front/rear bumper and plates are positioned)
    y1_crop = max(0, int(y1 + 0.58 * h))
    y2_crop = min(h_img, int(y2 + 0.06 * h))
    x1_crop = max(0, int(x1 - 0.04 * w))
    x2_crop = min(w_img, int(x2 + 0.04 * w))

    if y2_crop <= y1_crop or x2_crop <= x1_crop:
        return None

    crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
    return crop if crop.size > 0 else None
