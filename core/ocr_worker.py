"""Asynchronous ANPR -- India-Specific Lexicon-Constrained Pipeline.

Key improvements over the original positional character repair:
  1. BH (Bharat Series) branch -- prevents corruption of YY BH #### XX plates.
  2. State code lexicon scoring -- joint inference over positions 1-2 against the
     37 valid state/UT codes, instead of independent per-character substitution.
  3. Geographic prior -- WB-weighted scoring for Kolkata-belt deployments.
  4. Plate colour classification -- HSV thresholding for vehicle class.
  5. Two-line plate handling -- aspect ratio classification + horizontal projection.
  6. NON_HSRP detection -- plates that fail OCR after max retries are logged as
     CMVR Rule 50 violations instead of silent drops.
"""

from __future__ import annotations

import queue
import re
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from .helmet import enhance_crop


# ====================================================================== #
# Data Classes
# ====================================================================== #

@dataclass
class OCRJob:
    track_id: int
    crop: np.ndarray


@dataclass
class OCRResult:
    track_id: int
    plate: Optional[str]
    confidence: float
    plate_colour: str = "UNKNOWN"       # WHITE / YELLOW / GREEN / BLACK / RED
    vehicle_class: str = "UNKNOWN"      # PRIVATE / COMMERCIAL / EV / RENTAL / TEMP
    is_hsrp: bool = True                # False if hand-painted / non-standard
    format_type: str = "STANDARD"       # STANDARD / BH


# ====================================================================== #
# Indian State/UT Code Lexicon (37 valid codes + BH)
# ====================================================================== #

VALID_STATE_CODES: Set[str] = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "GA",
    "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TR", "TS", "UK", "UP", "WB",
}

# Geographic prior: probability weights for Kolkata-belt deployment.
# WB dominates, then neighbouring states on interstate corridors.
GEO_PRIOR: Dict[str, float] = {
    "WB": 5.0,    # West Bengal -- home state
    "BR": 2.0,    # Bihar -- adjacent
    "JH": 2.0,    # Jharkhand -- adjacent
    "OD": 1.8,    # Odisha -- adjacent
    "AS": 1.5,    # Assam -- NE corridor
    "UP": 1.3,    # Uttar Pradesh -- national highway traffic
    "DL": 1.2,    # Delhi -- commercial/interstate
    "MH": 1.1,    # Maharashtra -- commercial
    "BH": 1.5,    # Bharat series -- urban newer vehicles
}
DEFAULT_GEO_WEIGHT = 0.5  # All other states


# Character confusion maps (bidirectional optical similarities)
CHAR_CONFUSIONS: Dict[str, str] = {
    '0': 'O', 'O': '0', 'D': '0', 'Q': '0',
    '1': 'I', 'I': '1', 'L': '1',
    '2': 'Z', 'Z': '2',
    '5': 'S', 'S': '5',
    '8': 'B', 'B': '8',
    '6': 'G', 'G': '6',
}


def _char_similarity(a: str, b: str) -> float:
    """Score how likely OCR would confuse character a with character b."""
    if a == b:
        return 1.0
    if CHAR_CONFUSIONS.get(a) == b or CHAR_CONFUSIONS.get(b) == a:
        return 0.7
    return 0.0


def _score_state_code(c1: str, c2: str, geo_prior: Dict[str, float]) -> Tuple[str, float]:
    """Score the two-character read against all valid state codes using joint
    character similarity and geographic prior. Returns (best_code, score)."""
    best_code, best_score = c1 + c2, 0.0

    for code in VALID_STATE_CODES:
        sim = _char_similarity(c1, code[0]) * _char_similarity(c2, code[1])
        if sim <= 0.0:
            continue
        geo = geo_prior.get(code, DEFAULT_GEO_WEIGHT)
        score = sim * geo
        if score > best_score:
            best_score = score
            best_code = code

    return best_code, best_score


# ====================================================================== #
# BH (Bharat Series) Detection and Validation
# ====================================================================== #

# BH format: YY BH #### XX  (year, BH, 4 digits, 1-2 letters excluding I and O)
BH_REGEX = re.compile(r'^[0-9]{2}BH[0-9]{4}[A-HJ-NP-Z]{1,2}$')

# BH trailing letters explicitly exclude I and O to avoid 1/0 confusion
BH_FORBIDDEN_TRAILING = {'I', 'O'}


def _is_bh_candidate(text: str) -> bool:
    """Check if positions 3-4 read as BH (before any positional repair)."""
    if len(text) < 8:
        return False
    return text[2:4] in ('BH', '8H', 'B4', '84', 'BN', '8N')


def _fix_bh_plate(text: str) -> str:
    """Apply BH-specific disambiguation rules."""
    chars = list(text)
    n = len(chars)
    if n < 8:
        return text

    c2d = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1',
           'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
    d2l = {'0': 'D', '1': 'J', '5': 'S', '8': 'B'}

    # Positions 0-1: registration year digits
    for i in (0, 1):
        if chars[i] in c2d:
            chars[i] = c2d[chars[i]]

    # Positions 2-3: must be "BH"
    chars[2] = 'B'
    chars[3] = 'H'

    # Positions 4-7: four digits
    for i in range(4, min(8, n)):
        if chars[i] in c2d:
            chars[i] = c2d[chars[i]]

    # Positions 8+: trailing letters (I and O forbidden by regulation)
    for i in range(8, n):
        if chars[i] == 'I':
            chars[i] = 'J'
        elif chars[i] == 'O':
            chars[i] = 'D'
        elif chars[i] in d2l:
            chars[i] = d2l[chars[i]]

    return ''.join(chars)


# ====================================================================== #
# Standard Plate Disambiguation (with Lexicon Scoring)
# ====================================================================== #

def disambiguate_indian_plate(text: str, geo_prior: Optional[Dict[str, float]] = None) -> str:
    """Lexicon-constrained disambiguation for Indian license plates.

    Handles both standard state-series (XX ## XXXX) and Bharat series (## BH #### XX).
    Uses joint state code scoring instead of independent per-character substitution.
    """
    t = re.sub(r'[^A-Z0-9]', '', text.upper())
    if len(t) < 7 or len(t) > 11:
        return t

    prior = geo_prior or GEO_PRIOR

    # Branch: detect BH plates BEFORE applying state-code rules
    if _is_bh_candidate(t):
        fixed = _fix_bh_plate(t)
        if BH_REGEX.match(fixed):
            return fixed

    # Standard state-series plate
    chars = list(t)
    n = len(chars)

    # 1. Joint state code scoring (positions 0-1)
    best_code, score = _score_state_code(chars[0], chars[1], prior)
    if score > 0.0:
        chars[0], chars[1] = best_code[0], best_code[1]

    # 2. District code at position 2 (always a digit)
    c2d = {'O': '0', 'D': '0', 'Q': '0', 'I': '1', 'L': '1',
           'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
    if chars[2] in c2d:
        chars[2] = c2d[chars[2]]

    # 3. Last 4 characters are registration digits
    for i in range(max(3, n - 4), n):
        if chars[i] in c2d:
            chars[i] = c2d[chars[i]]

    return ''.join(chars)


# ====================================================================== #
# Plate Colour Classification (HSV Thresholding)
# ====================================================================== #

def classify_plate_colour(crop: np.ndarray) -> Tuple[str, str]:
    """Classify plate background colour using HSV thresholding.

    Returns (colour, vehicle_class):
        WHITE  -> PRIVATE
        YELLOW -> COMMERCIAL
        GREEN  -> EV
        BLACK  -> RENTAL
        RED    -> TEMPORARY
    """
    if crop is None or crop.size == 0:
        return "UNKNOWN", "UNKNOWN"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    total = max(1, h.size)

    white_ratio = np.count_nonzero((s < 60) & (v > 160)) / total
    yellow_ratio = np.count_nonzero((h >= 15) & (h <= 35) & (s > 80) & (v > 120)) / total
    green_ratio = np.count_nonzero((h >= 35) & (h <= 85) & (s > 50) & (v > 80)) / total
    black_ratio = np.count_nonzero(v < 60) / total
    red_ratio = np.count_nonzero(((h < 10) | (h > 170)) & (s > 80) & (v > 80)) / total

    ratios = {"WHITE": white_ratio, "YELLOW": yellow_ratio, "GREEN": green_ratio,
              "BLACK": black_ratio, "RED": red_ratio}
    colour = max(ratios, key=ratios.get)
    if ratios[colour] < 0.15:
        colour = "UNKNOWN"

    c2c = {"WHITE": "PRIVATE", "YELLOW": "COMMERCIAL", "GREEN": "EV",
           "BLACK": "RENTAL", "RED": "TEMPORARY", "UNKNOWN": "UNKNOWN"}
    return colour, c2c[colour]


# ====================================================================== #
# Two-Line Plate Splitter
# ====================================================================== #

def _split_two_line(crop: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Split a two-line plate into top and bottom halves using horizontal projection.

    Two-line plates (common on two-wheelers, ~95% of bikes) have aspect ratio ~2:1
    vs single-line ~4.5:1. The gap between lines appears as a valley in the
    horizontal projection of edge density.
    """
    h, w = crop.shape[:2]
    if w / max(1, h) > 3.0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    edges = cv2.Canny(gray, 50, 150)
    proj = np.sum(edges, axis=1).astype(float)

    s, e = int(h * 0.3), int(h * 0.7)
    if e <= s + 2:
        return None
    region = proj[s:e]
    if region.size == 0:
        return None

    gap = s + int(np.argmin(region))
    if proj[gap] > 0.3 * np.max(proj):
        return None

    top, bottom = crop[:gap, :], crop[gap:, :]
    if top.size == 0 or bottom.size == 0:
        return None
    return top, bottom


# ====================================================================== #
# Plate Reader (Core OCR Engine)
# ====================================================================== #

class PlateReader:
    """Owns the EasyOCR reader and the India-specific pre/post-processing."""

    STANDARD_REGEX = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$')
    BH_REGEX_PAT = BH_REGEX

    def __init__(self, cfg):
        self.cfg = cfg
        self.regex = re.compile(cfg.plate_regex)
        self._reader = None

    def _lazy(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"], gpu=self.cfg.gpu, verbose=False)
        return self._reader

    @staticmethod
    def _prep(crop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = crop.shape[:2]
        scale = min(3.0, max(1.0, 320.0 / max(1, w)))
        if scale > 1.05:
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_CUBIC)
        crop = enhance_crop(crop)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 7, 55, 55)
        _, otsu = cv2.threshold(denoised, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return denoised, otsu

    def _normalise(self, text: str) -> str:
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def _ocr_image(self, image: np.ndarray) -> Tuple[str, float]:
        """Run EasyOCR with beam search on a single image."""
        reader = self._lazy()
        try:
            results = reader.readtext(
                image, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -',
                detail=1, paragraph=False, decoder='beamsearch', beamWidth=5)
        except Exception:
            return "", 0.0
        if not results:
            return "", 0.0
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
        raw = self._normalise(''.join(r[1] for r in results))
        conf = float(np.mean([r[2] for r in results]))
        return raw, conf

    def read(self, crop: np.ndarray) -> Tuple[Optional[str], float, str, str, bool]:
        """Read a license plate from a bumper crop.

        Returns (plate_text, confidence, plate_colour, vehicle_class, is_hsrp).
        """
        if crop is None or crop.size == 0:
            return None, 0.0, "UNKNOWN", "UNKNOWN", True

        # Plate colour classification
        plate_colour, vehicle_class = classify_plate_colour(crop)

        # Preprocess
        gray, otsu = self._prep(crop)

        # Try two-line split first (for two-wheelers)
        split = _split_two_line(crop)
        best_text, best_conf = None, 0.0

        # If two-line split succeeded, OCR each half and concatenate
        if split is not None:
            top_crop, bot_crop = split
            parts, total_conf = [], 0.0
            for part in (top_crop, bot_crop):
                ph, pw = part.shape[:2]
                if pw < 10 or ph < 5:
                    continue
                p_gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY) \
                    if len(part.shape) == 3 else part
                p_dn = cv2.bilateralFilter(p_gray, 7, 55, 55)
                t, c = self._ocr_image(p_dn)
                if t:
                    parts.append(t)
                    total_conf += c
            if parts:
                joined = ''.join(parts)
                avg_conf = total_conf / len(parts)
                if len(joined) >= 5 and avg_conf > best_conf:
                    best_text, best_conf = joined, avg_conf

        # Standard single-pass OCR (two passes: bilateral + Otsu)
        for image in (gray, otsu):
            raw, conf = self._ocr_image(image)
            if not raw:
                continue
            if conf > best_conf and len(raw) >= 5:
                best_text, best_conf = raw, conf
            if conf > 0.70:
                break

        if not best_text:
            return None, 0.0, plate_colour, vehicle_class, True

        # Apply India-specific disambiguation
        # Constraint: yellow plate can never be BH (commercial not eligible)
        allow_bh = (plate_colour != "YELLOW")
        corrected = best_text

        if allow_bh and _is_bh_candidate(best_text):
            corrected = _fix_bh_plate(best_text)
            if self.BH_REGEX_PAT.match(corrected):
                return corrected, max(best_conf, 0.85), plate_colour, vehicle_class, True
        else:
            corrected = disambiguate_indian_plate(best_text, GEO_PRIOR)

        if self.regex.match(corrected) or self.STANDARD_REGEX.match(corrected):
            return corrected, max(best_conf, 0.85), plate_colour, vehicle_class, True
        if self.BH_REGEX_PAT.match(corrected):
            return corrected, max(best_conf, 0.85), plate_colour, vehicle_class, True
        if self.regex.match(best_text):
            return best_text, max(best_conf, 0.80), plate_colour, vehicle_class, True

        if len(corrected) >= 5:
            return corrected, best_conf * 0.75, plate_colour, vehicle_class, True
        return None, 0.0, plate_colour, vehicle_class, True

    def is_valid(self, plate: Optional[str]) -> bool:
        if not plate:
            return False
        return bool(self.regex.match(plate) or
                    self.STANDARD_REGEX.match(plate) or
                    self.BH_REGEX_PAT.match(plate))


# ====================================================================== #
# Async OCR Worker (same architecture, enhanced results)
# ====================================================================== #

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
        self._failed_tracks: Dict[int, int] = {}  # track_id -> consecutive failures

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
                plate, conf, colour, vclass, is_hsrp = self.reader.read(job.crop)
            except Exception:
                plate, conf, colour, vclass, is_hsrp = None, 0.0, "UNKNOWN", "UNKNOWN", True

            self.processed += 1
            with self._lock:
                prev = self._results.get(job.track_id)
                if plate and (prev is None or conf > prev.confidence):
                    self._results[job.track_id] = OCRResult(
                        job.track_id, plate, conf,
                        plate_colour=colour, vehicle_class=vclass,
                        is_hsrp=is_hsrp,
                        format_type="BH" if BH_REGEX.match(plate or "") else "STANDARD"
                    )
                    self._failed_tracks.pop(job.track_id, None)
                elif prev is None:
                    self._results[job.track_id] = OCRResult(
                        job.track_id, None, 0.0,
                        plate_colour=colour, vehicle_class=vclass,
                        is_hsrp=is_hsrp
                    )
                    self._failed_tracks[job.track_id] = self._failed_tracks.get(job.track_id, 0) + 1

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

    def is_non_hsrp_candidate(self, track_id: int) -> bool:
        """True if the track has failed OCR enough times to suggest a
        non-HSRP / hand-painted plate (itself a CMVR Rule 50 violation)."""
        with self._lock:
            return self._failed_tracks.get(track_id, 0) >= self.cfg.max_attempts_per_track

    @property
    def backlog(self) -> int:
        return self._q.qsize()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)


# ====================================================================== #
# Plate Crop Extraction
# ====================================================================== #

def plate_crop(frame: np.ndarray, bbox, min_h: int) -> Optional[np.ndarray]:
    """Crop the bumper/plate-bearing region of a vehicle."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    if w < 30 or h < 30:
        return None

    # Focus on the lower 42% of the vehicle body
    y1_crop = max(0, int(y1 + 0.58 * h))
    y2_crop = min(h_img, int(y2 + 0.06 * h))
    x1_crop = max(0, int(x1 - 0.04 * w))
    x2_crop = min(w_img, int(x2 + 0.04 * w))

    if y2_crop <= y1_crop or x2_crop <= x1_crop:
        return None

    crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
    return crop if crop.size > 0 else None
