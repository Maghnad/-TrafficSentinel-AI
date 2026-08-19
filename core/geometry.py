"""Ground-plane geometry.

The original plan used a single scalar `pixels_per_meter`. That is invalid
under perspective: a vehicle 10 m from the camera covers many more pixels per
metre than one at 60 m, so a scalar over-reports far vehicles and under-reports
near ones by factors of 3-5x. Since overspeeding carries a Rs 2000 CRITICAL
fine, that error is not acceptable.

Instead we compute a homography H mapping the road plane in image space to
metres in world space, from four points the operator clicks on the road
surface (a rectangle of known dimensions - lane markings work well). All
speed, wrong-way and near-miss maths then happens in metres.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


class GroundPlane:
    """Image (px) <-> road plane (metres) mapping."""

    def __init__(self, src: Sequence[Sequence[float]] | None,
                 dst: Sequence[Sequence[float]] | None):
        self.H: Optional[np.ndarray] = None
        if src and dst and len(src) == 4 and len(dst) == 4:
            s = np.asarray(src, dtype=np.float32)
            d = np.asarray(dst, dtype=np.float32)
            self.H = cv2.getPerspectiveTransform(s, d)

    @property
    def calibrated(self) -> bool:
        return self.H is not None

    def to_world(self, pt: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        if self.H is None:
            return None
        v = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
        w = cv2.perspectiveTransform(v, self.H)[0][0]
        return float(w[0]), float(w[1])

    def distance_m(self, p1, p2) -> Optional[float]:
        a, b = self.to_world(p1), self.to_world(p2)
        if a is None or b is None:
            return None
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# ---------------------------------------------------------------------- #
# Line crossing - the correct definition of a red-light violation.
# ---------------------------------------------------------------------- #

def _side(line, pt) -> float:
    (x1, y1), (x2, y2) = line
    return (x2 - x1) * (pt[1] - y1) - (y2 - y1) * (pt[0] - x1)


def crossed_line(line, prev_pt, curr_pt) -> bool:
    """True if the segment prev->curr crosses the stop line.

    A car legitimately *stopped at* a red light sits on one side of the line
    forever and never triggers. Only a vehicle whose ground-contact point
    passes from the approach side to the far side is in violation.
    """
    if not line or len(line) != 2:
        return False
    s1, s2 = _side(line, prev_pt), _side(line, curr_pt)
    if s1 == 0 or s2 == 0:
        return False
    if (s1 > 0) == (s2 > 0):
        return False
    # Also require the crossing point to fall within the drawn segment, so
    # vehicles in an adjacent lane beyond the line's extent are ignored.
    (x1, y1), (x2, y2) = line
    px, py = prev_pt
    qx, qy = curr_pt
    denom = (x1 - x2) * (py - qy) - (y1 - y2) * (px - qx)
    if abs(denom) < 1e-9:
        return False
    t = ((x1 - px) * (py - qy) - (y1 - py) * (px - qx)) / denom
    return 0.0 <= t <= 1.0


def point_in_polygon(polygon, pt) -> bool:
    if not polygon or len(polygon) < 3:
        return False
    contour = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def containment(inner, outer) -> float:
    """Fraction of `inner` box that lies inside `outer`. Unlike IoU this is not
    penalised by the size difference, which is what we want when asking
    'is this person inside this motorcycle's box'."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area = max(1e-6, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return inter / area


def ttc(p1_w, v1_w, p2_w, v2_w) -> Optional[float]:
    """Time-to-collision (seconds) between two point masses in world coords.
    Returns None if they are not closing."""
    rp = np.array(p2_w, dtype=float) - np.array(p1_w, dtype=float)
    rv = np.array(v2_w, dtype=float) - np.array(v1_w, dtype=float)
    closing = -float(np.dot(rp, rv))
    if closing <= 1e-6:
        return None
    speed_sq = float(np.dot(rv, rv))
    if speed_sq < 1e-6:
        return None
    return closing / speed_sq
