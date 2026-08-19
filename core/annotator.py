"""Frame annotation. Kept deliberately cheap - drawing is 3-5 ms and it is easy
to accidentally make it 30."""

from __future__ import annotations

import time
from typing import Dict, List

import cv2
import numpy as np

COLORS = {
    "violation": (60, 60, 235),
    "person": (235, 190, 60),
    "motorcycle": (120, 220, 120),
    "car": (200, 160, 90),
    "bus": (180, 120, 200),
    "truck": (120, 150, 220),
    "bicycle": (150, 220, 200),
    "traffic light": (90, 220, 235),
    "default": (180, 180, 180),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, org, color, scale=0.45):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = int(org[0]), int(org[1])
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + base - 1), color, -1)
    cv2.putText(img, text, (x + 3, y - 2), FONT, scale, (255, 255, 255), 1,
                cv2.LINE_AA)


class Annotator:
    def __init__(self, geometry, show_clean: bool = True):
        self.geo = geometry
        self.show_clean = show_clean

    def draw(self, frame, graph, violations, registry, hud: Dict) -> np.ndarray:
        img = frame  # caller passes a copy
        offenders = {}
        for v in violations:
            offenders.setdefault(v.track_id, []).append(v.vtype)

        self._draw_zones(img)

        for node in graph.nodes.values():
            det = node.det
            tid = det["track_id"]
            is_offender = tid in offenders
            if not is_offender and not self.show_clean:
                continue

            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            color = (COLORS["violation"] if is_offender
                     else COLORS.get(det["label"], COLORS["default"]))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2 if is_offender else 1)

            parts = [f"{det['label']}#{tid}"]
            tr = registry.get(tid)
            if tr:
                if tr.speed_kmh is not None:
                    parts.append(f"{tr.speed_kmh:.0f}km/h")
                if tr.plate:
                    parts.append(tr.plate)
            hs = node.attributes.get("helmet")
            if hs and hs != "UNKNOWN":
                parts.append(hs)
            state = node.attributes.get("state")
            if state and state != "UNKNOWN":
                parts.append(state)

            _label(img, " ".join(parts), (x1, y1 - 4), color)

            if is_offender:
                _label(img, ",".join(offenders[tid]), (x1, y2 + 16),
                       COLORS["violation"], 0.42)

        self._draw_hud(img, hud)
        return img

    def _draw_zones(self, img) -> None:
        if self.geo.stop_line and len(self.geo.stop_line) == 2:
            p1 = tuple(int(v) for v in self.geo.stop_line[0])
            p2 = tuple(int(v) for v in self.geo.stop_line[1])
            cv2.line(img, p1, p2, (0, 215, 255), 2)
            _label(img, "STOP LINE", p1, (0, 160, 200), 0.4)
        for poly in self.geo.no_parking_zones or []:
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (90, 90, 220), 2)

    def _draw_hud(self, img, hud: Dict) -> None:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 28), (28, 28, 32), -1)
        left = (f"{hud.get('camera_id','CAM')}  "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
        cv2.putText(img, left, (8, 19), FONT, 0.5, (235, 235, 235), 1,
                    cv2.LINE_AA)

        right = (f"FPS {hud.get('fps', 0):.1f} | "
                 f"infer {hud.get('infer_ms', 0):.0f}ms | "
                 f"OCR q{hud.get('ocr_backlog', 0)} | "
                 f"signal {hud.get('light', 'UNKNOWN')} | "
                 f"tracks {hud.get('tracks', 0)}")
        (tw, _), _ = cv2.getTextSize(right, FONT, 0.45, 1)
        cv2.putText(img, right, (max(8, w - tw - 8), 19), FONT, 0.45,
                    (180, 220, 180), 1, cv2.LINE_AA)

        if not hud.get("calibrated", False):
            _label(img, "UNCALIBRATED - speed/wrong-way/near-miss disabled",
                   (8, h - 10), (40, 120, 200), 0.5)
