"""
TrafficSentinel AI — Image Annotator
Simple clean boxes with colored text labels — no backgrounds.
Style: straight-edge rectangles, thin lines, label above top-left corner.
"""

import cv2
import numpy as np
from datetime import datetime


# ── Color Palette (BGR) ──────────────────────────────────────────────────

COLORS = {
    # Vehicles — each type gets a distinct color
    "car": (0, 255, 0),            # Green
    "motorcycle": (255, 178, 0),   # Cyan-ish
    "bus": (178, 102, 255),        # Purple
    "truck": (102, 255, 178),      # Teal
    "bicycle": (0, 255, 255),      # Yellow
    # Person — cycle through these
    "person_colors": [
        (0, 255, 0),       # Green
        (255, 0, 255),     # Pink / Magenta
        (255, 255, 0),     # Cyan
        (0, 128, 255),     # Orange
        (255, 0, 0),       # Blue
        (0, 255, 255),     # Yellow
    ],
    # Traffic light
    "traffic_light": (0, 0, 255),  # Red
    # Violations
    "HELMET_VIOLATION": (0, 0, 255),
    "TRIPLE_RIDING": (0, 80, 255),
    "RED_LIGHT_VIOLATION": (0, 0, 200),
    "NO_PLATE_VISIBLE": (0, 165, 255),
    "OVERCROWDING": (0, 100, 255),
    # Helmet status
    "HELMET": (0, 200, 0),
    "NO_HELMET": (0, 0, 255),
    "UNCERTAIN": (0, 200, 255),
}

# Counter for cycling person colors
_person_color_idx = 0


def _next_person_color():
    global _person_color_idx
    colors = COLORS["person_colors"]
    c = colors[_person_color_idx % len(colors)]
    _person_color_idx += 1
    return c


class ViolationAnnotator:
    """Simple clean annotations — thin boxes, colored text, no backgrounds."""

    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def annotate_detections(self, image: np.ndarray, pipeline_results: dict,
                             violations: list = None, show_all: bool = False) -> np.ndarray:
        """Draw boxes with violation-aware labels.
        Cross-references each detection with violations to show what rule it's breaking.
        """
        annotated = image.copy()
        global _person_color_idx
        _person_color_idx = 0  # Reset per frame

        # Build a lookup: node_id → list of violation types
        vio_by_node = {}
        if violations:
            for v in violations:
                for nid in v.involved_nodes:
                    vio_by_node.setdefault(nid, []).append(v)

        for det_id, det in enumerate(pipeline_results["all_detections"]):
            bbox = det["bbox"]
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            category = det.get("category", "other")
            if category == "other":
                continue

            # Check if this detection is involved in any violation
            node_violations = vio_by_node.get(det_id, [])
            has_violation = len(node_violations) > 0

            # OPTIMIZATION: Hide innocent vehicles to reduce UI congestion
            if not has_violation and not show_all:
                continue

            if category == "vehicle":
                vtype = det.get("vehicle_type", "vehicle")
                if has_violation:
                    # Show violation type on the label
                    vio_names = [v.violation_type.replace("_", " ") for v in node_violations]
                    color = (0, 0, 255)  # Red for violations
                    label = f"{vtype} | {' | '.join(vio_names)}"
                else:
                    color = COLORS.get(vtype, (0, 255, 0))
                    label = f"{vtype} {det['confidence']:.0%}"

            elif category == "person":
                helmet = det.get("helmet_status", "")
                if helmet == "NO_HELMET" or has_violation:
                    color = COLORS["NO_HELMET"]  # Red
                    if has_violation:
                        vio_names = [v.violation_type.replace("_", " ") for v in node_violations]
                        label = f"person | {' | '.join(vio_names)}"
                    else:
                        label = f"person [NO HELMET] {det['confidence']:.0%}"
                elif helmet == "HELMET":
                    color = COLORS["HELMET"]  # Green
                    label = f"person [HELMET] {det['confidence']:.0%}"
                else:
                    color = _next_person_color()
                    label = f"person {det['confidence']:.0%}"

            elif category == "traffic_light":
                tl_color = det.get("color", "UNKNOWN")
                if tl_color == "RED":
                    color = (0, 0, 255)
                elif tl_color == "GREEN":
                    color = (0, 200, 0)
                else:
                    color = (0, 200, 255)
                label = f"signal: {tl_color}"
            else:
                continue

            # Type-cast for OpenCV 4.11.0 strict typing
            color_tuple = (int(color[0]), int(color[1]), int(color[2]))
            
            # Simple straight rectangle using uniform integer literal for thickness
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color_tuple, 1)

            # Colored text above top-left — no background
            cv2.putText(annotated, label, (int(x1), int(y1) - 5),
                        self.font, 0.4, color_tuple, 1, cv2.LINE_AA)

        return annotated

    def annotate_violations(self, image: np.ndarray, violations: list) -> np.ndarray:
        """Draw violation boxes with colored text label."""
        annotated = image.copy()

        for v in violations:
            bbox = v.evidence_bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            color = COLORS.get(v.violation_type, (0, 0, 255))

            # Simple rectangle for violation
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

            # Descriptive violation text — no background
            short_type = v.violation_type.replace("_", " ")
            label = f"{short_type} | Conf:{v.confidence:.0%} | Fine: Rs.{v.fine_amount}"
            cv2.putText(annotated, label, (x1, y1 - 5),
                        self.font, 0.45, color, 1, cv2.LINE_AA)

        return annotated

    def annotate_plates(self, image: np.ndarray, plates: list) -> np.ndarray:
        """Draw plate text below vehicle box — colored, no background."""
        annotated = image.copy()

        for plate in plates:
            text = plate["text"]
            conf = plate["confidence"]
            vbbox = plate["vehicle_bbox"]
            x1, y1, x2, y2 = int(vbbox[0]), int(vbbox[1]), int(vbbox[2]), int(vbbox[3])

            label = f"Plate: {text} {conf:.0%}"
            cv2.putText(annotated, label, (x1, y2 + 28),
                        self.font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def add_watermark(self, image: np.ndarray, camera_id: str = "CAM-001") -> np.ndarray:
        """Small watermark text at bottom — no background."""
        annotated = image.copy()
        h, w = annotated.shape[:2]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
        watermark = f"TrafficSentinel AI | {camera_id} | {timestamp}"

        cv2.putText(annotated, watermark, (4, h - 6),
                    self.font, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

        return annotated

    def create_full_annotation(
        self,
        image: np.ndarray,
        pipeline_results: dict,
        violations: list,
        camera_id: str = "CAM-001",
        show_all: bool = False
    ) -> np.ndarray:
        """Create fully annotated evidence image."""
        annotated = self.annotate_detections(image, pipeline_results, violations, show_all)
        annotated = self.annotate_violations(annotated, violations)
        annotated = self.annotate_plates(annotated, pipeline_results.get("plates", []))
        annotated = self.add_watermark(annotated, camera_id)
        return annotated
