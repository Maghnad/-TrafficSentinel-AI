"""
TrafficSentinel AI — Detection Pipeline
Uses pretrained YOLOv8 (COCO) for vehicle/person detection and EasyOCR for plate reading.
No custom training required — all models work out-of-the-box.
"""

import cv2
import numpy as np
from pathlib import Path

# Lazy-loaded to avoid slow import at module level
_yolo_model = None
_ocr_reader = None


def get_yolo_model():
    """Lazy-load YOLOv8 model (Nano model for max speed)."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        # Using yolov8n.pt (Nano) instead of yolov8l.pt (Large) for 10x-15x faster FPS
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def get_ocr_reader():
    """Lazy-load EasyOCR reader (downloads ~150MB on first run)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


# ── COCO class IDs relevant to traffic ──────────────────────────────────
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    1: "bicycle",
}
PERSON_CLASS_ID = 0
TRAFFIC_LIGHT_CLASS_ID = 9


class TrafficPipeline:
    """
    End-to-end detection pipeline:
      Image → Preprocess → YOLOv8 Detect → OCR Plates → Structured Results
    """

    def __init__(self):
        self.model = get_yolo_model()
        self.ocr = get_ocr_reader()

    # ── Stage 1: Preprocessing ───────────────────────────────────────────

    @staticmethod
    def preprocess(image: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement (works well for low-light)."""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced

    # ── Stage 2: Object Detection ────────────────────────────────────────

    def detect_objects(self, image: np.ndarray, conf_threshold: float = 0.15):
        """Run YOLOv8 on image. Returns list of detection dicts.
        Lower threshold (0.15) catches smaller/distant objects better."""
        results = self.model(image, conf=conf_threshold, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                det = {
                    "bbox": box.xyxy[0].cpu().numpy().astype(int),
                    "class_id": class_id,
                    "class_name": r.names[class_id],
                    "confidence": round(float(box.conf), 3),
                }

                # Tag category for easier downstream processing
                if class_id in VEHICLE_CLASSES:
                    det["category"] = "vehicle"
                    det["vehicle_type"] = VEHICLE_CLASSES[class_id]
                elif class_id == PERSON_CLASS_ID:
                    det["category"] = "person"
                elif class_id == TRAFFIC_LIGHT_CLASS_ID:
                    det["category"] = "traffic_light"
                else:
                    det["category"] = "other"

                detections.append(det)

        return detections

    # ── Stage 3: Traffic Light Color Detection ───────────────────────────

    @staticmethod
    def detect_traffic_light_color(image: np.ndarray, bbox) -> str:
        """
        Determine traffic light color by analyzing brightness of
        red / yellow / green regions within the bounding box.
        """
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return "UNKNOWN"

        h = y2 - y1
        third = max(h // 3, 1)

        red_region = crop[:third, :]
        yellow_region = crop[third : 2 * third, :]
        green_region = crop[2 * third :, :]

        def region_brightness(region):
            if region.size == 0:
                return 0
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            return float(np.mean(gray))

        scores = {
            "RED": region_brightness(red_region),
            "YELLOW": region_brightness(yellow_region),
            "GREEN": region_brightness(green_region),
        }
        return max(scores, key=scores.get)

    # ── Stage 4: Helmet Heuristic Check ──────────────────────────────────

    @staticmethod
    def check_helmet(image: np.ndarray, person_bbox) -> tuple:
        """
        Heuristic helmet detection on the head region of a person.
        Returns (status, confidence):
          status: 'HELMET' | 'NO_HELMET' | 'UNCERTAIN'
        """
        x1, y1, x2, y2 = person_bbox
        head_h = max(int((y2 - y1) * 0.28), 1)
        head_crop = image[y1 : y1 + head_h, x1:x2]

        if head_crop.size == 0 or head_crop.shape[0] < 5 or head_crop.shape[1] < 5:
            return "UNCERTAIN", 0.40

        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)

        # Skin-tone detection in HSV space
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([25, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = float(np.sum(skin_mask > 0)) / max(skin_mask.size, 1)

        # Color uniformity — helmets tend to be more uniform than hair/face
        saturation = hsv[:, :, 1]
        sat_std = float(np.std(saturation))

        # Decision logic
        if skin_ratio > 0.25:
            # High skin → likely no helmet (lowered from 0.35 for better sensitivity)
            conf = min(0.55 + skin_ratio * 0.45, 0.88)
            return "NO_HELMET", round(conf, 2)
        elif sat_std < 40 and skin_ratio < 0.12:
            # Uniform color, low skin → likely helmet (slightly more conservative)
            conf = min(0.55 + (40 - sat_std) / 60, 0.85)
            return "HELMET", round(conf, 2)
        else:
            return "UNCERTAIN", 0.45

    # ── Stage 5: License Plate Reading ───────────────────────────────────

    def read_plates(self, image: np.ndarray, vehicle_detections: list) -> list:
        """Extract license plate text from vehicle regions using EasyOCR.
        Tries multiple crop regions to maximize plate detection."""
        plates = []
        found_vehicles = set()  # Track which vehicles already have a plate

        for det in vehicle_detections:
            x1, y1, x2, y2 = det["bbox"]
            
            # Smart Expansion: YOLOv8 often cuts off the bottom of motorcycles/cars where the plate is.
            # We artificially expand the bottom of the bounding box by 30% to guarantee the plate is included.
            veh_h = y2 - y1
            y2_expanded = min(image.shape[0], int(y2 + veh_h * 0.30))
            veh_h_expanded = y2_expanded - y1

            det_key = f"{x1}_{y1}_{x2}_{y2}"

            if det_key in found_vehicles:
                continue

            # Try multiple crop regions on the expanded box
            crop_regions = [
                ("bottom", y1 + int(veh_h_expanded * 0.50), y2_expanded, x1, x2),  # Bottom 50%
                ("mid",    y1 + int(veh_h_expanded * 0.30), y1 + int(veh_h_expanded * 0.75), x1, x2),
                ("full",   y1, y2_expanded, x1, x2),
            ]

            plate_found = False
            for region_name, ry1, ry2, rx1, rx2 in crop_regions:
                if plate_found:
                    break

                crop = image[ry1:ry2, rx1:rx2]
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 20:
                    continue

                try:
                    # 1. Resize the image 2x (EasyOCR works best when text height is 30-50px)
                    resized = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    
                    # 2. Convert to grayscale
                    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                    
                    # 3. Bilateral Filter (Removes noise/grain while keeping text edges razor sharp)
                    blur = cv2.bilateralFilter(gray, 11, 17, 17)

                    # First pass: clean blurred grayscale
                    ocr_results = self.ocr.readtext(
                        blur, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -"
                    )

                    # Second pass (fallback): Strict Otsu Thresholding (pure black & white)
                    if not ocr_results:
                        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        ocr_results = self.ocr.readtext(
                            thresh, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -"
                        )

                    import re
                    if ocr_results:
                        # Indian motorcycle plates are often on 2 lines. 
                        # Combine all text found in this crop region into one string.
                        combined_text = ""
                        min_conf = 1.0
                        for bbox_pts, text, conf in ocr_results:
                            combined_text += text + " "
                            if conf < min_conf:
                                min_conf = float(conf)

                        clean_text = combined_text.upper().replace(" ", "").replace("-", "").replace(".", "")
                        
                        # Validate using Indian License Plate Regex pattern
                        is_valid_plate = bool(re.search(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$', clean_text))
                        
                        # Fallback pattern for slightly misread plates
                        is_probable_plate = len(clean_text) >= 6 and len(clean_text) <= 10 and sum(c.isdigit() for c in clean_text) >= 3

                        if (is_valid_plate or is_probable_plate) and min_conf > 0.05:
                            plates.append({
                                "text": combined_text.strip().upper(),
                                "clean_text": clean_text,
                                "confidence": round(min_conf, 3),
                                "vehicle_bbox": det["bbox"].tolist() if hasattr(det["bbox"], "tolist") else list(det["bbox"]),
                                "vehicle_type": det.get("vehicle_type", "unknown"),
                            })
                            found_vehicles.add(det_key)
                            plate_found = True
                            break
                except Exception:
                    continue

        return plates

    # ── Full Pipeline ────────────────────────────────────────────────────

    def run(self, image: np.ndarray, skip_ocr: bool = False) -> dict:
        """
        Run the complete detection pipeline on a single image.
        Returns a structured dict with all detections and analyses.
        """
        # Preprocess
        enhanced = self.preprocess(image.copy())

        # Detect all objects
        detections = self.detect_objects(enhanced)

        # Categorize detections
        vehicles = [d for d in detections if d["category"] == "vehicle"]
        persons = [d for d in detections if d["category"] == "person"]
        traffic_lights = [d for d in detections if d["category"] == "traffic_light"]

        # Analyze traffic light colors
        for tl in traffic_lights:
            tl["color"] = self.detect_traffic_light_color(enhanced, tl["bbox"])

        # Helmet checks for persons
        for person in persons:
            status, conf = self.check_helmet(enhanced, person["bbox"])
            person["helmet_status"] = status
            person["helmet_confidence"] = conf

        # Read license plates (OPTIONAL)
        if skip_ocr:
            plates = []
        else:
            plates = self.read_plates(enhanced, vehicles)

        return {
            "vehicles": vehicles,
            "persons": persons,
            "traffic_lights": traffic_lights,
            "plates": plates,
            "all_detections": detections,
            "image_enhanced": enhanced,
        }
