"""Pipeline orchestrator.

Per-frame ordering is chosen so nothing blocking sits on the critical path:

    decode (thread)  ->  detect+track  ->  scene graph  ->  rules
                                                  |
                        OCR (thread)  <-----------+
                        evidence (thread) <-------+
                        db writes (thread) <------+

The main loop only ever does: inference, a handful of geometry, and drawing.
Everything with unbounded or high latency is handed to a worker with a bounded,
drop-oldest queue.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional

import numpy as np

from .annotator import Annotator
from .config import AppConfig
from .database import Database
from .detector import Detector
from .evidence import EvidenceWriter
from .geometry import GroundPlane
from .helmet import HelmetClassifier
from .ocr_worker import OCRWorker, plate_crop, get_car
from .alerts import TelegramAlertService, AlertJob
from .scene_graph import SceneGraphBuilder
from .tracks import TrackRegistry
from .violations import ViolationEngine


class TrafficSentinel:
    def __init__(self, cfg: AppConfig, fps: float = 25.0):
        self.cfg = cfg
        self.fps = fps

        self.detector = Detector(cfg.detector)
        self.ground = GroundPlane(cfg.geometry.homography_src,
                                  cfg.geometry.homography_dst)
        self.registry = TrackRegistry(fps, self.ground)
        self.builder = SceneGraphBuilder()
        self.helmet = HelmetClassifier(cfg.helmet)
        self.rules = ViolationEngine(cfg, cfg.geometry, self.ground,
                                     self.helmet)
        self.ocr = OCRWorker(cfg.ocr).start()
        self.alerts = TelegramAlertService(getattr(cfg, 'alerts', None) or AlertConfig()).start()
        self.db = Database(cfg.db_path)
        self.evidence = EvidenceWriter(cfg.evidence, fps,
                                       cfg.geometry.camera_id)
        self.annotator = Annotator(cfg.geometry)

        self._frame_times = deque(maxlen=30)
        self._last_flow_log = 0.0
        self.live_violations: deque = deque(maxlen=60)
        self.explain_log: deque = deque(maxlen=40)

        self.detector.warmup()

    # ------------------------------------------------------------------ #

    def process(self, frame: np.ndarray, frame_idx: int,
                draw: bool = True) -> Dict:
        t0 = time.perf_counter()

        self.rules.cfg = self.cfg.rules
        self.rules.geo = self.cfg.geometry
        self.annotator.geo = self.cfg.geometry
        if hasattr(self.cfg, 'alerts'):
            self.alerts.cfg = self.cfg.alerts

        detections = self.detector.detect(frame)
        self.registry.update(detections, frame_idx)
        graph = self.builder.build(detections)
        violations = self.rules.evaluate(frame, graph, self.registry, frame_idx)

        self._dispatch_ocr(frame, graph, violations)
        self._collect_ocr()
        new = self._commit(frame, violations, frame_idx)

        self.evidence.push_frame(frame)

        dt = time.perf_counter() - t0
        self._frame_times.append(dt)
        fps = 1.0 / max(1e-6, float(np.mean(self._frame_times)))

        hud = {
            "camera_id": self.cfg.geometry.camera_id,
            "fps": fps,
            "infer_ms": self.detector.last_infer_ms,
            "ocr_backlog": self.ocr.backlog,
            "light": self.rules.light_state,
            "tracks": len(self.registry.tracks),
            "calibrated": self.ground.calibrated,
        }

        annotated = None
        if draw:
            annotated = self.annotator.draw(frame.copy(), graph, violations,
                                            self.registry, hud)

        self._log_flow(graph)
        self._log_sightings(frame_idx)
        self.explain_log.extend(self.builder.explain(graph))

        return {
            "frame": annotated,
            "violations": violations,
            "new_violations": new,
            "graph": graph,
            "hud": hud,
            "latency_ms": dt * 1000.0,
        }

    # ------------------------------------------------------------------ #

    def _dispatch_ocr(self, frame, graph, violations=None) -> None:
        """Two-stage plate detection:

        Stage 1 (from reference main.py): Run YOLOv8 license_plate_detector on
        the FULL FRAME, crop each plate, threshold (BINARY_INV at 64), OCR, and
        assign to the parent vehicle via get_car() containment check.

        Stage 2 (fallback): For vehicles that Stage 1 missed, submit a bumper
        crop to the async OCR worker thread.
        """
        if not self.cfg.ocr.enabled:
            return

        # Build list of track IDs that are in violation
        violating_track_ids = set()
        if violations:
            for v in violations:
                violating_track_ids.add(v.get("track_id"))

        # ----- Stage 1: Full-frame plate detection (reference main.py) -----
        # Build vehicle_track_ids list: [[x1, y1, x2, y2, track_id], ...]
        vehicle_bboxes = []
        track_map = {}  # track_id -> track object
        for vnode in graph.vehicles():
            tr = vnode.det.get("_track")
            if tr is None:
                continue

            # Optimization: Only process OCR for vehicles that have committed a violation
            if tr.track_id not in violating_track_ids and not getattr(tr, 'logged', False):
                continue
            x1, y1, x2, y2 = vnode.bbox
            vehicle_bboxes.append([x1, y1, x2, y2, tr.track_id])
            track_map[tr.track_id] = tr

        # Detect license plates in full frame (reference: license_plates = license_plate_detector(frame)[0])
        plates = self.ocr.reader.detect_plates_in_frame(frame)

        for plate_info in plates:
            px1, py1, px2, py2 = plate_info["bbox"]
            plate_text = plate_info["text"]
            plate_score = plate_info["text_score"]

            if plate_text is None:
                continue

            # Assign plate to car (reference: get_car(license_plate, track_ids))
            _, _, _, _, car_id = get_car([px1, py1, px2, py2], vehicle_bboxes)

            if car_id != -1 and car_id in track_map:
                tr = track_map[car_id]
                if plate_score > tr.plate_conf:
                    tr.plate = plate_text
                    tr.plate_conf = plate_score
                    # Persist to DB and update live feed immediately
                    self.db.log_sighting(plate_text, self.cfg.geometry.camera_id,
                                         self.cfg.geometry.latitude,
                                         self.cfg.geometry.longitude,
                                         bool(tr.logged))
                    self.db.update_violation_plate(car_id, plate_text, plate_score)
                    for item in self.live_violations:
                        if item.get("track_id") == car_id and not item.get("plate"):
                            item["plate"] = plate_text

        # ----- Stage 2: Async fallback for vehicles still without plates -----
        budget = 3
        for vnode in graph.vehicles():
            tr = vnode.det.get("_track")
            if tr and tr.track_id not in violating_track_ids and not getattr(tr, 'logged', False):
                continue
            if budget <= 0:
                break
            tr = vnode.det.get("_track")
            if tr is None:
                continue
            if tr.plate and tr.plate_conf > 0.80:
                continue
            if tr.ocr_attempts >= self.cfg.ocr.max_attempts_per_track:
                continue

            last_frame = getattr(tr, "_last_ocr_submit", -10)
            if hasattr(tr, "last_frame") and tr.last_frame - last_frame < 3:
                continue

            x1, y1, x2, y2 = vnode.bbox
            h = y2 - y1
            if h < self.cfg.ocr.min_crop_height:
                continue
            crop = plate_crop(frame, vnode.bbox, self.cfg.ocr.min_crop_height)
            if crop is None:
                continue
            if self.ocr.submit(tr.track_id, crop):
                tr.ocr_attempts += 1
                tr._last_ocr_submit = getattr(tr, "last_frame", 0)
                budget -= 1


    def _collect_ocr(self) -> None:
        for tid, tr in self.registry.tracks.items():
            if tr.plate:
                continue
            res = self.ocr.result(tid)
            if res and res.plate and res.confidence > tr.plate_conf:
                tr.plate = res.plate
                tr.plate_conf = res.confidence
                self.db.log_sighting(res.plate, self.cfg.geometry.camera_id,
                                     self.cfg.geometry.latitude,
                                     self.cfg.geometry.longitude,
                                     bool(tr.logged))
                # Retroactively update database record and live list so review queue shows plate
                self.db.update_violation_plate(tid, res.plate, res.confidence)
                for item in self.live_violations:
                    if item.get("track_id") == tid and not item.get("plate"):
                        item["plate"] = res.plate

    # ------------------------------------------------------------------ #

    def _commit(self, frame, violations, frame_idx) -> List:
        """Deduplicate by (track_id, violation_type) and persist.

        This replaces the 150-px centre-distance heuristic, which merged two
        genuinely different motorcycles passing close together into one record.
        """
        new = []
        for v in violations:
            if v.vtype == "NEAR_MISS":
                key_id = f"{min(v.track_id, v.partner_track_id or -1)}-" \
                         f"{max(v.track_id, v.partner_track_id or -1)}"
                if self.registry.already_logged(v.track_id, f"NM:{key_id}"):
                    continue
                self.registry.mark_logged(v.track_id, f"NM:{key_id}")
            else:
                if self.registry.already_logged(v.track_id, v.vtype):
                    continue
                self.registry.mark_logged(v.track_id, v.vtype)

            tr = self.registry.get(v.track_id)
            plate = tr.plate if tr else None
            plate_conf = tr.plate_conf if tr else 0.0

            # Two-tier enforcement: auto-issue only when confident AND we know
            # who to bill. Everything else queues for a human.
            issue = (v.confidence >= self.cfg.rules.auto_issue_confidence
                     and bool(plate) and v.fine > 0)
            status = "issued" if issue else "review"
            if v.vtype == "NEAR_MISS":
                status = "analytics"

            crop_path, clip_path = self.evidence.capture(
                frame, v.bbox, v.vtype, v.track_id)

            self.db.log_violation(
                camera_id=self.cfg.geometry.camera_id,
                track_id=v.track_id, vtype=v.vtype, severity=v.severity,
                confidence=v.confidence, fine=v.fine, plate=plate,
                plate_conf=plate_conf, status=status,
                measured_value=v.measured_value, reasons=v.reasons,
                evidence_path=crop_path, clip_path=clip_path,
                lat=self.cfg.geometry.latitude,
                lon=self.cfg.geometry.longitude)

            # Dispatch real-time Telegram mobile alert
            try:
                self.alerts.dispatch(AlertJob(
                    vtype=v.vtype,
                    track_id=v.track_id,
                    severity=v.severity,
                    confidence=v.confidence,
                    fine=v.fine,
                    plate=plate,
                    camera_id=self.cfg.geometry.camera_id,
                    lat=self.cfg.geometry.latitude,
                    lon=self.cfg.geometry.longitude,
                    reasons=v.reasons,
                    evidence_path=crop_path
                ))
            except Exception:
                pass

            record = {
                "type": v.vtype, "track_id": v.track_id,
                "confidence": v.confidence, "fine": v.fine,
                "severity": v.severity, "plate": plate, "status": status,
                "reasons": v.reasons, "evidence": crop_path,
                "clip": clip_path, "frame": frame_idx,
            }
            self.live_violations.appendleft(record)
            new.append(record)
        return new

    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #

    def _log_sightings(self, frame_idx: int) -> None:
        """Log a single checkpoint sighting per vehicle per camera passage."""
        geo = self.cfg.geometry
        for tid, tr in self.registry.tracks.items():
            # If vehicle has not been logged at this camera yet, log once after it establishes track
            if not tr.sighting_logged or tr.last_sighting_cam != geo.camera_id:
                if tr.age_frames() >= 5:
                    plate_label = tr.plate if tr.plate else f"TRK-{tid}"
                    is_vio = bool(tr.logged)
                    self.db.log_sighting(plate_label, geo.camera_id,
                                         geo.latitude, geo.longitude, is_vio)
                    tr.sighting_logged = True
                    tr.last_sighting_cam = geo.camera_id
            elif tr.plate and f"TRK-{tid}" in (tr.last_sighting_cam or ""):
                # If plate was identified later, update sighting with actual plate number
                is_vio = bool(tr.logged)
                self.db.log_sighting(tr.plate, geo.camera_id,
                                     geo.latitude, geo.longitude, is_vio)
                tr.last_sighting_cam = geo.camera_id

    def _log_flow(self, graph) -> None:
        """Congestion analytics. Nearly free, and arguably more useful to a
        city than the fines: queue length and mean speed per junction over time
        is exactly the data traffic engineers do not have."""
        now = time.time()
        if now - self._last_flow_log < 5.0:
            return
        self._last_flow_log = now
        vehicles = graph.vehicles()
        speeds = [t.speed_kmh for t in self.registry.tracks.values()
                  if t.speed_kmh is not None]
        stopped = sum(1 for t in self.registry.tracks.values()
                      if t.speed_kmh is not None and t.speed_kmh < 5.0)
        self.db.log_flow(self.cfg.geometry.camera_id, len(vehicles),
                         float(np.mean(speeds)) if speeds else 0.0, stopped)

    def reset(self) -> None:
        self.detector.reset_tracks()
        self.registry.tracks.clear()
        self.live_violations.clear()

    def close(self) -> None:
        self.ocr.stop()
        self.alerts.stop()
        self.evidence.stop()
        self.db.close()




