"""Rule engine.

Differences from the original design that matter:

* RED_LIGHT now requires an actual stop-line crossing while the signal is red.
  The original rule ("vehicle centre below the light and horizontally aligned")
  fines every car correctly stopped at a red - which is close to 100% false
  positives at a busy junction.

* Traffic light state uses HSV hue masks, not "brightest third". Brightness
  alone inverts at night when the whole housing blooms, and fails completely
  for horizontally-mounted signals.

* SEATBELT is removed. `node_id % 7 == 0` is a random number generator wearing
  a violation label, and it was writing real rows with real fine amounts into
  the database. Detecting seatbelts through a windshield from a pole-mounted
  camera is not a solved problem; claiming it is will cost you more credibility
  than the feature gains.

* ILLEGAL_PARKING uses operator-drawn polygons plus a dwell timer, not
  "bottom-right 20% of the frame".

* Every violation carries a `reasons` list - the audit chain. This is what
  makes rule-based enforcement defensible in a way an end-to-end model is not,
  and it is worth putting on screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .config import VIOLATION_META
from .geometry import crossed_line, point_in_polygon, ttc


@dataclass
class Violation:
    vtype: str
    track_id: int
    bbox: tuple
    confidence: float
    frame_idx: int
    reasons: List[str] = field(default_factory=list)
    partner_track_id: Optional[int] = None
    measured_value: Optional[float] = None

    @property
    def fine(self) -> int:
        return VIOLATION_META.get(self.vtype, {}).get("fine", 0)

    @property
    def severity(self) -> str:
        return VIOLATION_META.get(self.vtype, {}).get("severity", "LOW")


# ---------------------------------------------------------------------- #
# Traffic light state
# ---------------------------------------------------------------------- #

def traffic_light_state(frame: np.ndarray, bbox) -> tuple:
    """Returns (state, confidence). state in RED / YELLOW / GREEN / UNKNOWN."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 6:
        return "UNKNOWN", 0.0

    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Only strongly saturated, bright pixels count as an illuminated lamp.
    masks = {
        "RED": cv2.bitwise_or(
            cv2.inRange(hsv, (0, 110, 140), (10, 255, 255)),
            cv2.inRange(hsv, (170, 110, 140), (180, 255, 255))),
        "YELLOW": cv2.inRange(hsv, (18, 110, 150), (33, 255, 255)),
        "GREEN": cv2.inRange(hsv, (45, 90, 120), (92, 255, 255)),
    }
    total = crop.shape[0] * crop.shape[1]
    scores = {k: float(np.count_nonzero(m)) / total for k, m in masks.items()}

    state = max(scores, key=scores.get)
    score = scores[state]
    if score < 0.035:
        return "UNKNOWN", 0.0

    runner_up = sorted(scores.values())[-2]
    conf = min(1.0, (score - runner_up) * 6.0 + 0.4)
    return state, conf


# ---------------------------------------------------------------------- #

class ViolationEngine:
    def __init__(self, cfg, geometry, ground, helmet_clf):
        self.cfg = cfg.rules
        self.geo = geometry
        self.ground = ground
        self.helmet = helmet_clf
        self.light_state = "UNKNOWN"
        self.light_conf = 0.0
        self._enabled = {k: True for k in VIOLATION_META}

    def set_enabled(self, vtype: str, on: bool) -> None:
        self._enabled[vtype] = on

    def on(self, vtype: str) -> bool:
        return self._enabled.get(vtype, True)

    # ------------------------------------------------------------------ #

    def evaluate(self, frame, graph, registry, frame_idx: int) -> List[Violation]:
        out: List[Violation] = []
        self._update_light(frame, graph)

        for vnode in graph.vehicles():
            tr = vnode.det.get("_track")
            # Always run helmet/triple-riding — they only need bounding boxes
            out.extend(self._two_wheeler_rules(frame, graph, vnode, tr, frame_idx))
            # Motion, parking, and red-light rules need track kinematics
            if tr is not None:
                out.extend(self._motion_rules(vnode, tr, frame_idx))
                out.extend(self._parking_rule(vnode, tr, frame_idx, registry))
                out.extend(self._red_light_rule(vnode, tr, frame_idx))

        if self.on("NEAR_MISS"):
            out.extend(self._near_miss(graph, frame_idx))
        return out

    # ------------------------------------------------------------------ #

    def _update_light(self, frame, graph) -> None:
        best_state, best_conf, best_area = "UNKNOWN", 0.0, 0.0
        for node in graph.traffic_lights():
            x1, y1, x2, y2 = node.bbox
            area = (x2 - x1) * (y2 - y1)
            state, conf = traffic_light_state(frame, node.bbox)
            if state != "UNKNOWN" and area > best_area:
                best_state, best_conf, best_area = state, conf, area
            node.attributes["state"] = state
        if best_state != "UNKNOWN":
            self.light_state, self.light_conf = best_state, best_conf

    # ------------------------------------------------------------------ #

    def _two_wheeler_rules(self, frame, graph, vnode, tr, frame_idx):
        if vnode.det["vehicle_type"] not in ("motorcycle", "bicycle"):
            return []
        riders = graph.riders_of(vnode.node_id)
        if not riders:
            return []
        found = []
        # Stable ID: use track_id when ByteTrack confirmed, else detection id
        tid = tr.track_id if tr is not None else vnode.det.get("track_id", -1)

        # --- Triple riding -------------------------------------------- #
        if self.on("TRIPLE_RIDING") and len(riders) >= self.cfg.triple_riding_min:
            assoc = [e.confidence for e in graph.edges
                     if e.target == vnode.node_id]
            det_conf = float(np.mean([r.det["conf"] for r in riders]))
            conf = 0.4 * det_conf + 0.4 * float(np.mean(assoc)) + 0.2 * min(
                1.0, len(riders) / 3.0)
            found.append(Violation(
                "TRIPLE_RIDING", tid, vnode.bbox, conf, frame_idx,
                reasons=[f"{len(riders)} persons associated ON "
                         f"motorcycle#{tid}",
                         "mean association score "
                         f"{float(np.mean(assoc)):.2f}"],
                measured_value=float(len(riders))))

        # --- Helmet ---------------------------------------------------- #
        if self.on("NO_HELMET"):
            for r in riders:
                status, hconf = self.helmet.classify(frame, r.bbox)
                r.attributes["helmet"] = status
                r.attributes["helmet_conf"] = hconf
                if status != "NO_HELMET":
                    continue
                assoc = next((e.confidence for e in graph.edges
                              if e.source == r.node_id), 0.5)
                conf = 0.3 * r.det["conf"] + 0.3 * assoc + 0.4 * hconf
                found.append(Violation(
                    "NO_HELMET", tid, r.bbox, conf, frame_idx,
                    reasons=[f"person#{r.det['track_id']} ON "
                             f"motorcycle#{tid} (assoc {assoc:.2f})",
                             f"head crop classified NO_HELMET ({hconf:.2f})"],
                    partner_track_id=r.det["track_id"]))
        return found

    # ------------------------------------------------------------------ #

    def _motion_rules(self, vnode, tr, frame_idx):
        found = []
        if not self.ground.calibrated:
            return found

        # --- Overspeeding ---------------------------------------------- #
        if (self.on("OVERSPEEDING") and tr.speed_kmh is not None
                and tr.age_frames() >= self.cfg.speed_min_track_frames
                and tr.speed_kmh > self.cfg.speed_limit_kmh):
            over = tr.speed_kmh - self.cfg.speed_limit_kmh
            # Confidence rises with margin over the limit and with track length,
            # so a marginal 61 km/h reading does not auto-issue.
            conf = min(0.98, 0.45
                       + min(0.35, over / max(1.0, self.cfg.speed_limit_kmh))
                       + min(0.2, tr.age_frames() / 120.0))
            found.append(Violation(
                "OVERSPEEDING", tr.track_id, vnode.bbox, conf, frame_idx,
                reasons=[f"median ground-plane speed {tr.speed_kmh:.1f} km/h",
                         f"limit {self.cfg.speed_limit_kmh:.0f} km/h",
                         f"{len(tr.speed_samples)} samples over "
                         f"{tr.age_frames()} frames"],
                measured_value=tr.speed_kmh))

        # --- Wrong way -------------------------------------------------- #
        if self.on("WRONG_WAY"):
            disp = None
            if len(tr.history) >= 6:
                w0 = next((w for _, _, w in tr.history if w is not None), None)
                w1 = tr.history[-1][2]
                if w0 and w1:
                    disp = (w1[0] - w0[0], w1[1] - w0[1])
            if disp is not None:
                mag = float(np.hypot(*disp))
                if mag >= self.cfg.wrongway_min_displacement_m:
                    lane = np.asarray(self.geo.lane_direction, dtype=float)
                    n = np.linalg.norm(lane)
                    if n > 1e-6:
                        dot = float(np.dot(np.asarray(disp) / mag, lane / n))
                        if dot <= self.cfg.wrongway_dot_threshold:
                            found.append(Violation(
                                "WRONG_WAY", tr.track_id, vnode.bbox,
                                min(0.95, 0.55 + abs(dot) * 0.4), frame_idx,
                                reasons=[f"travelled {mag:.1f} m against lane "
                                         f"direction (cos {dot:.2f})"],
                                measured_value=dot))
        return found

    # ------------------------------------------------------------------ #

    def _parking_rule(self, vnode, tr, frame_idx, registry):
        if not self.on("ILLEGAL_PARKING") or not self.geo.no_parking_zones:
            return []
        foot = vnode.det["foot"]
        inside = any(point_in_polygon(p, foot)
                     for p in self.geo.no_parking_zones)
        if not inside:
            tr.stationary_since = None
            return []

        moving = tr.speed_kmh is not None and tr.speed_kmh > 3.0
        if moving:
            tr.stationary_since = None
            return []

        if tr.stationary_since is None:
            tr.stationary_since = frame_idx
            return []

        dwell_frames = frame_idx - tr.stationary_since
        needed = self.geo.parking_dwell_seconds * registry.fps
        if dwell_frames < needed:
            return []

        return [Violation(
            "ILLEGAL_PARKING", tr.track_id, vnode.bbox,
            min(0.95, 0.6 + dwell_frames / (needed * 4.0)), frame_idx,
            reasons=[f"stationary inside no-parking polygon for "
                     f"{dwell_frames / registry.fps:.0f} s"],
            measured_value=dwell_frames / registry.fps)]

    # ------------------------------------------------------------------ #

    def _red_light_rule(self, vnode, tr, frame_idx):
        if not self.on("RED_LIGHT") or not self.geo.stop_line:
            return []
        if self.light_state != "RED" or self.light_conf < 0.45:
            return []
        if len(tr.history) < 2:
            return []

        prev = tr.history[-2][1]
        curr = tr.history[-1][1]
        if not crossed_line(self.geo.stop_line, prev, curr):
            return []

        conf = min(0.97, 0.5 + 0.35 * self.light_conf + 0.15 * tr.conf)
        return [Violation(
            "RED_LIGHT", tr.track_id, vnode.bbox, conf, frame_idx,
            reasons=[f"signal state RED (conf {self.light_conf:.2f})",
                     f"{vnode.det['vehicle_type']}#{tr.track_id} ground point "
                     f"crossed stop line between frames "
                     f"{tr.history[-2][0]} and {frame_idx}"])]

    # ------------------------------------------------------------------ #

    def _near_miss(self, graph, frame_idx):
        """Conflict detection. Not a chargeable offence - it is a road-safety
        analytics signal. Junctions that generate many near-misses are exactly
        the ones that need redesign, and this data does not exist today because
        nothing records collisions that *didn't* happen."""
        if not self.ground.calibrated:
            return []
        out = []
        vehicles = [n for n in graph.vehicles()
                    if n.det.get("_track") is not None]
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                a = vehicles[i].det["_track"]
                b = vehicles[j].det["_track"]
                if a.world_velocity is None or b.world_velocity is None:
                    continue
                if (a.speed_kmh or 0) < self.cfg.nearmiss_min_speed_kmh and \
                   (b.speed_kmh or 0) < self.cfg.nearmiss_min_speed_kmh:
                    continue
                pa, pb = a.history[-1][2], b.history[-1][2]
                if pa is None or pb is None:
                    continue
                t = ttc(pa, a.world_velocity, pb, b.world_velocity)
                if t is None or t > self.cfg.nearmiss_ttc_s:
                    continue
                gap = float(np.hypot(pa[0] - pb[0], pa[1] - pb[1]))
                if gap > 25.0:
                    continue
                out.append(Violation(
                    "NEAR_MISS", a.track_id, vehicles[i].bbox,
                    min(0.9, 0.5 + (self.cfg.nearmiss_ttc_s - t) / 3.0),
                    frame_idx,
                    reasons=[f"time-to-collision {t:.2f} s with "
                             f"track#{b.track_id}",
                             f"separation {gap:.1f} m"],
                    partner_track_id=b.track_id, measured_value=t))
        return out



