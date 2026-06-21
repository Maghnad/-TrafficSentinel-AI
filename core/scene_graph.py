"""
TrafficSentinel AI — Scene Graph Builder & Violation Engine
Builds spatial relationships between detected objects and infers violations
using rule-based reasoning (no ML training required).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import uuid


# ── Helper Functions ─────────────────────────────────────────────────────

def compute_iou(box1, box2) -> float:
    """Compute Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / max(union, 1e-6)


def compute_overlap_ratio(small_box, big_box) -> float:
    """What fraction of small_box's area overlaps with big_box."""
    x1 = max(small_box[0], big_box[0])
    y1 = max(small_box[1], big_box[1])
    x2 = min(small_box[2], big_box[2])
    y2 = min(small_box[3], big_box[3])
    
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h
    
    small_area = (small_box[2] - small_box[0]) * (small_box[3] - small_box[1])
    return intersection / max(small_area, 1e-6)


def vertical_overlap(box1, box2) -> float:
    """Check vertical alignment between two boxes."""
    y_overlap = min(box1[3], box2[3]) - max(box1[1], box2[1])
    min_height = min(box1[3] - box1[1], box2[3] - box2[1])
    return y_overlap / max(min_height, 1e-6)


def horizontal_center_distance(box1, box2) -> float:
    """Distance between horizontal centers of two boxes, normalized by avg width."""
    cx1 = (box1[0] + box1[2]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    avg_w = ((box1[2] - box1[0]) + (box2[2] - box2[0])) / 2
    return abs(cx1 - cx2) / max(avg_w, 1e-6)


# ── Data Classes ─────────────────────────────────────────────────────────

@dataclass
class SceneNode:
    """A single detected object in the scene."""
    node_id: int
    detection: dict
    relationships: list = field(default_factory=list)
    attributes: dict = field(default_factory=dict)


@dataclass
class SceneEdge:
    """A relationship between two scene nodes."""
    source: SceneNode
    relation: str  # ON, IN, NEAR, CARRIES
    target: SceneNode
    confidence: float


@dataclass
class Violation:
    """A detected traffic violation."""
    violation_id: str
    violation_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    confidence: float
    description: str
    involved_nodes: list
    fine_amount: int
    evidence_bbox: list  # [x1, y1, x2, y2] bounding box of violation region


@dataclass
class SceneGraph:
    """Complete scene graph with nodes, edges, and violations."""
    nodes: List[SceneNode] = field(default_factory=list)
    edges: List[SceneEdge] = field(default_factory=list)
    violations: List[Violation] = field(default_factory=list)

    def get_text_tree(self) -> str:
        """Generate a human-readable text tree of the scene graph."""
        lines = ["📊 Scene Graph:"]
        
        # Group nodes by category
        vehicles = [n for n in self.nodes if n.detection.get("category") == "vehicle"]
        persons = [n for n in self.nodes if n.detection.get("category") == "person"]
        tls = [n for n in self.nodes if n.detection.get("category") == "traffic_light"]
        
        for v in vehicles:
            vtype = v.detection.get("vehicle_type", v.detection["class_name"])
            conf = v.detection["confidence"]
            lines.append(f"├── 🚗 {vtype.title()} (conf: {conf:.0%})")
            
            # Find persons on/in this vehicle
            riders = [
                e.source for e in self.edges
                if e.target.node_id == v.node_id and e.relation in ("ON", "IN")
            ]
            for i, rider in enumerate(riders):
                prefix = "│   ├──" if i < len(riders) - 1 else "│   └──"
                helmet = rider.attributes.get("helmet_status", "N/A")
                helmet_icon = "✅" if helmet == "HELMET" else "❌" if helmet == "NO_HELMET" else "❓"
                role = "Rider" if vtype in ("motorcycle", "bicycle", "scooter") else "Occupant"
                lines.append(f"{prefix} 👤 {role} (conf: {rider.detection['confidence']:.0%})")
                if helmet != "N/A":
                    sub_prefix = "│   │   └──" if i < len(riders) - 1 else "│       └──"
                    lines.append(f"{sub_prefix} {helmet_icon} Helmet: {helmet}")
            
            # Find plate for this vehicle
            plate = v.attributes.get("plate_text")
            if plate:
                lines.append(f"│   └── 🔢 Plate: {plate}")
        
        for tl in tls:
            color = tl.detection.get("color", "UNKNOWN")
            color_icon = "🔴" if color == "RED" else "🟡" if color == "YELLOW" else "🟢" if color == "GREEN" else "⚪"
            lines.append(f"├── 🚦 Traffic Light: {color_icon} {color}")
        
        remaining_persons = [
            n for n in persons
            if not any(e.source.node_id == n.node_id for e in self.edges)
        ]
        for p in remaining_persons:
            lines.append(f"├── 🚶 Pedestrian (conf: {p.detection['confidence']:.0%})")
        
        if not lines[1:]:
            lines.append("└── (No objects detected)")
        
        return "\n".join(lines)


# ── Scene Graph Builder ──────────────────────────────────────────────────

class SceneGraphBuilder:
    """
    Builds a scene graph from raw detection results.
    Establishes spatial relationships between objects.
    """

    MOTORCYCLE_TYPES = {"motorcycle", "bicycle"}
    CAR_TYPES = {"car", "bus", "truck"}

    def build(self, pipeline_results: dict) -> SceneGraph:
        """Build scene graph from pipeline detection results."""
        graph = SceneGraph()

        # Create nodes for all detections
        for i, det in enumerate(pipeline_results["all_detections"]):
            node = SceneNode(node_id=i, detection=det)
            
            # Copy helmet attributes
            if det["category"] == "person":
                node.attributes["helmet_status"] = det.get("helmet_status", "N/A")
                node.attributes["helmet_confidence"] = det.get("helmet_confidence", 0.0)
            
            graph.nodes.append(node)

        # Get node groups
        vehicle_nodes = [n for n in graph.nodes if n.detection.get("category") == "vehicle"]
        person_nodes = [n for n in graph.nodes if n.detection.get("category") == "person"]

        # Build person-vehicle relationships
        for person in person_nodes:
            best_vehicle = None
            best_overlap = 0.0

            for vehicle in vehicle_nodes:
                overlap = compute_overlap_ratio(
                    person.detection["bbox"], vehicle.detection["bbox"]
                )
                # Also check proximity (person bottom near vehicle)
                v_overlap = vertical_overlap(
                    person.detection["bbox"], vehicle.detection["bbox"]
                )
                h_dist = horizontal_center_distance(
                    person.detection["bbox"], vehicle.detection["bbox"]
                )

                score = overlap * 0.5 + v_overlap * 0.3 + max(0, 1 - h_dist) * 0.2

                if score > best_overlap and score > 0.2:
                    best_overlap = score
                    best_vehicle = vehicle

            if best_vehicle is not None:
                vtype = best_vehicle.detection.get("vehicle_type", "")
                relation = "ON" if vtype in self.MOTORCYCLE_TYPES else "IN"
                
                edge = SceneEdge(
                    source=person,
                    relation=relation,
                    target=best_vehicle,
                    confidence=round(best_overlap, 2),
                )
                graph.edges.append(edge)
                person.relationships.append((relation, best_vehicle))
                best_vehicle.relationships.append(("CARRIES", person))

        # Associate plates with vehicles
        for plate_info in pipeline_results.get("plates", []):
            plate_bbox = plate_info["vehicle_bbox"]
            for vehicle in vehicle_nodes:
                vbbox = vehicle.detection["bbox"]
                if (abs(vbbox[0] - plate_bbox[0]) < 10 and abs(vbbox[1] - plate_bbox[1]) < 10):
                    vehicle.attributes["plate_text"] = plate_info["text"]
                    vehicle.attributes["plate_confidence"] = plate_info["confidence"]
                    break

        return graph


# ── Violation Engine ─────────────────────────────────────────────────────

class ViolationEngine:
    """
    Rule-based violation detection from scene graph.
    No ML training required — pure logic.
    """

    VIOLATION_CONFIG = {
        "HELMET_VIOLATION": {
            "severity": "HIGH",
            "fine": 1000,
            "description": "Rider/pillion on two-wheeler without helmet",
        },
        "TRIPLE_RIDING": {
            "severity": "HIGH",
            "fine": 1000,
            "description": "More than 2 persons on a two-wheeler",
        },
        "RED_LIGHT_VIOLATION": {
            "severity": "CRITICAL",
            "fine": 5000,
            "description": "Vehicle crossing intersection during red signal",
        },
        "NO_PLATE_VISIBLE": {
            "severity": "MEDIUM",
            "fine": 2000,
            "description": "Vehicle with no visible/readable license plate",
        },
        "OVERCROWDING": {
            "severity": "MEDIUM",
            "fine": 1500,
            "description": "Excessive occupants detected in vehicle",
        },
        "SEATBELT_NON_COMPLIANCE": {
            "severity": "MEDIUM",
            "fine": 1000,
            "description": "Occupant inside four-wheeler not wearing seatbelt",
        },
        "ILLEGAL_PARKING": {
            "severity": "LOW",
            "fine": 500,
            "description": "Vehicle stopped/parked in restricted 'No Parking' zone",
        },
    }

    def detect_violations(self, graph: SceneGraph, pipeline_results: dict) -> List[Violation]:
        """Run all violation checks and return list of violations."""
        violations = []
        violations.extend(self._check_helmet_violations(graph))
        violations.extend(self._check_triple_riding(graph))
        violations.extend(self._check_red_light(graph, pipeline_results))
        violations.extend(self._check_seatbelt_violations(graph))
        violations.extend(self._check_illegal_parking(graph, pipeline_results))
        
        graph.violations = violations
        return violations

    def _check_helmet_violations(self, graph: SceneGraph) -> List[Violation]:
        """Check for riders on motorcycles without helmets."""
        violations = []
        motorcycles = [
            n for n in graph.nodes
            if n.detection.get("vehicle_type") in ("motorcycle", "bicycle")
        ]

        for moto in motorcycles:
            riders = [
                e.source for e in graph.edges
                if e.target.node_id == moto.node_id and e.relation == "ON"
            ]

            for rider in riders:
                helmet_status = rider.attributes.get("helmet_status", "UNCERTAIN")
                helmet_conf = rider.attributes.get("helmet_confidence", 0.0)

                if helmet_status == "NO_HELMET":
                    # Composite confidence
                    det_conf = rider.detection["confidence"]
                    assoc_conf = next(
                        (e.confidence for e in graph.edges
                         if e.source.node_id == rider.node_id),
                        0.5,
                    )
                    composite = 0.4 * det_conf + 0.3 * assoc_conf + 0.3 * helmet_conf

                    bbox = self._merge_bboxes(
                        rider.detection["bbox"], moto.detection["bbox"]
                    )

                    violations.append(Violation(
                        violation_id=f"VIO-{uuid.uuid4().hex[:8].upper()}",
                        violation_type="HELMET_VIOLATION",
                        severity="HIGH",
                        confidence=round(composite, 2),
                        description=f"Rider on {moto.detection.get('vehicle_type', 'motorcycle')} without helmet",
                        involved_nodes=[rider.node_id, moto.node_id],
                        fine_amount=1000,
                        evidence_bbox=bbox,
                    ))

        return violations

    def _check_triple_riding(self, graph: SceneGraph) -> List[Violation]:
        """Check for more than 2 persons on a two-wheeler."""
        violations = []
        motorcycles = [
            n for n in graph.nodes
            if n.detection.get("vehicle_type") in ("motorcycle", "bicycle")
        ]

        for moto in motorcycles:
            riders = [
                e.source for e in graph.edges
                if e.target.node_id == moto.node_id and e.relation == "ON"
            ]

            if len(riders) >= 3:
                all_bboxes = [r.detection["bbox"] for r in riders] + [moto.detection["bbox"]]
                merged = self._merge_multiple_bboxes(all_bboxes)
                
                avg_det_conf = np.mean([r.detection["confidence"] for r in riders])
                composite = 0.5 * avg_det_conf + 0.5 * min(1.0, len(riders) / 3.0)

                violations.append(Violation(
                    violation_id=f"VIO-{uuid.uuid4().hex[:8].upper()}",
                    violation_type="TRIPLE_RIDING",
                    severity="HIGH",
                    confidence=round(composite, 2),
                    description=f"{len(riders)} persons detected on two-wheeler (max allowed: 2)",
                    involved_nodes=[r.node_id for r in riders] + [moto.node_id],
                    fine_amount=1000,
                    evidence_bbox=merged,
                ))

        return violations

    def _check_red_light(self, graph: SceneGraph, pipeline_results: dict) -> List[Violation]:
        """Check for vehicles at a red traffic light."""
        violations = []
        
        red_lights = [
            n for n in graph.nodes
            if n.detection.get("category") == "traffic_light"
            and n.detection.get("color") == "RED"
        ]

        if not red_lights:
            return violations

        vehicles = [n for n in graph.nodes if n.detection.get("category") == "vehicle"]

        for vehicle in vehicles:
            for rl in red_lights:
                # Check if vehicle is near / beyond the traffic light
                vbbox = vehicle.detection["bbox"]
                rl_bbox = rl.detection["bbox"]

                # Simple heuristic: if vehicle center is below traffic light
                v_center_y = (vbbox[1] + vbbox[3]) / 2
                rl_bottom = rl_bbox[3]

                # Vehicle should be in similar horizontal range
                h_dist = horizontal_center_distance(vbbox, rl_bbox)

                if v_center_y > rl_bottom and h_dist < 3.0:
                    composite = 0.5 * vehicle.detection["confidence"] + 0.3 * rl.detection["confidence"] + 0.2 * max(0, 1 - h_dist / 3)
                    
                    violations.append(Violation(
                        violation_id=f"VIO-{uuid.uuid4().hex[:8].upper()}",
                        violation_type="RED_LIGHT_VIOLATION",
                        severity="CRITICAL",
                        confidence=round(composite, 2),
                        description=f"{vehicle.detection.get('vehicle_type', 'Vehicle').title()} detected at red signal",
                        involved_nodes=[vehicle.node_id],
                        fine_amount=5000,
                        evidence_bbox=self._merge_bboxes(vbbox, rl_bbox),
                    ))
                    break  # One violation per vehicle

        return violations

    def _check_no_plate(self, graph: SceneGraph, pipeline_results: dict) -> List[Violation]:
        """Flag vehicles with no readable license plate."""
        violations = []
        plates = pipeline_results.get("plates", [])
        plate_vehicle_bboxes = [p["vehicle_bbox"] for p in plates]

        vehicles = [n for n in graph.nodes if n.detection.get("category") == "vehicle"]

        for vehicle in vehicles:
            vbbox = list(vehicle.detection["bbox"])
            has_plate = any(
                self._bbox_match(vbbox, pvb) for pvb in plate_vehicle_bboxes
            )
            if not has_plate:
                violations.append(Violation(
                    violation_id=f"VIO-{uuid.uuid4().hex[:8].upper()}",
                    violation_type="NO_PLATE_VISIBLE",
                    severity="MEDIUM",
                    confidence=round(0.5 * vehicle.detection["confidence"], 2),
                    description=f"No readable plate on {vehicle.detection.get('vehicle_type', 'vehicle')}",
                    involved_nodes=[vehicle.node_id],
                    fine_amount=2000,
                    evidence_bbox=list(vehicle.detection["bbox"]),
                ))

        return violations

    # ── Helpers ───────────────────────────────────────────────────────────

    def _merge_bboxes(self, box1, box2):
        """Combine two boxes into a single bounding box."""
        return [
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3]),
        ]

    def _merge_multiple_bboxes(self, boxes):
        """Combine multiple boxes into a single bounding box."""
        if not boxes:
            return [0, 0, 0, 0]
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return [x1, y1, x2, y2]

    def _check_seatbelt_violations(self, graph: SceneGraph) -> List[Violation]:
        """Check for occupants in four-wheelers without seatbelts (Heuristic Demo)."""
        violations = []
        cars = [n for n in graph.nodes if n.detection.get("vehicle_type") in ("car", "truck", "bus")]
        
        for car in cars:
            occupants = [
                e.source for e in graph.edges
                if e.target.node_id == car.node_id and e.relation == "IN"
            ]
            
            for occ in occupants:
                # Heuristic: Since standard YOLO doesn't detect seatbelts inside vehicles,
                # we randomly flag occupants deterministically based on ID to simulate it for the demo.
                if occ.node_id % 7 == 0:  
                    bbox = self._merge_bboxes(occ.detection["bbox"], car.detection["bbox"])
                    conf = occ.detection["confidence"] * 0.9  # Simulated confidence
                    
                    violations.append(Violation(
                        violation_id=f"VIO-{uuid.uuid4().hex[:8].upper()}",
                        violation_type="SEATBELT_NON_COMPLIANCE",
                        severity="MEDIUM",
                        confidence=round(conf, 2),
                        description=f"Occupant inside {car.detection.get('vehicle_type', 'car')} without seatbelt",
                        involved_nodes=[occ.node_id, car.node_id],
                        fine_amount=self.VIOLATION_CONFIG["SEATBELT_NON_COMPLIANCE"]["fine"],
                        evidence_bbox=bbox,
                    ))
        return violations

    def _check_illegal_parking(self, graph: SceneGraph, pipeline_results: dict) -> List[Violation]:
        """Check if vehicles are stopped in a simulated 'No Parking Zone'."""
        violations = []
        
        img_h, img_w = pipeline_results.get("image_shape", (1080, 1920))[:2]
        vehicles = [n for n in graph.nodes if n.detection.get("category") == "vehicle"]
        
        for v in vehicles:
            bbox = v.detection["bbox"]
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            # Simulated No Parking Zone: bottom 20% and right 20% of the screen
            if cx > img_w * 0.8 and cy > img_h * 0.8:
                conf = v.detection["confidence"] * 0.95
                violations.append(Violation(
                    violation_id=f"VIO-{uuid.uuid4().hex[:8].upper()}",
                    violation_type="ILLEGAL_PARKING",
                    severity="LOW",
                    confidence=round(conf, 2),
                    description="Vehicle detected in restricted 'No Parking' zone",
                    involved_nodes=[v.node_id],
                    fine_amount=self.VIOLATION_CONFIG["ILLEGAL_PARKING"]["fine"],
                    evidence_bbox=bbox,
                ))
        return violations

    @staticmethod
    def _bbox_match(box1, box2, tolerance=15) -> bool:
        return all(abs(a - b) < tolerance for a, b in zip(box1, box2))


# ── Composite Confidence Scorer ──────────────────────────────────────────

class ConfidenceScorer:
    """Calculate composite confidence for violations."""

    @staticmethod
    def categorize(confidence: float) -> str:
        if confidence >= 0.75:
            return "AUTO_FLAG"
        elif confidence >= 0.50:
            return "HUMAN_REVIEW"
        else:
            return "LOW_CONFIDENCE"

    @staticmethod
    def get_color(confidence: float) -> str:
        if confidence >= 0.75:
            return "#ef4444"  # Red — high confidence violation
        elif confidence >= 0.50:
            return "#f59e0b"  # Amber — needs review
        else:
            return "#6b7280"  # Gray — low confidence
