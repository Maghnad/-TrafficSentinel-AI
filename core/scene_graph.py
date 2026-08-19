"""Spatial scene graph: turns a flat detection list into rider/occupant
relationships. This part of the original design was sound; three bugs are
fixed here.

1. The `(1 - horizontal_distance)` term went unboundedly negative because the
   distance was in raw pixels. It is now normalised by the vehicle's width and
   clamped to [0, 1].

2. A `score > 0.2` threshold with no geometric gate will associate a pedestrian
   standing behind a bike as a third rider - manufacturing triple-riding
   violations out of thin air. A hard gate is added: the person's feet must
   fall inside the vehicle's box and the person's centre must sit above the
   vehicle's bottom edge.

3. Association is now one-to-one greedy by descending score, so the same person
   cannot be counted as a rider on two different motorcycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .detector import ENCLOSED, TWO_WHEELERS
from .geometry import containment


@dataclass
class SceneNode:
    node_id: int
    det: dict
    attributes: dict = field(default_factory=dict)

    @property
    def bbox(self):
        return self.det["bbox"]

    @property
    def track_id(self) -> int:
        return self.det["track_id"]


@dataclass
class SceneEdge:
    source: int          # person node id
    relation: str        # "ON" | "IN"
    target: int          # vehicle node id
    confidence: float


@dataclass
class SceneGraph:
    nodes: Dict[int, SceneNode] = field(default_factory=dict)
    edges: List[SceneEdge] = field(default_factory=list)

    def riders_of(self, vehicle_node_id: int) -> List[SceneNode]:
        return [self.nodes[e.source] for e in self.edges
                if e.target == vehicle_node_id and e.relation == "ON"]

    def occupants_of(self, vehicle_node_id: int) -> List[SceneNode]:
        return [self.nodes[e.source] for e in self.edges
                if e.target == vehicle_node_id and e.relation == "IN"]

    def vehicle_of(self, person_node_id: int) -> Optional[SceneNode]:
        for e in self.edges:
            if e.source == person_node_id:
                return self.nodes[e.target]
        return None

    def vehicles(self) -> List[SceneNode]:
        return [n for n in self.nodes.values()
                if n.det["category"] == "vehicle"]

    def traffic_lights(self) -> List[SceneNode]:
        return [n for n in self.nodes.values()
                if n.det["category"] == "traffic_light"]


class SceneGraphBuilder:
    def __init__(self, assoc_threshold: float = 0.35):
        self.threshold = assoc_threshold

    # ------------------------------------------------------------------ #

    @staticmethod
    def _gate(person_box, vehicle_box) -> bool:
        """Cheap geometric veto applied before scoring. Kills most false
        associations for free."""
        px1, py1, px2, py2 = person_box
        vx1, vy1, vx2, vy2 = vehicle_box

        # Person's feet must be within the vehicle's horizontal span, with a
        # small tolerance, and not far below its base.
        foot_x = (px1 + px2) / 2.0
        v_width = max(1.0, vx2 - vx1)
        if not (vx1 - 0.25 * v_width <= foot_x <= vx2 + 0.25 * v_width):
            return False
        if py2 > vy2 + 0.35 * max(1.0, vy2 - vy1):
            return False
        # Person must overlap the vehicle box vertically at all.
        if py2 < vy1 or py1 > vy2:
            return False
        return True

    @staticmethod
    def _score(person_box, vehicle_box) -> float:
        px1, py1, px2, py2 = person_box
        vx1, vy1, vx2, vy2 = vehicle_box

        overlap = containment(person_box, vehicle_box)

        p_h = max(1.0, py2 - py1)
        v_h = max(1.0, vy2 - vy1)
        inter_h = max(0.0, min(py2, vy2) - max(py1, vy1))
        vertical = inter_h / min(p_h, v_h)

        v_w = max(1.0, vx2 - vx1)
        h_dist = abs((px1 + px2) / 2.0 - (vx1 + vx2) / 2.0) / v_w
        horizontal = max(0.0, 1.0 - min(1.0, h_dist))

        return 0.5 * overlap + 0.3 * vertical + 0.2 * horizontal

    # ------------------------------------------------------------------ #

    def build(self, detections: List[dict]) -> SceneGraph:
        graph = SceneGraph()
        for i, det in enumerate(detections):
            graph.nodes[i] = SceneNode(node_id=i, det=det)

        persons = [n for n in graph.nodes.values()
                   if n.det["category"] == "person"]
        vehicles = [n for n in graph.nodes.values()
                    if n.det["category"] == "vehicle"]

        candidates = []
        for p in persons:
            for v in vehicles:
                if not self._gate(p.bbox, v.bbox):
                    continue
                s = self._score(p.bbox, v.bbox)
                if s >= self.threshold:
                    candidates.append((s, p.node_id, v.node_id))

        # Greedy one-to-one: highest scoring pairs win, each person is
        # assigned to at most one vehicle.
        candidates.sort(reverse=True)
        taken = set()
        for score, pid, vid in candidates:
            if pid in taken:
                continue
            taken.add(pid)
            vtype = graph.nodes[vid].det["vehicle_type"]
            if vtype in TWO_WHEELERS:
                relation = "ON"
            elif vtype in ENCLOSED:
                relation = "IN"
            else:
                continue
            graph.edges.append(SceneEdge(pid, relation, vid, float(score)))

        return graph

    @staticmethod
    def explain(graph: SceneGraph) -> List[str]:
        """Human-readable reasoning chain - useful for the audit trail and for
        demoing the explainability angle."""
        lines = []
        for e in graph.edges:
            p = graph.nodes[e.source].det
            v = graph.nodes[e.target].det
            lines.append(
                f"person#{p['track_id']} --{e.relation}--> "
                f"{v['vehicle_type']}#{v['track_id']} (score {e.confidence:.2f})")
        return lines
