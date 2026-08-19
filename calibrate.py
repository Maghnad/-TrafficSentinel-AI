"""Per-camera calibration tool.

Without this the system cannot legitimately claim to measure speed, direction
or dwell time. It takes about two minutes per camera and it is the difference
between "Rs 2000 fine backed by a measurement" and "Rs 2000 fine backed by a
pixel count".

    python calibrate.py --source traffic.mp4 --out camera.json

Steps, in order:
  1. HOMOGRAPHY - click 4 points on the ROAD SURFACE forming a rectangle whose
     real dimensions you know (lane markings are ideal: standard Indian lane
     width is 3.5 m, and dashed centre-line segments are 3 m long with 6 m
     gaps). Then enter the real width and length in metres.
  2. STOP LINE - click 2 points spanning the stop line for the approach you
     are monitoring.
  3. LANE DIRECTION - click 2 points along the LEGAL direction of travel
     (from, then to). Used for wrong-way detection.
  4. NO-PARKING ZONES - click polygon vertices, press 'n' to close a polygon,
     'q' when done.

Keys: u = undo last point, s = skip current step, q = finish step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

STEPS = ["homography", "stop_line", "lane_direction", "no_parking"]


class Picker:
    def __init__(self, frame):
        self.base = frame
        self.points = []
        self.polygons = []
        self.step = 0

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([int(x), int(y)])

    def render(self, prompt):
        img = self.base.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 52), (25, 25, 30), -1)
        cv2.putText(img, prompt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(img, "u=undo  n=next polygon  s=skip  q=done",
                    (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (160, 200, 160), 1, cv2.LINE_AA)

        for poly in self.polygons:
            cv2.polylines(img, [np.asarray(poly, np.int32).reshape(-1, 1, 2)],
                          True, (90, 90, 220), 2)
        for i, p in enumerate(self.points):
            cv2.circle(img, tuple(p), 5, (60, 220, 250), -1)
            cv2.putText(img, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 250), 1)
        if len(self.points) > 1:
            cv2.polylines(img, [np.asarray(self.points, np.int32).reshape(-1, 1, 2)],
                          False, (60, 220, 250), 1)
        return img


def collect(frame, prompt, need=None, allow_polygons=False):
    picker = Picker(frame)
    cv2.namedWindow("calibrate", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("calibrate", picker.on_mouse)
    while True:
        cv2.imshow("calibrate", picker.render(prompt))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and picker.points:
            picker.points.pop()
        elif key == ord("n") and allow_polygons and len(picker.points) >= 3:
            picker.polygons.append(picker.points)
            picker.points = []
        elif key in (ord("q"), 13):
            break
        elif key == ord("s"):
            picker.points, picker.polygons = [], []
            break
        if need and len(picker.points) >= need:
            cv2.imshow("calibrate", picker.render(prompt))
            cv2.waitKey(400)
            break
    if allow_polygons and len(picker.points) >= 3:
        picker.polygons.append(picker.points)
    return picker.points, picker.polygons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="camera.json")
    ap.add_argument("--camera-id", default="CAM-01")
    ap.add_argument("--lat", type=float, default=22.5726)
    ap.add_argument("--lon", type=float, default=88.3639)
    args = ap.parse_args()

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read a frame from {args.source}")

    geom = {
        "camera_id": args.camera_id,
        "latitude": args.lat,
        "longitude": args.lon,
        "homography_src": [],
        "homography_dst": [],
        "stop_line": [],
        "lane_direction": [0.0, -1.0],
        "no_parking_zones": [],
        "parking_dwell_seconds": 20.0,
    }

    # 1. Homography
    pts, _ = collect(frame, "STEP 1/4  Click 4 ROAD-SURFACE points "
                            "(near-left, near-right, far-right, far-left)",
                     need=4)
    if len(pts) == 4:
        print("\nEnter the real-world size of the quadrilateral you clicked.")
        width = float(input("  width in metres (near edge, left->right): "))
        length = float(input("  length in metres (near edge -> far edge): "))
        geom["homography_src"] = pts
        # World frame: x across the road, y along it (positive = away).
        geom["homography_dst"] = [[0.0, 0.0], [width, 0.0],
                                  [width, length], [0.0, length]]

    # 2. Stop line
    pts, _ = collect(frame, "STEP 2/4  Click 2 points spanning the STOP LINE",
                     need=2)
    if len(pts) == 2:
        geom["stop_line"] = pts

    # 3. Lane direction
    pts, _ = collect(frame, "STEP 3/4  Click 2 points along LEGAL travel "
                            "direction (from -> to)", need=2)
    if len(pts) == 2 and geom["homography_src"]:
        H = cv2.getPerspectiveTransform(
            np.asarray(geom["homography_src"], np.float32),
            np.asarray(geom["homography_dst"], np.float32))
        w = cv2.perspectiveTransform(
            np.asarray([pts], np.float32), H)[0]
        v = w[1] - w[0]
        n = float(np.linalg.norm(v))
        if n > 1e-6:
            geom["lane_direction"] = [float(v[0] / n), float(v[1] / n)]

    # 4. No-parking polygons
    _, polys = collect(frame, "STEP 4/4  Click NO-PARKING polygon vertices "
                              "(n = close polygon, q = done)",
                       allow_polygons=True)
    geom["no_parking_zones"] = polys

    cv2.destroyAllWindows()

    out = Path(args.out)
    existing = json.loads(out.read_text()) if out.exists() else {}
    existing["geometry"] = geom
    out.write_text(json.dumps(existing, indent=2))

    homo_msg = ("yes" if geom["homography_src"]
                else "NO - speed, wrong-way and near-miss stay disabled")
    line_msg = ("yes" if geom["stop_line"]
                else "NO - red-light detection stays disabled")
    print("\nSaved -> {}".format(out))
    print("  homography : {}".format(homo_msg))
    print("  stop line  : {}".format(line_msg))
    print("  zones      : {}".format(len(geom["no_parking_zones"])))


if __name__ == "__main__":
    main()
