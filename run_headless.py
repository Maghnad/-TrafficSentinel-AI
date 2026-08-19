"""Headless runner + benchmark.

Use this to measure real pipeline throughput. Streamlit's rerun loop caps the
displayed rate around 10-15 FPS, so benchmarking inside the dashboard tells you
about Streamlit, not about your pipeline.

    python run_headless.py --source traffic.mp4 --config camera.json
    python run_headless.py --source traffic.mp4 --benchmark 300 --no-display
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from core.config import AppConfig
from core.engine import TrafficSentinel
from core.video_source import VideoSource


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--config", default="camera.json")
    ap.add_argument("--benchmark", type=int, default=0,
                    help="stop after N frames and print a latency breakdown")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    src = int(args.source) if str(args.source).isdigit() else args.source
    realtime = not isinstance(src, str) or src.startswith("rtsp")

    cfg = AppConfig.load(args.config)
    video = VideoSource(src, realtime=realtime).start()
    engine = TrafficSentinel(cfg, fps=video.fps)

    print(f"[run] backend={engine.detector.backend} "
          f"imgsz={cfg.detector.imgsz} source_fps={video.fps:.1f} "
          f"calibrated={engine.ground.calibrated}")

    latencies, infer, n = [], [], 0
    t_start = time.time()
    try:
        while True:
            ok, frame, idx = video.read()
            if not ok:
                break
            res = engine.process(frame, idx, draw=not args.no_display)
            latencies.append(res["latency_ms"])
            infer.append(res["hud"]["infer_ms"])
            n += 1

            for v in res["new_violations"]:
                print(f"  [{v['status']:9s}] {v['type']:15s} "
                      f"track#{v['track_id']:<4d} conf={v['confidence']:.2f} "
                      f"plate={v['plate'] or '-'}")

            if not args.no_display:
                cv2.imshow("TrafficSentinel", res["frame"])
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.benchmark and n >= args.benchmark:
                break
    finally:
        video.stop()
        engine.close()
        cv2.destroyAllWindows()

    if n:
        wall = time.time() - t_start
        lat = np.asarray(latencies)
        print("\n--- benchmark ---------------------------------------")
        print(f"frames             : {n}")
        print(f"wall clock         : {wall:.1f} s  ({n / wall:.1f} FPS)")
        print(f"pipeline mean      : {lat.mean():.1f} ms")
        print(f"pipeline p50 / p95 : {np.percentile(lat, 50):.1f} / "
              f"{np.percentile(lat, 95):.1f} ms")
        print(f"inference mean     : {np.mean(infer):.1f} ms "
              f"({np.mean(infer) / lat.mean() * 100:.0f}% of budget)")
        print("-----------------------------------------------------")


if __name__ == "__main__":
    main()
