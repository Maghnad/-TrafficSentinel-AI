"""Evidence capture.

Saves high-resolution JPEG crops and web-compatible H.264 video clips spanning
the moments of the violation for legally verifiable proof.

Both crops and clips are written on a background worker thread with a bounded
queue to prevent any stutter on the main real-time detection pipeline.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False


class EvidenceWriter:
    def __init__(self, cfg, fps: float, camera_id: str):
        self.cfg = cfg
        self.fps = max(1.0, float(fps))
        self.camera_id = camera_id
        self.root = Path(cfg.root)
        (self.root / "crops").mkdir(parents=True, exist_ok=True)
        (self.root / "clips").mkdir(parents=True, exist_ok=True)

        # Buffer holding up to 3 seconds of recent video frames
        maxlen = int(self.fps * max(2.0, cfg.clip_seconds_before + 1.0))
        self._ring: Deque[np.ndarray] = deque(maxlen=max(25, maxlen))
        self._ring_lock = threading.Lock()

        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ #

    def push_frame(self, frame: np.ndarray) -> None:
        """Call once per processed frame."""
        with self._ring_lock:
            # Store a compact copy for evidence buffer
            self._ring.append(frame.copy())

    # ------------------------------------------------------------------ #

    def capture(self, frame: np.ndarray, bbox, vtype: str,
                track_id: int) -> Tuple[str, Optional[str]]:
        """Returns (crop_path, clip_path). Written asynchronously."""
        stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%1000:03d}"
        name = f"{self.camera_id}_{vtype}_{track_id}_{stamp}"
        crop_path = str(self.root / "crops" / f"{name}.jpg")
        clip_path = (str(self.root / "clips" / f"{name}.mp4")
                     if self.cfg.save_clips else None)

        with self._ring_lock:
            pre_frames = list(self._ring)

        if not pre_frames:
            pre_frames = [frame]

        try:
            self._q.put_nowait(("write", frame.copy(), bbox, crop_path, clip_path, pre_frames))
        except queue.Full:
            pass

        return crop_path, clip_path

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            _, frame, bbox, crop_path, clip_path, pre_frames = job
            try:
                self._write_crop(frame, bbox, crop_path)
                if clip_path and pre_frames:
                    self._write_clip(clip_path, pre_frames)
            except Exception as exc:
                print(f"[evidence] error writing evidence: {exc}")

    def _write_crop(self, frame: np.ndarray, bbox, path: str) -> None:
        h, w = frame.shape[:2]
        m = self.cfg.crop_margin_px
        x1 = max(0, int(bbox[0]) - m)
        y1 = max(0, int(bbox[1]) - m)
        x2 = min(w, int(bbox[2]) + m)
        y2 = min(h, int(bbox[3]) + m)
        if x2 <= x1 or y2 <= y1:
            return
        cv2.imwrite(path, frame[y1:y2, x1:x2],
                    [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality])

    def _write_clip(self, path: str, frames: List[np.ndarray]) -> None:
        """Write self-contained browser-playable H.264 video clip."""
        if not frames:
            return
        
        # Method 1: Use imageio with libx264 (Native HTML5 Browser Playable)
        if HAS_IMAGEIO:
            try:
                writer = imageio.get_writer(
                    path,
                    fps=self.fps,
                    codec="libx264",
                    format="FFMPEG",
                    quality=6,
                    pixelformat="yuv420p"
                )
                for f in frames:
                    # Convert BGR to RGB for browser
                    writer.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                writer.close()
                return
            except Exception as e:
                print(f"[evidence] imageio failed, falling back to cv2: {e}")

        # Method 2: Fallback to OpenCV VideoWriter
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, (w, h))
        if writer.isOpened():
            for f in frames:
                writer.write(f)
            writer.release()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
