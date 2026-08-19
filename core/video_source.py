"""Threaded frame reader.

Decoding a 720p H.264 stream costs 5-15 ms per frame and OpenCV's
VideoCapture.read() is blocking. Running it on the main loop means inference
and decode serialise. This class decodes on its own thread and always hands
back the *newest* frame, dropping stale ones. For live sources that is
correct behaviour: you want current reality, not a backlog.

For file sources set `realtime=False` so no frames are dropped (you want to
process the whole video, not simulate a live feed).
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class VideoSource:
    def __init__(self, src, realtime: bool = True, reconnect: bool = True):
        self.src = src
        self.realtime = realtime
        self.reconnect = reconnect

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_idx = -1
        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_served = -1

        self.fps = 25.0
        self.width = 0
        self.height = 0

    # ------------------------------------------------------------------ #

    def _open(self) -> bool:
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            return False
        # Keep the driver buffer tiny; we do our own dropping.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps and 1.0 < fps < 240.0:
            self.fps = float(fps)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap
        return True

    def start(self) -> "VideoSource":
        if not self._open():
            raise RuntimeError(f"Could not open video source: {self.src}")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        idx = 0
        while not self._stopped.is_set():
            if self._cap is None:
                if not (self.reconnect and self._open()):
                    time.sleep(0.5)
                    continue

            ok, frame = self._cap.read()
            if not ok:
                if self.realtime and self.reconnect:
                    self._cap.release()
                    self._cap = None
                    time.sleep(0.5)
                    continue
                self._stopped.set()
                break

            with self._lock:
                self._frame = frame
                self._frame_idx = idx
            idx += 1

            if not self.realtime:
                # File playback: block until the consumer has taken this frame,
                # otherwise we would race ahead and drop most of the video.
                while (not self._stopped.is_set()
                       and self._last_served < self._frame_idx):
                    time.sleep(0.001)

    # ------------------------------------------------------------------ #

    def read(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """Return (ok, frame, frame_index). Never blocks for long."""
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._frame_idx > self._last_served:
                    self._last_served = self._frame_idx
                    return True, self._frame.copy(), self._frame_idx
            if self._stopped.is_set():
                return False, None, -1
            time.sleep(0.002)
        return False, None, -1

    def stop(self) -> None:
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def running(self) -> bool:
        return not self._stopped.is_set()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
