"""Asynchronous Telegram Alert Service for Traffic Police.

Dispatches instant violation alerts, e-challans, and photo evidence to a
designated Telegram chat / channel / group without impacting video FPS.
"""

from __future__ import annotations

import io
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.parse

try:
    from .config import AlertConfig
except ImportError:
    from core.config import AlertConfig

logger = logging.getLogger("trafficsentinel.alerts")


@dataclass
class AlertJob:
    vtype: str
    track_id: int
    severity: str
    confidence: float
    fine: int
    plate: Optional[str]
    camera_id: str
    lat: float
    lon: float
    reasons: List[str]
    evidence_path: Optional[str]
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class TelegramAlertService:
    """Non-blocking background worker for Telegram mobile alerts."""

    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self._queue: queue.Queue[Optional[AlertJob]] = queue.Queue(maxsize=100)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_alert_time: Dict[str, float] = {}

    def start(self) -> "TelegramAlertService":
        if self._running or not self.cfg.enabled or not self.cfg.bot_token or not self.cfg.chat_id:
            return self
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="TelegramAlerts")
        self._thread.start()
        logger.info("[alerts] Telegram alert service worker started")
        return self

    def stop(self) -> None:
        self._running = False
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

    def dispatch(self, job: AlertJob) -> bool:
        """Enqueue an alert job non-blockingly."""
        if not self.cfg.enabled or not self.cfg.bot_token or not self.cfg.chat_id:
            return False

        if job.confidence < self.cfg.min_confidence:
            return False

        # Anti-spam cooldown per track and violation type
        key = f"{job.track_id}:{job.vtype}"
        now = time.time()
        last = self._last_alert_time.get(key, 0.0)
        if now - last < self.cfg.cooldown_seconds:
            return False
        self._last_alert_time[key] = now

        # Ensure worker is alive
        if not self._thread or not self._thread.is_alive():
            self.start()

        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            logger.warning("[alerts] Alert queue full, dropping notification for track %d", job.track_id)
            return False

    def _worker_loop(self) -> None:
        while self._running:
            try:
                job = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if job is None:
                break

            try:
                self._send_telegram(job)
            except Exception as exc:
                logger.error("[alerts] Failed to send Telegram alert: %s", exc)
            finally:
                self._queue.task_done()

    def _format_caption(self, job: AlertJob) -> str:
        icons = {
            "OVERSPEEDING": "⚡",
            "RED_LIGHT": "🚦",
            "NO_HELMET": "🪖",
            "TRIPLE_RIDING": "👥",
            "WRONG_WAY": "⛔",
            "ILLEGAL_PARKING": "🅿️",
            "NEAR_MISS": "⚠️",
        }
        icon = icons.get(job.vtype, "🚨")
        date_str = time.strftime("%d-%b-%Y %H:%M:%S", time.localtime(job.timestamp))
        plate_str = f"<b>{job.plate}</b>" if job.plate else "<i>Unidentified</i>"

        gmaps_url = f"https://www.google.com/maps?q={job.lat},{job.lon}"
        reasons_text = "\n".join([f"• {r}" for r in job.reasons]) if job.reasons else "• Rule threshold exceeded"

        lines = [
            f"{icon} <b>TRAFFIC VIOLATION DETECTED</b> {icon}",
            "━━━━━━━━━━━━━━━━━━━━━",
            f"📋 <b>Type:</b> {job.vtype.replace('_', ' ')} ({job.severity.upper()})",
            f"🔢 <b>Vehicle Plate:</b> {plate_str}",
            f"💰 <b>Challan Fine:</b> ₹{job.fine:,}",
            f"🎯 <b>Detection Confidence:</b> {job.confidence * 100:.1f}%",
            f"📹 <b>Camera ID:</b> <code>{job.camera_id}</code>",
            f"📍 <b>Location:</b> <a href=\"{gmaps_url}\">{job.lat:.4f}° N, {job.lon:.4f}° E</a>",
            f"⏱️ <b>Time:</b> {date_str}",
            "━━━━━━━━━━━━━━━━━━━━━",
            f"<b>Details:</b>\n{reasons_text}",
            "━━━━━━━━━━━━━━━━━━━━━",
            "<i>TrafficSentinel AI Automated Enforcement</i>"
        ]
        return "\n".join(lines)

    def _send_telegram(self, job: AlertJob) -> bool:
        bot_token = self.cfg.bot_token.strip()
        chat_id = self.cfg.chat_id.strip()
        caption = self._format_caption(job)

        has_image = (self.cfg.send_photos and job.evidence_path
                     and os.path.exists(job.evidence_path))

        if has_image:
            return self._send_photo_request(bot_token, chat_id, job.evidence_path, caption)
        else:
            return self._send_message_request(bot_token, chat_id, caption)

    def _send_message_request(self, token: str, chat_id: str, text: str) -> bool:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            return resp.status == 200

    def _send_photo_request(self, token: str, chat_id: str, photo_path: str, caption: str) -> bool:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        boundary = f"----WebKitFormBoundary{int(time.time() * 1000)}"

        with open(photo_path, "rb") as f:
            photo_data = f.read()

        filename = Path(photo_path).name
        body = io.BytesIO()

        # chat_id
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="chat_id"\r\n\r\n')
        body.write(f"{chat_id}\r\n".encode())

        # parse_mode
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="parse_mode"\r\n\r\n')
        body.write(b"HTML\r\n")

        # caption
        body.write(f"--{boundary}\r\n".encode())
        body.write(b'Content-Disposition: form-data; name="caption"\r\n\r\n')
        body.write(f"{caption}\r\n".encode())

        # photo
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="photo"; filename="{filename}"\r\n'.encode())
        body.write(b"Content-Type: image/jpeg\r\n\r\n")
        body.write(photo_data)
        body.write(b"\r\n")

        body.write(f"--{boundary}--\r\n".encode())
        payload = body.getvalue()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            return resp.status == 200

    @classmethod
    def send_test_alert(cls, bot_token: str, chat_id: str) -> Tuple[bool, str]:
        """Synchronous helper for testing connection from Streamlit UI."""
        if not bot_token or not chat_id:
            return False, "Bot Token and Chat ID cannot be empty."

        text = (
            "🚦 <b>TrafficSentinel AI — Test Alert</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "✅ <b>Connection Status:</b> ACTIVE\n"
            "📱 <b>Police Alert Channel:</b> Connected\n"
            f"⏱️ <b>Timestamp:</b> {time.strftime('%d-%b-%Y %H:%M:%S')}\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "<i>Real-time violation alerts are configured and operational!</i>"
        )
        url = f"https://api.telegram.org/bot{bot_token.strip()}/sendMessage"
        payload = json.dumps({
            "chat_id": chat_id.strip(),
            "text": text,
            "parse_mode": "HTML"
        }).encode("utf-8")

        try:
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                if resp.status == 200:
                    return True, "Test message sent successfully! Check your Telegram."
                return False, f"Telegram API returned status code {resp.status}."
        except urllib.error.HTTPError as http_err:
            try:
                err_body = http_err.read().decode("utf-8")
                err_json = json.loads(err_body)
                desc = err_json.get("description", str(http_err))
                if "chat not found" in desc.lower() or "bot can't initiate" in desc.lower():
                    return False, f"Telegram Error: {desc}. Tip: Please open your bot in Telegram and press START first!"
                return False, f"Telegram Error: {desc}"
            except Exception:
                return False, f"Telegram HTTP Error {http_err.code}: {http_err.reason}"
        except Exception as exc:
            return False, f"Failed to send test alert: {exc}"
