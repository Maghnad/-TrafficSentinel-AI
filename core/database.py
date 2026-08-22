"""SQLite persistence.

Writes happen on a background thread with a bounded queue, for the same reason
OCR does: a synchronous INSERT plus an fsync is 5-15 ms, and at 25 FPS with a
busy junction that is enough to visibly hitch the loop. WAL mode also lets the
dashboard read while the pipeline writes.

The `status` column implements the two-tier enforcement model: violations above
the auto-issue confidence threshold become challans, everything else lands in a
human review queue. That is how real enforcement systems work, and it is the
honest place to put an uncertain helmet call.
"""

from __future__ import annotations

import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS violations (
    violation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             REAL    NOT NULL,
    camera_id      TEXT    NOT NULL,
    track_id       INTEGER,
    violation_type TEXT    NOT NULL,
    severity       TEXT,
    confidence     REAL,
    fine_amount    INTEGER,
    plate_number   TEXT,
    plate_conf     REAL,
    status         TEXT    NOT NULL DEFAULT 'review',
    measured_value REAL,
    reasons        TEXT,
    evidence_path  TEXT,
    clip_path      TEXT,
    latitude       REAL,
    longitude      REAL
);
CREATE INDEX IF NOT EXISTS idx_v_type   ON violations(violation_type);
CREATE INDEX IF NOT EXISTS idx_v_plate  ON violations(plate_number);
CREATE INDEX IF NOT EXISTS idx_v_status ON violations(status);
CREATE INDEX IF NOT EXISTS idx_v_ts     ON violations(ts);

CREATE TABLE IF NOT EXISTS sightings (
    sighting_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    plate_number TEXT NOT NULL,
    plate_norm   TEXT NOT NULL,
    camera_id    TEXT,
    latitude     REAL,
    longitude    REAL,
    is_violation INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_s_norm ON sightings(plate_norm);

CREATE TABLE IF NOT EXISTS flow_stats (
    ts         REAL,
    camera_id  TEXT,
    vehicles   INTEGER,
    mean_speed REAL,
    queue_len  INTEGER
);
"""


def normalise_plate(text: Optional[str]) -> str:
    if not text:
        return ""
    return "".join(ch for ch in text.upper() if ch.isalnum())


class Database:
    def __init__(self, path: str, async_writes: bool = True):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=512)
        self._stop = threading.Event()
        self._thread = None
        if async_writes:
            self._thread = threading.Thread(target=self._writer, daemon=True)
            self._thread.start()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=5.0,
                              check_same_thread=False)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.row_factory = sqlite3.Row
        return con

    def _init_schema(self) -> None:
        con = self._connect()
        con.executescript(SCHEMA)
        con.commit()
        con.close()

    # ------------------------------------------------------------------ #

    def _writer(self) -> None:
        con = self._connect()
        pending: List[tuple] = []
        last_flush = time.time()
        while not self._stop.is_set() or not self._q.empty():
            try:
                pending.append(self._q.get(timeout=0.2))
            except queue.Empty:
                pass
            if pending and (len(pending) >= 16 or time.time() - last_flush > 0.5):
                for sql, params in pending:
                    try:
                        con.execute(sql, params)
                    except Exception as exc:
                        print(f"[db] write failed: {exc}")
                con.commit()
                pending.clear()
                last_flush = time.time()
        con.commit()
        con.close()

    def _submit(self, sql: str, params: tuple) -> None:
        try:
            self._q.put_nowait((sql, params))
        except queue.Full:
            pass  # analytics rows are droppable; never stall the pipeline

    # ------------------------------------------------------------------ #

    def log_violation(self, *, camera_id, track_id, vtype, severity,
                      confidence, fine, plate, plate_conf, status,
                      measured_value, reasons, evidence_path, clip_path,
                      lat, lon) -> None:
        self._submit(
            """INSERT INTO violations
               (ts, camera_id, track_id, violation_type, severity, confidence,
                fine_amount, plate_number, plate_conf, status, measured_value,
                reasons, evidence_path, clip_path, latitude, longitude)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (time.time(), camera_id, track_id, vtype, severity, confidence,
             fine, plate, plate_conf, status, measured_value,
             " | ".join(reasons or []), evidence_path, clip_path, lat, lon))

    def log_sighting(self, plate, camera_id, lat, lon, is_violation) -> None:
        if not plate:
            return
        self._submit(
            """INSERT INTO sightings
               (ts, plate_number, plate_norm, camera_id, latitude, longitude,
                is_violation) VALUES (?,?,?,?,?,?,?)""",
            (time.time(), plate, normalise_plate(plate), camera_id, lat, lon,
             int(bool(is_violation))))

    def log_flow(self, camera_id, vehicles, mean_speed, queue_len) -> None:
        self._submit(
            "INSERT INTO flow_stats VALUES (?,?,?,?,?)",
            (time.time(), camera_id, vehicles, mean_speed, queue_len))

    def update_violation_plate(self, track_id: int, plate: str, plate_conf: float) -> None:
        """Retroactively updates plate number on previously logged violations for this track."""
        if not plate:
            return
        self._submit(
            """UPDATE violations
               SET plate_number = ?, plate_conf = ?
               WHERE track_id = ? AND (plate_number IS NULL OR plate_number = '' OR plate_conf < ?)""",
            (plate, plate_conf, track_id, plate_conf))

    # ------------------------------------------------------------------ #
    # Reads (synchronous - dashboard side)
    # ------------------------------------------------------------------ #

    def query(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        con = self._connect()
        try:
            return con.execute(sql, params).fetchall()
        finally:
            con.close()

    def recent(self, limit: int = 200, status: Optional[str] = None):
        if status:
            return self.query(
                "SELECT * FROM violations WHERE status=? "
                "ORDER BY ts DESC LIMIT ?", (status, limit))
        return self.query(
            "SELECT * FROM violations ORDER BY ts DESC LIMIT ?", (limit,))

    def vehicle_route(self, plate: str):
        return self.query(
            "SELECT * FROM sightings WHERE plate_norm=? ORDER BY ts",
            (normalise_plate(plate),))

    def set_status(self, violation_id: int, status: str) -> None:
        con = self._connect()
        con.execute("UPDATE violations SET status=? WHERE violation_id=?",
                    (status, violation_id))
        con.commit()
        con.close()

    def stats(self):
        return {
            "total": self.query("SELECT COUNT(*) c FROM violations")[0]["c"],
            "issued": self.query(
                "SELECT COUNT(*) c FROM violations WHERE status='issued'"
            )[0]["c"],
            "review": self.query(
                "SELECT COUNT(*) c FROM violations WHERE status='review'"
            )[0]["c"],
            "fines": self.query(
                "SELECT COALESCE(SUM(fine_amount),0) s FROM violations "
                "WHERE status='issued'")[0]["s"],
        }

    def clear_all(self, clear_evidence: bool = True) -> None:
        """Delete all violation logs, sightings, and flow statistics."""
        con = self._connect()
        con.execute("DELETE FROM violations")
        con.execute("DELETE FROM sightings")
        con.execute("DELETE FROM flow_stats")
        con.commit()
        con.close()

        if clear_evidence:
            for folder in ("evidence/crops", "evidence/clips"):
                p = Path(folder)
                if p.exists():
                    for f in p.glob("*.*"):
                        try:
                            f.unlink()
                        except Exception:
                            pass

    def close(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

