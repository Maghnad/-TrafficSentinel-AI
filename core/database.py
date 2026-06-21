import sqlite3
import os
import cv2
import numpy as np
from datetime import datetime

class ViolationDatabase:
    def __init__(self, db_path="violations.db", evidence_dir="evidence"):
        self.db_path = db_path
        self.evidence_dir = evidence_dir
        
        # Ensure evidence directory exists
        if not os.path.exists(self.evidence_dir):
            os.makedirs(self.evidence_dir)
            
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database and create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    violation_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    frame_idx INTEGER,
                    violation_type TEXT,
                    severity TEXT,
                    confidence REAL,
                    fine_amount INTEGER,
                    plate_number TEXT,
                    evidence_path TEXT
                )
            ''')
            
            # New table for the ANPR tracking network
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sightings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_number TEXT,
                    timestamp TEXT,
                    camera_id TEXT,
                    latitude REAL,
                    longitude REAL,
                    is_violation BOOLEAN
                )
            ''')
            conn.commit()

    def violation_exists(self, violation_id: str) -> bool:
        """Check if a violation is already logged."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT 1 FROM violations WHERE violation_id = ?', (violation_id,))
            return cursor.fetchone() is not None

    def add_violation(self, violation_id, frame_idx, violation_type, severity, confidence, fine_amount, plate_number, frame, bbox):
        """Save evidence image and log violation to database."""
        if self.violation_exists(violation_id):
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        evidence_path = ""

        # Crop and save evidence image
        if frame is not None and bbox is not None:
            try:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                # Add a 20px margin around the crop for better context
                h, w = frame.shape[:2]
                x1 = max(0, x1 - 20)
                y1 = max(0, y1 - 20)
                x2 = min(w, x2 + 20)
                y2 = min(h, y2 + 20)
                
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    filename = f"{violation_id}.jpg"
                    filepath = os.path.join(self.evidence_dir, filename)
                    cv2.imwrite(filepath, crop)
                    evidence_path = filepath
            except Exception as e:
                print(f"Failed to save evidence image: {e}")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO violations 
                (violation_id, timestamp, frame_idx, violation_type, severity, confidence, fine_amount, plate_number, evidence_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (violation_id, timestamp, frame_idx, violation_type, severity, confidence, fine_amount, plate_number, evidence_path))
            conn.commit()

    def get_all_violations(self, limit=100):
        """Retrieve recent violations from the database."""
        with sqlite3.connect(self.db_path) as conn:
            # Return rows as dictionaries
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM violations 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_recent_violations(self, limit=10):
        """Alias for UI compatibility if needed."""
        return self.get_all_violations(limit)

    def clear_database(self):
        """Delete all records from the database and optionally clear the evidence directory."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM violations')
            cursor.execute('DELETE FROM sightings')
            conn.commit()
            
        # Also delete all images in the evidence directory
        if os.path.exists(self.evidence_dir):
            for file in os.listdir(self.evidence_dir):
                if file.endswith('.jpg'):
                    try:
                        os.remove(os.path.join(self.evidence_dir, file))
                    except Exception:
                        pass

    # --- ANPR TRACKING NETWORK METHODS ---
    
    def log_sighting(self, plate_number: str, camera_id: str, latitude: float, longitude: float, is_violation: bool = False):
        """Log a vehicle sighting into the tracking network database."""
        if not plate_number:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sightings 
                (plate_number, timestamp, camera_id, latitude, longitude, is_violation)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (plate_number, timestamp, camera_id, latitude, longitude, is_violation))
            conn.commit()

    def get_vehicle_route(self, plate_number: str):
        """Retrieve the chronological route of a specific vehicle."""
        clean_plate = plate_number.replace(" ", "").replace("-", "").upper()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Remove spaces and hyphens from the stored plate_number on the fly to match the clean_plate
            cursor.execute('''
                SELECT * FROM sightings 
                WHERE REPLACE(REPLACE(UPPER(plate_number), ' ', ''), '-', '') LIKE ? 
                ORDER BY timestamp ASC
            ''', (f"%{clean_plate}%",))
            return [dict(row) for row in cursor.fetchall()]

