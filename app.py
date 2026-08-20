"""TrafficSentinel AI - dashboard.

Note on frame rate: Streamlit's rerun model caps the *displayed* rate at
roughly 10-15 FPS regardless of how fast the pipeline runs. That is a UI
limit, not a pipeline limit, and the HUD reports true pipeline FPS separately
so the two are never confused. For a genuinely real-time viewer, run
`python run_headless.py` (OpenCV window) or move to streamlit-webrtc.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core.config import AppConfig, VIOLATION_META
from core.database import Database
from core.engine import TrafficSentinel
from core.video_source import VideoSource

st.set_page_config(page_title="TrafficSentinel AI", page_icon="🚦",
                   layout="wide")

CONFIG_PATH = "camera.json"


# ---------------------------------------------------------------------- #
# Resources
# ---------------------------------------------------------------------- #

@st.cache_resource(show_spinner="Loading models (first run exports to "
                                "OpenVINO, ~30s)...")
def get_engine(cfg_signature: str, fps: float) -> TrafficSentinel:
    cfg = AppConfig.load(CONFIG_PATH)
    return TrafficSentinel(cfg, fps=fps)


@st.cache_resource
def get_db(path: str) -> Database:
    return Database(path, async_writes=False)


cfg = AppConfig.load(CONFIG_PATH)
db = get_db(cfg.db_path)


# ---------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------- #

st.sidebar.title("🚦 TrafficSentinel AI")
page = st.sidebar.radio("View", [
    "Live Detection", "Review Queue", "Analytics", "ANPR Trace",
    "Calibration Status",
])

st.sidebar.divider()
st.sidebar.caption("📍 Camera Checkpoint & Location")
cam_preset = st.sidebar.selectbox("Camera Node", [
    "CAM-01 (Park Street Junction)",
    "CAM-02 (Howrah Bridge Approach)",
    "CAM-03 (Salt Lake Sector V)",
    "CAM-04 (Airport Expressway)",
    "Custom Camera Node"
])

CAMERA_PRESETS = {
    "CAM-01 (Park Street Junction)": ("CAM-01", 22.5526, 88.3539),
    "CAM-02 (Howrah Bridge Approach)": ("CAM-02", 22.5850, 88.3426),
    "CAM-03 (Salt Lake Sector V)": ("CAM-03", 22.5735, 88.4331),
    "CAM-04 (Airport Expressway)": ("CAM-04", 22.6450, 88.4410),
}

if cam_preset == "Custom Camera Node":
    cfg.geometry.camera_id = st.sidebar.text_input("Camera ID", cfg.geometry.camera_id)
    c_lat, c_lon = st.sidebar.columns(2)
    cfg.geometry.latitude = c_lat.number_input("Latitude", value=float(cfg.geometry.latitude), format="%.4f")
    cfg.geometry.longitude = c_lon.number_input("Longitude", value=float(cfg.geometry.longitude), format="%.4f")
else:
    cid, lat, lon = CAMERA_PRESETS[cam_preset]
    cfg.geometry.camera_id = cid
    cfg.geometry.latitude = lat
    cfg.geometry.longitude = lon
    st.sidebar.caption(f"ID: `{cid}` | GPS: `{lat:.4f}, {lon:.4f}`")

st.sidebar.divider()
st.sidebar.caption("Detection")
cfg.detector.imgsz = st.sidebar.select_slider(
    "Inference size", [320, 416, 480, 544, 640], value=cfg.detector.imgsz,
    help="480 is ~1.7x faster than 640 with minimal recall loss on "
         "vehicle-sized objects.")
cfg.detector.conf = st.sidebar.slider("Confidence", 0.15, 0.8,
                                      cfg.detector.conf, 0.05)
cfg.rules.speed_limit_kmh = st.sidebar.number_input(
    "Speed limit (km/h)", 10.0, 150.0, cfg.rules.speed_limit_kmh, 5.0)
cfg.rules.auto_issue_confidence = st.sidebar.slider(
    "Auto-issue threshold", 0.5, 0.99, cfg.rules.auto_issue_confidence, 0.01,
    help="Violations below this go to the human review queue instead of "
         "becoming challans.")
cfg.ocr.enabled = st.sidebar.checkbox("ANPR (plate reading)", cfg.ocr.enabled)
show_clean = st.sidebar.checkbox("Show non-violating vehicles + ANPR Plates", True)

st.sidebar.caption("Active rules")
enabled_rules = {}
for vtype in VIOLATION_META:
    enabled_rules[vtype] = st.sidebar.checkbox(
        vtype.replace("_", " ").title(), True, key=f"rule_{vtype}")


# ---------------------------------------------------------------------- #
# Live detection
# ---------------------------------------------------------------------- #

def page_live():
    st.header("Live Detection")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        source_kind = st.radio("Source", ["Upload video", "Webcam", "RTSP"],
                               horizontal=True)
    with col_b:
        run = st.toggle("Run", value=False)

    source = None
    realtime = True
    if source_kind == "Upload video":
        upload = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
        if upload:
            tmp = Path("_upload.mp4")
            tmp.write_bytes(upload.read())
            source, realtime = str(tmp), False
    elif source_kind == "Webcam":
        source = st.number_input("Device index", 0, 8, 0)
    else:
        source = st.text_input("RTSP URL", "rtsp://user:pass@192.168.1.10:554/stream")

    if not run or source is None:
        st.info("Select a source and switch **Run** on.")
        return

    video = VideoSource(source, realtime=realtime).start()
    engine = get_engine(str(cfg.detector.imgsz), video.fps)
    engine.cfg = cfg
    engine.annotator.show_clean = show_clean
    for vtype, on in enabled_rules.items():
        engine.rules.set_enabled(vtype, on)

    frame_slot = st.empty()
    metric_slot = st.empty()
    feed_slot = st.empty()

    try:
        while run:
            ok, frame, idx = video.read()
            if not ok:
                st.success("Stream ended.")
                break

            result = engine.process(frame, idx, draw=True)
            hud = result["hud"]

            frame_slot.image(cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB),
                             use_container_width=True)

            m1, m2, m3, m4, m5 = metric_slot.columns(5)
            m1.metric("Pipeline FPS", f"{hud['fps']:.1f}")
            m2.metric("Inference", f"{hud['infer_ms']:.0f} ms")
            m3.metric("OCR backlog", hud["ocr_backlog"])
            m4.metric("Tracks", hud["tracks"])
            m5.metric("Signal", hud["light"])

            if show_clean:
                # Show ALL active vehicles (Non-Violating + Violating) with real-time ANPR plates
                rows = []
                for tr in list(engine.registry.tracks.values()):
                    is_offender = bool(tr.logged)
                    v_type = tr.vehicle_type or tr.label
                    speed_str = f"{tr.speed_kmh:.0f} km/h" if tr.speed_kmh is not None else "-"
                    plate_str = tr.plate if tr.plate else ("Reading OCR..." if tr.ocr_attempts > 0 else "Pending")
                    conf_str = f"{tr.plate_conf:.0%}" if tr.plate else "-"
                    status_str = "🚨 VIOLATION (" + ", ".join(list(tr.logged)) + ")" if is_offender else "✅ CLEAN (Non-Violating)"

                    rows.append({
                        "Track ID": f"#{tr.track_id}",
                        "Vehicle": v_type,
                        "Speed": speed_str,
                        "Detected Plate (ANPR)": plate_str,
                        "OCR Conf": conf_str,
                        "Vehicle Status": status_str,
                    })
                if rows:
                    feed_slot.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            elif engine.live_violations:
                rows = []
                for v in list(engine.live_violations)[:12]:
                    rows.append({
                        "Type": v["type"],
                        "Track": v["track_id"],
                        "Plate": v["plate"] or "-",
                        "Conf": f"{v['confidence']:.2f}",
                        "Status": v["status"],
                        "Fine": f"Rs {v['fine']}" if v["fine"] else "-",
                        "Why": " | ".join(v["reasons"][:2]),
                    })
                feed_slot.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    finally:
        video.stop()


# ---------------------------------------------------------------------- #
# Review queue - the two-tier enforcement model
# ---------------------------------------------------------------------- #

def page_review():
    head_col1, head_col2 = st.columns([3, 1])
    with head_col1:
        st.header("Review Queue")
        st.caption("Violations awaiting officer confirmation or dismissal before issuing challans.")
    with head_col2:
        st.write("")
        with st.popover("🗑️ Clear All Logs", use_container_width=True):
            st.warning("This will delete all violation logs, ANPR sightings, and evidence clips.")
            if st.button("Confirm Delete All", type="primary", use_container_width=True):
                st.cache_resource.clear()
                db_inst = Database(cfg.db_path, async_writes=False)
                db_inst.clear_all(clear_evidence=True)
                st.success("All logs and evidence deleted successfully!")
                st.rerun()

    rows = db.recent(100, status="review")
    if not rows:
        st.info("No violations currently awaiting review.")
        return

    for r in rows:
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                path = r["evidence_path"]
                if path and Path(path).exists():
                    st.image(str(path), caption="Evidence Snapshot", use_container_width=True)
                clip = r["clip_path"]
                if clip and Path(clip).exists() and Path(clip).stat().st_size > 500:
                    try:
                        with open(clip, "rb") as vf:
                            video_bytes = vf.read()
                        st.video(video_bytes, format="video/mp4")
                    except Exception as exc:
                        st.caption(f"Error loading clip: {exc}")
            with c2:
                st.subheader(f"{r['violation_type']}  ·  "
                             f"track #{r['track_id']}")
                st.write(f"**Plate:** {r['plate_number'] or 'not read'}   "
                         f"**Confidence:** {r['confidence']:.2f}   "
                         f"**Fine:** Rs {r['fine_amount']}")
                st.write("**Reasoning chain**")
                for reason in (r["reasons"] or "").split(" | "):
                    if reason:
                        st.write(f"- {reason}")
                b1, b2 = st.columns(2)
                if b1.button("Issue challan", key=f"i{r['violation_id']}"):
                    db.set_status(r["violation_id"], "issued")
                    st.rerun()
                if b2.button("Dismiss", key=f"d{r['violation_id']}"):
                    db.set_status(r["violation_id"], "dismissed")
                    st.rerun()


# ---------------------------------------------------------------------- #

def page_analytics():
    st.header("Analytics")
    stats = db.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total detections", stats["total"])
    c2.metric("Challans issued", stats["issued"])
    c3.metric("Awaiting review", stats["review"])
    c4.metric("Fines issued", f"Rs {stats['fines']:,}")

    rows = db.query("SELECT violation_type, COUNT(*) c FROM violations "
                    "GROUP BY violation_type ORDER BY c DESC")
    if rows:
        import plotly.express as px
        df = pd.DataFrame([dict(r) for r in rows])
        st.plotly_chart(px.bar(df, x="violation_type", y="c",
                               title="Violations by type"),
                        use_container_width=True)

    flow = db.query("SELECT ts, vehicles, mean_speed, queue_len "
                    "FROM flow_stats ORDER BY ts DESC LIMIT 500")
    if flow:
        import plotly.express as px
        df = pd.DataFrame([dict(r) for r in flow])
        df["time"] = pd.to_datetime(df["ts"], unit="s")
        st.subheader("Traffic flow")
        st.caption("Queue length and mean speed per junction over time - "
                   "arguably more useful to a city than the fines, and this "
                   "data largely does not exist today.")
        st.plotly_chart(px.line(df, x="time", y=["mean_speed", "queue_len"]),
                        use_container_width=True)

    near = db.query("SELECT COUNT(*) c FROM violations "
                    "WHERE violation_type='NEAR_MISS'")[0]["c"]
    st.metric("Near-miss conflicts recorded", near,
              help="Collisions that did not happen. High counts flag junctions "
                   "that need redesign, not enforcement.")


def page_anpr():
    st.header("ANPR Trace & Multi-Camera Trajectory")
    st.caption("Track vehicles across city-wide camera checkpoints in real-time.")

    # Fetch recent plates for quick selection
    recent_sightings = db.query("SELECT DISTINCT plate_number FROM sightings ORDER BY ts DESC LIMIT 15")
    recent_plates = [r["plate_number"] for r in recent_sightings if r["plate_number"]]

    c1, c2 = st.columns([2, 1])
    with c1:
        plate = st.text_input("Search Plate Number", placeholder="e.g. DL3CCP6535 or HR51BF1188")
    with c2:
        if recent_plates:
            selected_recent = st.selectbox("Or Pick Recently Detected Vehicle", ["-- Select --"] + recent_plates)
            if selected_recent != "-- Select --":
                plate = selected_recent

    if not plate:
        st.info("Enter or select a plate number to trace its city trajectory.")
        return

    rows = db.vehicle_route(plate)
    if not rows:
        st.warning(f"No sightings recorded for plate `{plate}`.")
        return

    df = pd.DataFrame([dict(r) for r in rows])
    df["time"] = pd.to_datetime(df["ts"], unit="s")

    # Clean multi-camera passage aggregation:
    # Group consecutive sightings from the same camera to avoid spamming 20 rows for a single 5s video pass
    passages = []
    current_pass = None
    for _, row in df.iterrows():
        if current_pass is None or current_pass["camera_id"] != row["camera_id"] or (row["ts"] - current_pass["last_ts"]) > 30:
            if current_pass is not None:
                passages.append(current_pass)
            current_pass = {
                "checkpoint": len(passages) + 1,
                "camera_id": row["camera_id"],
                "time": row["time"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "is_violation": int(row["is_violation"]),
                "last_ts": row["ts"]
            }
        else:
            current_pass["last_ts"] = row["ts"]
            if row["is_violation"]:
                current_pass["is_violation"] = 1
    if current_pass is not None:
        passages.append(current_pass)

    df_pass = pd.DataFrame(passages)

    st.subheader(f"📍 Vehicle Journey Checkpoints: `{plate}` ({len(df_pass)} Distinct Checkpoints)")
    st.dataframe(df_pass[["checkpoint", "time", "camera_id", "latitude", "longitude", "is_violation"]],
                 use_container_width=True, hide_index=True)

    try:
        import folium
        from streamlit_folium import st_folium
        m = folium.Map(location=[df_pass.latitude.mean(), df_pass.longitude.mean()], zoom_start=12)
        pts = list(zip(df_pass.latitude, df_pass.longitude))
        if len(pts) > 1:
            folium.PolyLine(pts, color="#ff4444", weight=4, opacity=0.8, dash_array="6").add_to(m)

        for i, row in df_pass.iterrows():
            marker_color = "red" if row["is_violation"] else "blue"
            popup_text = f"<b>Checkpoint {i+1}</b><br>Camera: {row['camera_id']}<br>Time: {row['time']}<br>Violation: {'YES' if row['is_violation'] else 'NO'}"
            folium.Marker(
                [row.latitude, row.longitude],
                popup=popup_text,
                tooltip=f"{row['camera_id']} ({row['time'].strftime('%H:%M:%S')})",
                icon=folium.Icon(color=marker_color, icon="car", prefix="fa")
            ).add_to(m)

        st_folium(m, height=450, use_container_width=True)
    except Exception:
        st.map(df.rename(columns={"latitude": "lat", "longitude": "lon"}))


def page_calibration():
    st.header("Calibration Status")
    g = cfg.geometry
    checks = [
        ("Homography (ground plane)", bool(g.homography_src),
         "Required for speed, wrong-way and near-miss. Without it a scalar "
         "pixels-per-metre would misreport far vehicles by 3-5x."),
        ("Stop line", bool(g.stop_line),
         "Required for red-light violations. Without it the system cannot "
         "distinguish a car crossing on red from one correctly stopped."),
        ("Lane direction", g.lane_direction != [0.0, -1.0],
         "Required for wrong-way detection."),
        ("No-parking zones", bool(g.no_parking_zones),
         "Required for parking violations."),
    ]
    for name, ok, why in checks:
        st.write(f"{'✅' if ok else '⛔'} **{name}**")
        st.caption(why)
    st.divider()
    st.code("python calibrate.py --source traffic.mp4 --out camera.json",
            language="bash")
    st.caption("Rules whose calibration is missing are disabled rather than "
               "approximated. A rule that guesses is worse than one that "
               "abstains.")


PAGES = {
    "Live Detection": page_live,
    "Review Queue": page_review,
    "Analytics": page_analytics,
    "ANPR Trace": page_anpr,
    "Calibration Status": page_calibration,
}
PAGES[page]()









