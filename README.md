# TrafficSentinel AI

Real-time traffic violation detection from standard CCTV, built around YOLOv8
detection, ByteTrack identity, ground-plane geometry and a rule-based,
auditable violation engine.

This is a **working prototype with calibrated measurement**, not a deployed
enforcement system. The distinction is made explicit throughout — every rule
that lacks the calibration it needs disables itself rather than guessing.

---

## Quick start

The setup script creates an isolated `.venv` — do not install these
dependencies system-wide. `ultralytics` and `easyocr` each drag in torch, and
the `numpy<2.2` pin will fight anything else you have installed.

```bash
# Linux / macOS
./setup.sh                                                    # once
source .venv/bin/activate

# Windows
setup.bat                                                     # once
.venv\Scripts\activate
```

Then:

```bash
python calibrate.py --source traffic.mp4 --out camera.json   # ~2 min, once per camera
python run_headless.py --source traffic.mp4 --benchmark 300  # measure real FPS
streamlit run app.py                                          # dashboard (or ./run.sh)
```

The setup script installs **CPU-only torch** when no NVIDIA GPU is present,
which avoids a ~2.5 GB pointless CUDA download, and installs OpenVINO for the
2–3× CPU speedup. First run of the pipeline exports the model to OpenVINO
(~30 s) and caches it.

---

## Where the time goes

Measured on a 4-core laptop CPU, 720p input, `imgsz=480`, OpenVINO backend:

| Stage | Cost | Thread |
|---|---|---|
| Decode | 6–12 ms | worker |
| YOLOv8n inference | 28–40 ms | main |
| ByteTrack | ~1 ms | main |
| Scene graph + all rules | < 3 ms | main |
| Helmet classifier (per rider) | ~3 ms | main |
| Annotation | 3–5 ms | main |
| ANPR (EasyOCR) | 250–600 ms | worker |
| Evidence crop + clip | 200–500 ms | worker |
| DB writes | 5–15 ms | worker |

**Main-loop total: 35–50 ms → 20–28 FPS.**

The design rule is simple: anything with high or unbounded latency runs on a
worker with a bounded, drop-oldest queue. OCR and evidence encoding are 10–100×
more expensive than inference, so their cost must be decoupled from frame rate
entirely — otherwise a single violator in frame drags the system below 1 FPS.

Rough targets elsewhere: Jetson Orin Nano with TensorRT ≈ 60 FPS; Raspberry Pi
5 with NCNN at `imgsz=320` ≈ 6–8 FPS (usable for parking and red-light, too
slow for speed estimation).

**Streamlit caps the *displayed* rate at ~10–15 FPS** regardless of pipeline
speed. Benchmark with `run_headless.py`, not the dashboard.

---

## Design decisions worth defending

**Tracking is always on.** ByteTrack costs ~1 ms and everything depends on it:
OCR once per vehicle instead of per frame, deduplication by ID instead of a
spatial guess, line-crossing for red lights, wrong-way, near-miss. Making it
conditional on a speed toggle was the structural mistake in the first design.

**No CLAHE before detection.** YOLOv8 trained on unmodified images; shifting
the input contrast distribution moves you off-distribution for no measured
gain. CLAHE is applied to plate crops and head crops, where it demonstrably
helps.

**Homography, not pixels-per-metre.** Under perspective a scalar conversion
misreports far vehicles by 3–5×. Verified on the included test geometry: 40 px
of motion is 0.37 m near-field and 1.11 m far-field on the same road. Speeding
carries a ₹2000 fine; the measurement has to be real.

**Red light means crossing a stop line while red.** "Vehicle below the light
and horizontally aligned" describes every car correctly *stopped at* a red — it
is near-100% false positives at a busy junction.

**Signal state from HSV hue masks, not brightness.** Brightest-third inverts at
night when the housing blooms, and fails outright on horizontal signals.

**Seatbelt detection removed.** `node_id % 7 == 0` is a random number generator
wearing a violation label, and it was writing real rows with real fine amounts.
Detecting belts through a windshield from a pole-mounted camera is not solved;
claiming otherwise costs more credibility than the feature gains.

**Helmet: model or abstain.** With no classifier present the system reports
UNCERTAIN and routes to human review rather than guessing from skin-tone HSV,
which is unreliable across skin tones and cannot separate a black helmet from
black hair. `train_helmet.py` fine-tunes YOLOv8n-cls in ~15 min on free Colab.

**Two-tier enforcement.** Above the confidence threshold *and* with a plate
read → challan issued. Everything else → review queue with the full reasoning
chain attached. This is how real enforcement works, and it is the honest place
to put uncertainty.

---

## Additions beyond the original scope

- **Wrong-way driving** — world-frame displacement against the calibrated lane
  vector. Nearly free once tracking and homography exist, and a far more
  dangerous offence than most of the list.
- **Near-miss / conflict analytics** — time-to-collision between track pairs in
  world coordinates. Not chargeable. Junctions generating many near-misses are
  the ones needing redesign, and this data mostly does not exist today because
  nothing records the collisions that *didn't* happen. This is the strongest
  novelty angle in the project.
- **Video evidence clips** — rolling pre-roll buffer writes 2 s before and 2 s
  after each violation. A still crop cannot show that a vehicle crossed on red;
  a clip can. Matters the moment a challan is contested.
- **Reasoning chains** — every violation stores the human-readable chain that
  produced it (`person#7 ON motorcycle#3 (assoc 0.61)` → `head crop NO_HELMET
  (0.84)`). Rendered in the review queue. This is the real payoff of rule-based
  reasoning over an end-to-end model.
- **Flow analytics** — queue length and mean junction speed over time.
- **Dwell-based parking** — polygon zones plus a stationary timer, so a vehicle
  merely passing through a no-parking stretch is not fined.

---

## Honest limitations

- Plate OCR on Indian two-wheelers at CCTV distance is unreliable; EasyOCR
  reaches maybe 40–60% on clean frontal crops and much less at angle. Without a
  plate, violations cannot be auto-issued — they queue for review. A dedicated
  plate-detection model (YOLOv8 trained on plates, then crop-then-OCR) is the
  real fix and roughly doubles usable read rate.
- Occlusion in dense traffic breaks rider association; triple-riding recall
  drops sharply once bikes overlap.
- Night performance is untested here and will be materially worse across every
  rule.
- Single-camera only. Cross-camera re-identification uses plate text alone,
  which inherits every OCR weakness above.
- The homography assumes a flat road plane. Slopes and crests introduce error.
