# 🚦 TrafficSentinel AI

**Automated Photo Identification and Classification for Traffic Violations Using Computer Vision**

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## How to Use

1. Open the app in your browser
2. Go to **🔍 Analyze Image**
3. Upload a traffic camera image (JPG/PNG)
4. Click **Analyze for Violations**
5. View results: annotated image, scene graph, violations, plate readings
6. Check **📊 Analytics Dashboard** for cumulative stats

## Violations Detected

| Violation | How it works |
|---|---|
| 🪖 **Helmet non-compliance** | Detects person on motorcycle → checks head region for helmet |
| 👨‍👩‍👧 **Triple riding** | Counts persons associated with a single motorcycle (≥3 = violation) |
| 🚦 **Red-light violation** | Detects red traffic signal + vehicle in intersection |
| 🔢 **No visible plate** | Flags vehicles with no readable license plate |

## Project Structure

```
Traffic/
├── app.py                 # Streamlit UI (main entry point)
├── core/
│   ├── __init__.py        # Package exports
│   ├── pipeline.py        # YOLOv8 detection + EasyOCR + preprocessing
│   ├── scene_graph.py     # Scene graph builder + violation engine
│   └── annotator.py       # Image annotation with bounding boxes
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Technology Stack

- **YOLOv8** (Ultralytics) — Pretrained object detection (COCO)
- **EasyOCR** — License plate text extraction
- **Scene Graph Engine** — Custom spatial reasoning for violations
- **Streamlit** — Interactive web dashboard
- **Plotly** — Analytics charts
- **OpenCV** — Image processing (CLAHE, annotation)

## Sample Images for Testing

Download traffic images from:
- Google Images: search "Indian traffic camera" or "traffic violation India"
- Kaggle datasets: search "traffic violation detection"
- YouTube: screenshot from traffic camera footage

## Note

- First run downloads YOLOv8 model (~87MB) and EasyOCR models (~150MB)
- GPU is optional — works on CPU (slower but functional)
- Best results with clear, daytime traffic images at 720p or higher
