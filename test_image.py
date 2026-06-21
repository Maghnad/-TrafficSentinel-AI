import cv2
from core.pipeline import TrafficPipeline
from core.scene_graph import SceneGraphBuilder, ViolationEngine
import sys

pipeline = TrafficPipeline()
image = cv2.imread(r"C:\Users\debna\.gemini\antigravity-ide\brain\9cfce2f4-29c1-466b-865f-46edb9c5ce00\media__1781919357700.jpg")

if image is None:
    print("Could not load image.")
    sys.exit()

results = pipeline.run(image, skip_ocr=True) # Let's run OCR manually next
print(f"Detected {len(results['all_detections'])} objects")

# Let's test the specific OCR pipeline crop logic on the motorcycle crop
import cv2
import numpy as np
import easyocr
import re

image = cv2.imread(r"C:\Users\debna\.gemini\antigravity-ide\brain\9cfce2f4-29c1-466b-865f-46edb9c5ce00\media__1781919357700.jpg")
reader = easyocr.Reader(['en'], gpu=False)

# Let's hardcode the approximate crop of the central motorcycle based on the image
# Actually, let's just crop the bottom 45% of the center of the image
h, w = image.shape[:2]
crop = image[int(h*0.5):h, int(w*0.3):int(w*0.7)]
cv2.imwrite("test_crop.jpg", crop)

# 1. Resize 2x
resized = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(gray, 11, 17, 17)

print("--- With Bilateral Filter ---")
res1 = reader.readtext(blur, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -")
for bbox, text, conf in res1:
    print(f"[{conf:.2f}] {text}")

print("--- Grayscale Only ---")
res2 = reader.readtext(gray, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -")
for bbox, text, conf in res2:
    print(f"[{conf:.2f}] {text}")

print("--- Resize 3x Grayscale ---")
resized3 = cv2.resize(crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
gray3 = cv2.cvtColor(resized3, cv2.COLOR_BGR2GRAY)
res3 = reader.readtext(gray3, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -")
for bbox, text, conf in res3:
    print(f"[{conf:.2f}] {text}")
