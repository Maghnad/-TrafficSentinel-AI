"""Fine-tune a helmet classifier. Run this on Colab's free GPU (~15 min).

This is the one place custom training genuinely earns its cost. The HSV
skin-tone heuristic it replaces fails in two ways that matter for Indian
footage: fixed skin-tone hue ranges are unreliable across darker skin tones,
and a black helmet against black hair is nearly identical in both skin ratio
and saturation variance.

Expected dataset layout (any helmet/no-helmet set works - several are on
Kaggle and Roboflow Universe; ~1000 images per class is enough):

    helmet_ds/
      train/helmet/*.jpg      train/no_helmet/*.jpg
      val/helmet/*.jpg        val/no_helmet/*.jpg

Crops should be HEADS, not whole riders - match what HelmetClassifier.head_crop
produces (top ~30% of the person box, widened 12%). prepare_crops() below can
generate these from a rider-level dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def prepare_crops(src_root: str, dst_root: str, ratio: float = 0.30) -> None:
    """Convert full-rider images into head crops matching inference-time
    geometry. Train/test skew here is the usual reason a classifier scores 97%
    offline and 60% on real footage."""
    import cv2
    src, dst = Path(src_root), Path(dst_root)
    for split in ("train", "val"):
        for cls in ("helmet", "no_helmet"):
            out = dst / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate((src / split / cls).glob("*.*")):
                img = cv2.imread(str(p))
                if img is None:
                    continue
                h, w = img.shape[:2]
                crop = img[0:max(24, int(h * ratio)), :]
                cv2.imwrite(str(out / f"{i:05d}.jpg"), crop)
    print(f"head crops -> {dst}")


def train(data: str, epochs: int, out: str) -> None:
    from ultralytics import YOLO
    model = YOLO("yolov8n-cls.pt")
    model.train(data=data, epochs=epochs, imgsz=96, batch=64,
                pretrained=True, patience=8,
                # Heavy augmentation: real footage is motion-blurred, small,
                # and shot at every hour of the day.
                hsv_h=0.02, hsv_s=0.6, hsv_v=0.5, degrees=12,
                translate=0.12, scale=0.4, fliplr=0.5, erasing=0.3)
    metrics = model.val()
    print(metrics)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.export(format="openvino", imgsz=96)
    print(f"\nCopy best.pt to {out} and set helmet.model_path in camera.json")
    print("Sanity check before trusting it: top-1 on a held-out set of YOUR "
          "camera's footage, not the public val split. If it is below ~0.90, "
          "leave it out and let the review queue handle helmets.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="helmet_ds")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="models/helmet_cls.pt")
    ap.add_argument("--prepare-from", default=None)
    args = ap.parse_args()

    if args.prepare_from:
        prepare_crops(args.prepare_from, args.data)
    train(args.data, args.epochs, args.out)
