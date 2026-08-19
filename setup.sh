#!/usr/bin/env bash
# One-time setup. Creates an isolated venv and installs dependencies.
# Usage:  ./setup.sh          then   ./run.sh
set -e

cd "$(dirname "$0")"

PY=${PYTHON:-python3}
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: $PY not found. Install Python 3.9-3.12."
    exit 1
fi

VER=$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
echo "[setup] using Python $VER"
case "$VER" in
    3.9|3.10|3.11|3.12) ;;
    *) echo "[setup] WARNING: torch/ultralytics wheels may not exist for $VER." ;;
esac

if [ ! -d ".venv" ]; then
    echo "[setup] creating .venv ..."
    "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip wheel

# CPU-only torch first: the default index pulls the ~2.5GB CUDA build, which
# is useless if you have no NVIDIA GPU. Skip this block if you do.
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[setup] no NVIDIA GPU detected - installing CPU-only torch (much smaller)"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

pip install -r requirements.txt

# OpenVINO gives a 2-3x CPU speedup and is the difference between ~12 FPS and
# ~25 FPS on a laptop. Not fatal if it fails (e.g. on ARM).
echo "[setup] installing OpenVINO (optional accelerator) ..."
pip install openvino || echo "[setup] OpenVINO unavailable - falling back to PyTorch/ONNX"

python - <<'EOF'
import importlib
missing = [m for m in ("cv2", "numpy", "ultralytics", "streamlit")
           if importlib.util.find_spec(m) is None]
print("[setup] MISSING:", missing) if missing else print("[setup] core imports OK")
EOF

echo
echo "[setup] done."
echo "  activate :  source .venv/bin/activate"
echo "  calibrate:  python calibrate.py --source traffic.mp4 --out camera.json"
echo "  benchmark:  python run_headless.py --source traffic.mp4 --benchmark 300"
echo "  dashboard:  streamlit run app.py     (or ./run.sh)"
