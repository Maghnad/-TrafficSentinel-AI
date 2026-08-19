@echo off
REM One-time setup. Creates an isolated venv and installs dependencies.
REM Usage:  setup.bat      then   run.bat
setlocal
cd /d "%~dp0"

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found on PATH. Install Python 3.9-3.12 from python.org
    echo and tick "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

if not exist ".venv" (
    echo [setup] creating .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: venv creation failed.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

python -m pip install --upgrade pip wheel

REM CPU-only torch. The default index pulls the ~2.5GB CUDA build, which is
REM wasted download if you have no NVIDIA GPU. Comment out if you do.
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [setup] no NVIDIA GPU detected - installing CPU-only torch
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: dependency install failed. See messages above.
    pause
    exit /b 1
)

echo [setup] installing OpenVINO (optional, 2-3x CPU speedup) ...
pip install openvino

echo.
echo [setup] done.
echo   activate :  .venv\Scripts\activate
echo   calibrate:  python calibrate.py --source traffic.mp4 --out camera.json
echo   benchmark:  python run_headless.py --source traffic.mp4 --benchmark 300
echo   dashboard:  run.bat
pause
