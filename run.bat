@echo off
REM Launch the dashboard. Assumes setup.bat has been run once.
setlocal
cd /d "%~dp0"
if not exist ".venv" (
    echo No .venv found. Run setup.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
streamlit run app.py
pause
