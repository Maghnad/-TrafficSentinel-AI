@echo off
echo 🚦 Starting TrafficSentinel AI Setup...

:: Check if Python is installed
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ❌ Python could not be found. Please install Python 3.9+ to continue.
    pause
    exit /b
)

:: Create a virtual environment if it doesn't exist
IF NOT EXIST "venv\Scripts\activate.bat" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
echo 🔄 Activating virtual environment...
call venv\Scripts\activate.bat

:: Install requirements
echo 📥 Installing required packages...
pip install -r requirements.txt

:: Start the application
echo 🚀 Launching TrafficSentinel AI Dashboard...
streamlit run app.py

pause
