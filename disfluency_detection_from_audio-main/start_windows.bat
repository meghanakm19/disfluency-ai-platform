@echo off
REM Stuttering Disfluency Detection System - Windows Startup Script

echo.
echo ============================================================
echo Stuttering Disfluency Detection System - Windows Setup
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo ✓ Python found
echo.

REM Create virtual environment
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo ✓ Virtual environment created
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo Error: pip is not available
    pause
    exit /b 1
)

echo ✓ Virtual environment activated
echo.

REM Install requirements
echo Installing dependencies...
pip install -q -r requirements.txt
if errorlevel 1 (
    echo Error installing dependencies
    pause
    exit /b 1
)
echo ✓ Dependencies installed
echo.

REM Download models
echo Checking for models...
if not exist "demo_models\acoustic.pt" (
    echo.
    echo Models not found. Downloading...
    python download_models.py
    if errorlevel 1 (
        echo Warning: Some models failed to download
        echo You can try again later with: python download_models.py
    )
) else (
    echo ✓ Models already present
)

echo.
echo ============================================================
echo Ready to start!
echo ============================================================
echo.
echo Starting Stuttering Disfluency Detection API...
echo.
echo Access the web interface at:
echo   👉 http://localhost:5000/index.html
echo.
echo Press Ctrl+C to stop the server
echo.
python run.py

pause
