#!/bin/bash
# Stuttering Disfluency Detection System - Linux/Mac Installation Script

set -e  # Exit on error

echo ""
echo "============================================================"
echo "Stuttering Disfluency Detection System - Setup"
echo "============================================================"
echo ""

# Check Python version
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Python $PYTHON_VERSION found"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Install requirements
echo "Installing Python dependencies..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create directories
echo "Creating required directories..."
mkdir -p uploads/audio results data demo_models/asr
echo "✓ Directories created"
echo ""

# Download models
if [ ! -f "demo_models/acoustic.pt" ]; then
    echo "Downloading pre-trained models..."
    echo "(This may take 5-10 minutes depending on connection)"
    echo ""
    python3 download_models.py
    echo ""
else
    echo "✓ Models already present"
fi

echo ""
echo "============================================================"
echo "✓ Setup Complete!"
echo "============================================================"
echo ""
echo "To start the application, run:"
echo "  source venv/bin/activate"
echo "  python3 run.py"
echo ""
echo "Then open in your browser:"
echo "  http://localhost:5000/index.html"
echo ""
