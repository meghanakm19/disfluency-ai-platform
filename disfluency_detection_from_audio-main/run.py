#!/usr/bin/env python
"""
Startup Script for Stuttering Disfluency Detection System
Initializes the backend API server
"""
import os
import sys
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def check_requirements():
    """Check if all required packages are installed"""
    print("Checking dependencies...")
    try:
        import torch
        import flask
        import pandas
        print("All dependencies found")
        return True
    except ImportError as e:
        print(f"Missing dependency: {str(e)}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt")
        return False

def check_models():
    """Check if models are downloaded"""
    model_dir = PROJECT_ROOT / 'demo_models'
    required_models = [
        'acoustic.pt',
        'multimodal.pt',
        'language.pt'
    ]
    
    print("\nChecking models...")
    missing_models = []
    
    for model in required_models:
        if (model_dir / model).exists():
            print(f"{model} found")
        else:
            missing_models.append(model)
            print(f"{model} missing")
    
    if missing_models:
        print("\nDownloading missing models...")
        try:
            subprocess.run([sys.executable, str(PROJECT_ROOT / 'download_models.py')], check=True)
            return True
        except subprocess.CalledProcessError:
            print("Error downloading models. Please run: python download_models.py")
            return False
    
    return True

def create_directories():
    """Create required directories"""
    directories = [
        PROJECT_ROOT / 'uploads' / 'audio',
        PROJECT_ROOT / 'results',
        PROJECT_ROOT / 'data',
        PROJECT_ROOT / 'demo_models' / 'asr'
    ]
    
    print("\nCreating directories...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"{directory}")

def start_backend():
    """Start the Flask backend server"""
    print("\n" + "=" * 60)
    print("Starting Stuttering Disfluency Detection API")
    print("=" * 60)
    
    try:
        from app import app, db
        
        # Initialize database
        db.init_db()
        print("Database initialized")
        
        # Start Flask server
        print("\nBackend server starting...")
        print("API available at http://localhost:5000")
        print("Frontend available at http://localhost:5000/index.html")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 60 + "\n")
        
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        return False
    
    return True

def main():
    """Main startup sequence"""
    print("\n" + "=" * 60)
    print("Stuttering Disfluency Detection System - Setup")
    print("=" * 60 + "\n")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check and download models
    if not check_models():
        print("\nNote: You can start the server without models,")
        print("but analysis will fail until models are downloaded.")
    
    # Start backend
    if not start_backend():
        sys.exit(1)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nServer stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)
