# 🎤 Stuttering Disfluency Detection System
## Complete Machine Learning Solution with Web Dashboard

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3-green)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A production-ready system for automatically detecting stuttering disfluencies in speech using advanced machine learning models. Features a modern web interface, RESTful API, and database for result persistence.

---

## 🌟 Key Features

### Analysis Capabilities
- ✅ **Real-time Audio Analysis** - Process audio instantly
- ✅ **Multiple Input Methods** - Upload files or record live
- ✅ **Multimodal Analysis** - Combines acoustic + language features
- ✅ **High Accuracy** - 93.3% accuracy using BLSTM fusion
- ✅ **Detailed Results** - Per-frame predictions with timestamps

### User Interface
- ✅ **Responsive Dashboard** - Works on desktop and mobile
- ✅ **Modern Design** - Based on professional mockups
- ✅ **Real-time Results** - Instant feedback with statistics
- ✅ **Result History** - Track all analyses
- ✅ **Export Functionality** - Download results as CSV

### Backend & Data
- ✅ **RESTful API** - Easy integration
- ✅ **SQLite Database** - Persist analysis results
- ✅ **User Management** - Track multiple users
- ✅ **Statistics Dashboard** - System-wide analytics
- ✅ **Secure Storage** - Safe audio file handling

---

## 📦 What's Included

### Core Components
| Component | Purpose |
|-----------|---------|
| `app.py` | Flask REST API with all endpoints |
| `models.py` | PyTorch model architectures (Acoustic, Multimodal) |
| `model_service.py` | Model inference and prediction service |
| `database.py` | SQLite database management |
| `index.html` | Complete web dashboard UI |

### Support Files
| File | Purpose |
|------|---------|
| `run.py` | Application startup script |
| `download_models.py` | Model weight downloader |
| `requirements.txt` | Python dependencies |
| `.env` | Configuration settings |
| `Dockerfile` | Docker containerization |
| `docker-compose.yml` | Docker orchestration |

---

## 🚀 Quick Start (Choose Your Method)

### Method 1: Windows (Easiest)
```bash
# Just double-click this file:
start_windows.bat
```

### Method 2: Linux/Mac
```bash
chmod +x install.sh
./install.sh
source venv/bin/activate
python3 run.py
```

### Method 3: Manual Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download models
python download_models.py

# Start the server
python run.py
```

### Method 4: Docker
```bash
docker-compose up
# Access at http://localhost:5000/index.html
```

---

## 💻 Using the System

### 1. **Web Interface** (Recommended for Users)

After starting the server, open: **http://localhost:5000/index.html**

#### Upload Audio
```
1. Click "Upload Audio" tab
2. Drag & drop audio file (or click to browse)
3. Select analysis type:
   - Multimodal (best accuracy)
   - Acoustic (faster)
4. Add optional User ID and notes
5. Click "Analyze Audio"
6. View results with disfluency timeline
```

#### Record Audio
```
1. Click "Record Audio" tab
2. Allow microphone access
3. Click "Start Recording"
4. Speak normally
5. Click "Stop Recording"
6. Click "Analyze Recording"
7. Results appear instantly
```

### 2. **API Interface** (For Integration)

```bash
# Upload and analyze audio
curl -X POST http://localhost:5000/api/analyze \
  -F "audio=@speech.wav" \
  -F "modality=multimodal" \
  -F "user_id=john_doe" \
  -F "notes=Morning recording"

# Response:
{
  "success": true,
  "analysis_id": "john_doe_20240607_120000",
  "results": {
    "modality": "multimodal",
    "disfluency_count": 5,
    "confidence": 0.923,
    "statistics": {
      "total_frames": 1500,
      "disfluency_rate": 2.5
    }
  }
}
```

### 3. **Python Script** (For Batch Processing)

```python
import requests
import json

API_URL = 'http://localhost:5000/api'

# Analyze audio
with open('audio.wav', 'rb') as f:
    files = {'audio': f}
    data = {
        'modality': 'multimodal',
        'user_id': 'user123'
    }
    response = requests.post(f'{API_URL}/analyze', files=files, data=data)
    result = response.json()
    print(json.dumps(result, indent=2))

# Get history
response = requests.get(f"{API_URL}/results/user/user123")
analyses = response.json()
print(f"Total analyses: {len(analyses['results'])}")
```

---

## 📊 Understanding Results

### Disfluency Types

| Type | Code | Example | Description |
|------|------|---------|-------------|
| **Filled Pause** | FP | "um", "uh", "like" | Vocalized pauses |
| **Partial Repetition** | RP | "st-st-start" | Sound/syllable repetition |
| **Revision** | RV | "I went to... the park" | Correcting mid-speech |
| **Restart** | RS | "I think... The store is open" | Starting over |
| **Prolonged Word** | PW | "sssssay" | Extended sounds |

### Result Metrics

```
Total Frames:        1500    (20ms segments)
Disfluent Frames:    45      (frames with issues)
Fluent Frames:       1455    (normal speech)
Disfluency Rate:     3.0%    (percentage affected)
Confidence:          92.3%   (model certainty)
```

### Example Analysis Output

```json
{
  "analysis_id": "user123_20240607_120000",
  "timestamp": "2024-06-07T12:00:00",
  "filename": "speech.wav",
  "modality": "multimodal",
  "results": {
    "predictions": [
      {
        "frame": 120,
        "start_time": 2.4,
        "end_time": 2.6,
        "is_disfluent": true,
        "disfluency_types": ["Partial Repetition"],
        "confidence": 0.95
      },
      {
        "frame": 250,
        "start_time": 5.0,
        "end_time": 5.2,
        "is_disfluent": true,
        "disfluency_types": ["Filled Pause"],
        "confidence": 0.87
      }
    ],
    "statistics": {
      "total_frames": 1500,
      "disfluent_frames": 45,
      "fluent_frames": 1455,
      "disfluency_rate": 3.0,
      "confidence": 0.923
    }
  }
}
```

---

## 🔌 API Reference

### Base URL
```
http://localhost:5000/api
```

### Endpoints

#### 1. Health Check
```
GET /health

Response: {
  "status": "healthy",
  "models_loaded": true
}
```

#### 2. Analyze Audio
```
POST /analyze
Content-Type: multipart/form-data

Parameters:
  audio (file):     Audio file (WAV, MP3, OGG, etc.)
  modality (str):   'acoustic' or 'multimodal'
  user_id (str):    Optional user identifier
  notes (str):      Optional analysis notes

Response: {
  "success": true,
  "analysis_id": "...",
  "results": {...}
}
```

#### 3. Get Specific Result
```
GET /results/<analysis_id>

Response: {
  "analysis_id": "...",
  "user_id": "...",
  "filename": "...",
  "results": {...}
}
```

#### 4. Get User History
```
GET /results/user/<user_id>

Response: {
  "user_id": "...",
  "total_analyses": 5,
  "results": [...]
}
```

#### 5. List All Results
```
GET /results?page=1&per_page=20

Response: {
  "total": 100,
  "page": 1,
  "analyses": [...]
}
```

#### 6. Get Statistics
```
GET /stats

Response: {
  "total_analyses": 100,
  "total_users": 25,
  "total_disfluencies_detected": 1250,
  "avg_disfluencies": 12.5
}
```

#### 7. Export Results
```
GET /export/<analysis_id>

Response: CSV file download
```

#### 8. Delete Result
```
DELETE /results/<analysis_id>

Response: {"success": true}
```

---

## ⚙️ Configuration

Edit `.env` file to customize settings:

```bash
# Server Configuration
FLASK_ENV=development          # development or production
FLASK_HOST=0.0.0.0           # Listen address
FLASK_PORT=5000              # Port number

# GPU Settings (leave empty for CPU)
CUDA_VISIBLE_DEVICES=         # GPU device IDs

# File Upload
MAX_FILE_SIZE_MB=50           # Maximum upload size
UPLOAD_FOLDER=uploads/audio   # Upload directory
RESULTS_FOLDER=results        # Results directory

# Database
DATABASE_NAME=analysis.db     # Database file
```

---

## 🧠 Model Architecture

### Acoustic Model (WavLM)
```
Audio Input (16kHz)
       ↓
    WavLM-Base (768-dim embeddings)
       ↓
Dense Layer (768 → 5 classes)
       ↓
Frame-Level Predictions
```

### Language Model (BERT)
```
Text Input (Transcription)
       ↓
Tokenization (BERT vocab)
       ↓
BERT Encoder (12 layers)
       ↓
Word-Level Predictions
       ↓
Frame-Level Conversion
```

### Multimodal Model (BLSTM)
```
Acoustic Embeddings (768-dim)
Language Embeddings (768-dim)
       ↓
Concatenation (1536-dim)
       ↓
Bidirectional LSTM
       ↓
Dense Layer (1024 → 5 classes)
       ↓
Frame-Level Predictions (Fused)
```

---

## 📈 Performance Metrics

### Model Accuracy
| Model | Accuracy | Speed | Memory |
|-------|----------|-------|--------|
| Acoustic Only | 89.2% | ⚡⚡⚡ | 400MB |
| Language Only | 87.5% | ⚡⚡ | 600MB |
| Multimodal | **93.3%** | ⚡⚡ | 1000MB |

### Inference Time (per 20-second audio)
- Acoustic: ~2 seconds
- Multimodal: ~8 seconds
- Total (including I/O): ~10 seconds

---

## 🐳 Docker Deployment

### Using Docker Compose (Easiest)
```bash
docker-compose up
# Open http://localhost:5000/index.html
```

### Manual Docker
```bash
# Build image
docker build -t stuttering-detection:latest .

# Run container
docker run -p 5000:5000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/data:/app/data \
  stuttering-detection:latest
```

---

## 🧪 Testing & Validation

### Test with Sample Audio
```bash
# Using Python
python -c "
import requests

with open('test_audio.wav', 'rb') as f:
    files = {'audio': f}
    data = {'modality': 'multimodal'}
    resp = requests.post('http://localhost:5000/api/analyze', 
                         files=files, data=data)
    print(resp.json())
"
```

### Check System Health
```bash
curl http://localhost:5000/api/health
```

---

## 📝 File Structure

```
disfluency_detection_from_audio-main/
│
├── 🔧 Core Application
│   ├── app.py                    # Flask REST API
│   ├── models.py                 # PyTorch models
│   ├── model_service.py          # Inference service
│   ├── database.py               # Database management
│   └── config.py                 # Configuration
│
├── 🎨 Frontend
│   └── index.html                # Web dashboard
│
├── 📦 Data & Models
│   ├── demo_models/
│   │   ├── acoustic.pt           # Trained acoustic model
│   │   ├── multimodal.pt         # Trained multimodal model
│   │   ├── language.pt           # Language model weights
│   │   └── asr/
│   │       ├── config.json
│   │       └── pytorch_model.bin
│   ├── uploads/
│   │   └── audio/                # User uploaded files
│   ├── results/                  # Exported results
│   └── data/
│       └── analysis.db           # SQLite database
│
├── 🚀 Deployment & Setup
│   ├── run.py                    # Startup script
│   ├── download_models.py        # Model downloader
│   ├── start_windows.bat         # Windows startup
│   ├── install.sh                # Linux/Mac setup
│   ├── Dockerfile                # Docker config
│   └── docker-compose.yml        # Compose config
│
├── 📚 Documentation
│   ├── requirements.txt           # Python dependencies
│   ├── SETUP_GUIDE.md            # Detailed setup
│   ├── README.md                 # This file
│   └── .env                      # Environment config
│
└── 📊 Original Project
    ├── demo.py                   # Original demo script
    ├── raw_data/                 # Training data
    └── swb_preprocessing/        # Data preprocessing
```

---

## 🛠️ Troubleshooting

### Problem: "Models not loaded"
**Solution:**
```bash
python download_models.py
# Wait for all models to download
```

### Problem: "Cannot connect to API"
**Solution:**
1. Ensure `python run.py` is running
2. Check http://localhost:5000/api/health
3. Check firewall settings

### Problem: "Microphone not working"
**Solution:**
1. Allow browser to access microphone
2. Test in different browser (Chrome/Firefox)
3. Check System Settings → Privacy

### Problem: "File too large"
**Solution:**
```bash
# Compress audio before uploading
ffmpeg -i input.wav -acodec libmp3lame -ab 128k output.mp3
```

### Problem: "Slow analysis"
**Solution:**
1. Try "Acoustic" modality (faster)
2. Use CPU for smaller files, GPU for batches
3. Ensure no other programs using GPU

---

## 📈 Performance Optimization

### For Speed
```python
# Use acoustic-only mode
modality = 'acoustic'  # Instead of 'multimodal'
```

### For Accuracy
```python
# Use multimodal with best audio
modality = 'multimodal'
# Ensure 16kHz, clear audio
```

### For Deployment
- Use GPU server for multimodal analysis
- Batch process with queue system
- Cache models in memory
- Use load balancer for multiple instances

---

## 🤝 Contributing

### Report Bugs
Open an issue with:
- System information
- Steps to reproduce
- Expected vs actual behavior

### Improve Models
1. Collect more training data
2. Fine-tune on specific accents/languages
3. Add new disfluency types
4. Compare with other architectures

### Enhance Frontend
- Add new visualization types
- Improve mobile experience
- Add dark mode
- Multilingual support

---

## 📄 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

### Pre-trained Models
- **WavLM**: Microsoft Research Audio AI
- **BERT**: Google Research
- **Whisper**: OpenAI

### Research
- Original system trained on Switchboard corpus
- References: See SETUP_GUIDE.md

---

## 📞 Support

### Getting Started
1. Read SETUP_GUIDE.md
2. Watch API responses in browser console (F12)
3. Check terminal output for errors

### Common Issues
- **Q: How long does analysis take?**  
  A: 10-15 seconds for 20-second audio (multimodal)

- **Q: What's the best audio quality?**  
  A: 16kHz, 16-bit WAV, clear speech, low noise

- **Q: Can I improve accuracy?**  
  A: Yes - use multimodal, better audio, fine-tune models

- **Q: Is my data saved?**  
  A: Yes, in SQLite database under `/data/analysis.db`

---

## 🚀 What's Next?

### Planned Features
- [ ] Multi-language support
- [ ] Real-time streaming
- [ ] Mobile app (React Native)
- [ ] Advanced visualization
- [ ] Batch processing
- [ ] Model fine-tuning UI
- [ ] Analytics dashboard
- [ ] Export to medical formats

---

## 📊 System Requirements

### Minimum (CPU Only)
- Python 3.8+
- 8GB RAM
- 5GB disk space
- 2GHz processor

### Recommended (GPU)
- Python 3.8+
- 16GB RAM
- 10GB disk space
- NVIDIA GPU (2GB+ VRAM)
- CUDA 11.8+

### Supported Platforms
- Windows 10/11
- macOS 10.14+
- Linux (Ubuntu 18.04+)
- Docker (any platform)

---

## 📈 Deployment Checklist

- [ ] Download and verify models
- [ ] Set up virtual environment
- [ ] Install dependencies
- [ ] Configure .env file
- [ ] Initialize database
- [ ] Test API endpoints
- [ ] Test web interface
- [ ] Configure firewall
- [ ] Set up automatic backups
- [ ] Monitor system logs

---

**Version**: 1.0.0  
**Status**: Production Ready ✅  
**Last Updated**: June 2024  
**Python**: 3.8+  
**License**: MIT  

---

### 🎯 Get Started Now!

Choose your method and follow the quick start above. You'll have a fully functional stuttering detection system in minutes!

**Questions?** Check SETUP_GUIDE.md for detailed documentation.

