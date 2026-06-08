"""
Configuration file for Stuttering Disfluency Detection System
"""
import os
from pathlib import Path

class Config:
    """Base configuration"""
    
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_ENV', 'production') == 'development'
    JSON_SORT_KEYS = False
    
    # Project root
    PROJECT_ROOT = Path(__file__).parent.absolute()
    
    # File upload settings
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max file size
    UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'uploads', 'audio')
    RESULTS_FOLDER = os.path.join(PROJECT_ROOT, 'results')
    MODELS_FOLDER = os.path.join(PROJECT_ROOT, 'demo_models')
    
    # Database settings
    DATABASE_FOLDER = os.path.join(PROJECT_ROOT, 'data')
    DATABASE_PATH = os.path.join(DATABASE_FOLDER, 'analysis.db')
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'flac'}
    
    # Model settings
    DEVICE = 'cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu'
    
    # API settings
    API_TITLE = 'Stuttering Disfluency Detection API'
    API_VERSION = '1.0.0'
    
    # CORS settings
    CORS_HEADERS = 'Content-Type'
    
    # Audio processing settings
    SAMPLE_RATE = 16000
    FRAME_SIZE = 0.02  # 20ms frames
    
    # Analysis settings
    CONFIDENCE_THRESHOLD = 0.5
    SUPPORTED_MODALITIES = ['acoustic', 'language', 'multimodal']
    
    # Disfluency labels
    DISFLUENCY_LABELS = {
        'FP': 'Filled Pause',
        'RP': 'Partial Repetition',
        'RV': 'Revision',
        'RS': 'Restart',
        'PW': 'Prolonged Word'
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_PATH = ':memory:'

# Select config based on environment
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
