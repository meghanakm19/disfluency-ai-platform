"""
Model Download Script
Downloads pre-trained model weights from Google Drive
"""
import os
import sys
import gdown
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def download_models():
    """Download pre-trained model weights"""
    
    # Model directory
    model_dir = PROJECT_ROOT / 'demo_models'
    model_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Downloading Stuttering Disfluency Detection Models")
    print("=" * 60)
    
    models = {
        'Whisper ASR (config)': {
            'id': '1BeT7m_5qv19Sb5yrZ2zhKu6fEprUoB9N',
            'path': model_dir / 'asr' / 'config.json'
        },
        'Whisper ASR (weights)': {
            'id': '13n8VrTFVq4jGouCDamkReHlHm_1yz20U',
            'path': model_dir / 'asr' / 'pytorch_model.bin'
        },
        'Language Model': {
            'id': '1GQIXgCSF3Usiuy5hkxgOl483RPX3f_SX',
            'path': model_dir / 'language.pt'
        },
        'Acoustic Model': {
            'id': '1wWrmopvvdhlBw-cL7EDyih9zn_IJu5Wr',
            'path': model_dir / 'acoustic.pt'
        },
        'Multimodal Model': {
            'id': '1LPchbScA_cuFx1XoNxpFCYZfGoJCfWao',
            'path': model_dir / 'multimodal.pt'
        }
    }
    
    # Create asr directory
    (model_dir / 'asr').mkdir(exist_ok=True)
    
    for model_name, model_info in models.items():
        filepath = model_info['path']
        
        if filepath.exists():
            print(f"{model_name} already downloaded")
            continue
        
        try:
            print(f"\nDownloading {model_name}...")
            gdown.download(
                f"https://drive.google.com/uc?id={model_info['id']}",
                str(filepath),
                quiet=False
            )
            print(f"{model_name} downloaded successfully")
        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")
            return False
    
    print("\n" + "=" * 60)
    print("All models downloaded successfully!")
    print("=" * 60)
    return True

if __name__ == '__main__':
    success = download_models()
    sys.exit(0 if success else 1)
