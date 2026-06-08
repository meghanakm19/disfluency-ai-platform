from huggingface_hub import hf_hub_download
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def download_models():
    model_dir = PROJECT_ROOT / "demo_models"
    model_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Downloading Models From Hugging Face")
    print("=" * 60)

    repo_id = "aryankarna79/disfluency-ai-models"

    files = [
        "acoustic.pt",
        "language.pt",
        "multimodal.pt"
    ]

    for file in files:
        try:
            print(f"Downloading {file}...")

            hf_hub_download(
                repo_id=repo_id,
                filename=file,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False
            )

            print(f"{file} downloaded successfully")

        except Exception as e:
            print(f"Failed downloading {file}: {e}")
            return False

    print("=" * 60)
    print("All models downloaded successfully!")
    print("=" * 60)

    return True

if __name__ == "__main__":
    download_models()