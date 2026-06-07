"""Audio preprocessing helpers

Functions: load, trim silence, normalize RMS, resample and save.
"""
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf


def load_audio(path, sr=16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr


def trim_silence(y, top_db=20):
    yt, idx = librosa.effects.trim(y, top_db=top_db)
    return yt


def normalize_rms(y, target_rms=0.1):
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        y = y * (target_rms / rms)
    return y


def preprocess_file(src, dst, sr=16000, trim_db=20, target_rms=0.1):
    y, _ = load_audio(src, sr=sr)
    y = trim_silence(y, top_db=trim_db)
    y = normalize_rms(y, target_rms=target_rms)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, y, sr)
    return dst


def batch_preprocess(manifest_csv, out_dir, sr=16000, trim_db=20, target_rms=0.1):
    import pandas as pd
    from pathlib import Path

    df = pd.read_csv(manifest_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in df.iterrows():
        src = row["path"]
        dst = out_dir / Path(src).name
        preprocess_file(src, str(dst), sr=sr, trim_db=trim_db, target_rms=target_rms)

    print("Preprocessing done to:", out_dir)
