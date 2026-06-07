"""Feature extraction: MFCC, pitch, energy

Usage examples:
    from features.extract_features import extract_mfcc, extract_pitch, extract_energy
    mfcc = extract_mfcc("dataset/audio.wav")
"""
import numpy as np
import librosa
from pathlib import Path


def extract_mfcc(path, sr=16000, n_mfcc=13, hop_length=160, n_fft=512):
    y, _ = librosa.load(path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    return mfcc.T


def extract_pitch(path, sr=16000, fmin=50, fmax=500, hop_length=160):
    y, _ = librosa.load(path, sr=sr, mono=True)
    # use YIN algorithm for fundamental frequency
    try:
        pitches = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    except Exception:
        # fallback to librosa.yin availability issues
        pitches = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length)
    # replace unvoiced frames (NaNs) with 0
    pitches = np.nan_to_num(pitches)
    return pitches


def extract_energy(path, sr=16000, hop_length=160, frame_length=512):
    y, _ = librosa.load(path, sr=sr, mono=True)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    return rms


def extract_all(path, sr=16000):
    mfcc = extract_mfcc(path, sr=sr)
    pitch = extract_pitch(path, sr=sr)
    energy = extract_energy(path, sr=sr)
    return {
        "mfcc": mfcc,
        "pitch": pitch,
        "energy": energy,
    }


def save_features_npz(path, out_path):
    feats = extract_all(path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, mfcc=feats["mfcc"], pitch=feats["pitch"], energy=feats["energy"])
    return out_path


def batch_extract(manifest_csv, out_dir):
    import pandas as pd
    df = pd.read_csv(manifest_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in df.iterrows():
        src = row["path"]
        basename = Path(src).stem
        out_path = out_dir / (basename + ".npz")
        save_features_npz(src, out_path)
    print("Features saved to:", out_dir)
