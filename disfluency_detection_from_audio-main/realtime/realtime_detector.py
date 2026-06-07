"""Realtime demo: record short frames from microphone and run a saved model (joblib SVM/RF) or a torch model.

Usage:
    python realtime/realtime_detector.py --model models/svm_baseline.joblib

This lightweight demo uses `sounddevice` to capture audio and `librosa` to extract features.
"""
import argparse
import queue
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import joblib
import tempfile
import soundfile as sf
from features.extract_features import save_features_npz, extract_all


def record_block(duration=1.0, sr=16000):
    print(f"Recording {duration}s...")
    data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return data.flatten()


def make_feature_vector_from_array(y, sr=16000):
    # save to temp file and reuse extractor
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(tmp.name, y, sr)
    fv = save_features_npz(tmp.name, tmp.name + '.npz')
    arr = np.load(tmp.name + '.npz')
    mfcc = arr['mfcc']
    pitch = arr['pitch']
    energy = arr['energy']
    vec = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0), [pitch.mean()], [pitch.std()], [energy.mean()], [energy.std()]])
    return vec


def run_loop(model_path, sr=16000, duration=1.0):
    model_path = Path(model_path)
    model = joblib.load(model_path)
    print("Loaded model:", model_path)
    try:
        while True:
            y = record_block(duration=duration, sr=sr)
            vec = make_feature_vector_from_array(y, sr=sr)
            pred = model.predict([vec])[0]
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba([vec]).max()
            else:
                prob = None
            print(f"Prediction: {pred} (conf={prob})")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping realtime demo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args()
    run_loop(args.model, sr=args.sr, duration=args.duration)


if __name__ == "__main__":
    main()
