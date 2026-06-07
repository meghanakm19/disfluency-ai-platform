"""Dataset preparation helpers

Creates a CSV manifest for files under a `raw_data` folder, copies/resamples files
into a `dataset/` folder and produces train/val/test splits.

Usage:
    python pipeline/prepare_dataset.py --raw raw_data --out dataset --sr 16000 --test-size 0.1 --val-size 0.1
"""
import argparse
import csv
import os
import random
import shutil
from pathlib import Path

import soundfile as sf
import librosa


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def resample_copy(src_path, dst_path, sr=16000):
    y, _ = librosa.load(src_path, sr=sr, mono=True)
    sf.write(dst_path, y, sr)


def build_manifest(raw_dir, out_dir, sr=16000):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    manifest_path = out_dir / "manifest.csv"
    rows = []
    for root, _dirs, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith((".wav", ".flac", ".mp3", ".m4a", ".ogg")):
                src = Path(root) / f
                rel = src.relative_to(raw_dir)
                dst = out_dir / rel
                ensure_dir(dst.parent)
                resample_copy(str(src), str(dst), sr=sr)
                rows.append({"path": str(dst), "label": "unknown"})

    with open(manifest_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["path", "label"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return manifest_path


def split_manifest(manifest_csv, out_dir, test_size=0.1, val_size=0.1, seed=42):
    import pandas as pd

    df = pd.read_csv(manifest_csv)
    idx = list(df.index)
    random.Random(seed).shuffle(idx)
    n = len(idx)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    df.iloc[train_idx].to_csv(out_dir / "train_manifest.csv", index=False)
    df.iloc[val_idx].to_csv(out_dir / "val_manifest.csv", index=False)
    df.iloc[test_idx].to_csv(out_dir / "test_manifest.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    args = parser.parse_args()

    manifest = build_manifest(args.raw, args.out, sr=args.sr)
    split_manifest(manifest, args.out, test_size=args.test_size, val_size=args.val_size)
    print("Manifest and splits written to:", args.out)


if __name__ == "__main__":
    main()
