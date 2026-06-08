"""Train classical ML baselines (SVM, RandomForest) on extracted features.

Expect input CSV with columns: feature_path,label
Feature files are `.npz` created by `features.extract_features.save_features_npz`.

Usage:
    python models/classical_baselines.py --manifest features_manifest.csv --out models/
"""
import argparse
from pathlib import Path
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_feature_vector(npz_path):
    data = np.load(npz_path)
    mfcc = data["mfcc"]
    pitch = data["pitch"]
    energy = data["energy"]
    # simple pooling: mean and std
    feats = np.concatenate([mfcc.mean(axis=0), mfcc.std(axis=0), [pitch.mean()], [pitch.std()], [energy.mean()], [energy.std()]])
    return feats


def build_dataset(manifest_csv):
    import pandas as pd
    manifest_path = Path(manifest_csv).resolve()
    df = pd.read_csv(manifest_path)
    X = []
    y = []
    for i, row in df.iterrows():
        path = row["feature_path"] if "feature_path" in row else row["path"]
        feature_path = Path(path)
        if not feature_path.is_absolute():
            feature_path = manifest_path.parent / feature_path
        label = row.get("label", "unknown")
        vec = load_feature_vector(str(feature_path))
        X.append(vec)
        y.append(label)
    X = np.vstack(X)
    return X, y


def train_and_save(X, y, out_dir):
    n_samples = len(y)
    n_test = max(2, int(round(n_samples * 0.2)))
    if n_test >= n_samples:
        n_test = max(2, n_samples // 2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=42, stratify=y
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    svm = SVC(kernel='rbf', probability=True)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    print("SVM results:\n", classification_report(y_test, y_pred))
    joblib.dump(svm, out_dir / "svm_baseline.joblib")

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    print("RandomForest results:\n", classification_report(y_test, y_pred))
    joblib.dump(rf, out_dir / "rf_baseline.joblib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    X, y = build_dataset(args.manifest)
    train_and_save(X, y, args.out)


if __name__ == "__main__":
    main()
