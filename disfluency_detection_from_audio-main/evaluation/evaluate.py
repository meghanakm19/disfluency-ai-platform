"""Evaluation utilities: compute metrics and save report

Usage:
    python evaluation/evaluate.py --preds preds.csv --out report.json

preds CSV expected columns: true_label,pred_label,prob
"""
import argparse
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd


def evaluate_frame_level(df):
    y_true = df["true_label"].values
    y_pred = df["pred_label"].values
    acc = float(accuracy_score(y_true, y_pred))
    prec, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.preds)
    report = evaluate_frame_level(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("Saved report to", args.out)


if __name__ == "__main__":
    main()
