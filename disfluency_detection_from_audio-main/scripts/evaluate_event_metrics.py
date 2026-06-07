import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support

# Simple event-level conversion: collapse contiguous frame-level positives into events
def frames_to_events(frame_activity):
    events = []
    in_event = False
    start = None
    for i, v in enumerate(frame_activity):
        if v and not in_event:
            in_event = True
            start = i
        elif not v and in_event:
            in_event = False
            events.append((start, i-1))
    if in_event:
        events.append((start, len(frame_activity)-1))
    return events


def event_level_scores(y_true_frames, y_pred_frames):
    # y_true_frames and y_pred_frames are arrays of shape (n_samples, n_frames)
    precisions = []
    recalls = []
    fs = []
    for i in range(len(y_true_frames)):
        true_events = frames_to_events(y_true_frames[i])
        pred_events = frames_to_events(y_pred_frames[i])
        # naive matching: count predicted events that overlap any true event
        tp = 0
        for pe in pred_events:
            for te in true_events:
                if not (pe[1] < te[0] or pe[0] > te[1]):
                    tp += 1
                    break
        fp = max(0, len(pred_events) - tp)
        fn = max(0, len(true_events) - tp)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        fs.append(f)
    return np.mean(precisions), np.mean(recalls), np.mean(fs)


def load_manifest(manifest_csv):
    df = pd.read_csv(manifest_csv)
    X = []
    y = []
    for _, row in df.iterrows():
        path = row['path'] if 'path' in row else row['feature_path']
        X.append(path)
        y.append(int(row.get('label', 0)))
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--predictions', required=True, help='CSV with per-frame predictions (one row per file, comma-separated 0/1)')
    args = parser.parse_args()

    # Load ground truth frames
    gt = pd.read_csv(args.manifest)
    preds = pd.read_csv(args.predictions)

    y_true = []
    y_pred = []
    for _, row in gt.iterrows():
        key = row['path']
        gt_frames = np.fromstring(row['frames'], sep=',', dtype=int) if 'frames' in row else np.array([])
        pred_row = preds[preds['path'] == key]
        if pred_row.empty:
            continue
        pred_frames = np.fromstring(pred_row.iloc[0]['frames'], sep=',', dtype=int)
        y_true.append(gt_frames)
        y_pred.append(pred_frames)

    # Pad sequences to same length
    max_len = max([len(a) for a in y_true])
    y_true_padded = np.array([np.pad(a, (0, max_len - len(a))) for a in y_true])
    y_pred_padded = np.array([np.pad(a, (0, max_len - len(a))) for a in y_pred])

    prec, rec, f = event_level_scores(y_true_padded, y_pred_padded)
    print(f'Event-level scores: Precision={prec:.3f}, Recall={rec:.3f}, F1={f:.3f}')

if __name__ == '__main__':
    main()
