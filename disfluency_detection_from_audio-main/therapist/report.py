"""Therapist helper: convert analysis results into a concise summary report.

Simple utility used by the web UI or offline scripts to create human-readable summaries.
"""
import json
from pathlib import Path


def generate_report(analysis_result: dict) -> dict:
    # analysis_result expected keys: total_frames, disfluent_frames, per_label_counts, avg_confidences, modality
    total = analysis_result.get("total_frames", 0)
    disfluent = analysis_result.get("disfluent_frames", 0)
    rate = disfluent / total if total else 0.0
    per_label = analysis_result.get("per_label_counts", {})
    avg_conf = analysis_result.get("avg_confidences", {})

    summary = {
        "total_frames": total,
        "disfluent_frames": disfluent,
        "disfluency_rate": rate,
        "per_label_counts": per_label,
        "avg_confidences": avg_conf,
        "notes": [],
    }

    # quick heuristics for therapist guidance
    if rate > 0.15:
        summary["notes"].append("High disfluency rate — consider targeted practice sessions.")
    elif rate > 0.05:
        summary["notes"].append("Moderate disfluency rate — monitor progress and provide exercises.")
    else:
        summary["notes"].append("Low disfluency rate — continue current therapy.")

    # highlight most frequent labels
    if per_label:
        top = sorted(per_label.items(), key=lambda kv: kv[1], reverse=True)[:3]
        summary["notes"].append(f"Most common disfluency types: {', '.join(k for k, _ in top)}")

    return summary


def save_report(summary: dict, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return out_path
