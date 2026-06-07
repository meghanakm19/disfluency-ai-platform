Research & Implementation Plan — Stuttering Disfluency Detection

This document captures the research objectives and the accompanying scripts added to this repo to support dataset collection, preprocessing, feature extraction, classical baselines, evaluation, realtime demo, and therapist-assist reporting.

Objectives:

- Study and understand disfluency forms: repetitions, prolongations, pauses.
- Collect and prepare speech datasets (manifest generation, train/val/test splits).
- Preprocess audio: resample to 16 kHz, trim silence, normalize amplitude.
- Extract acoustic features: MFCC, pitch (fundamental frequency), and energy (RMS).
- Implement ML models for disfluency detection: SVM, Random Forest, and deep learning hooks.
- Compare algorithms using accuracy/precision/recall/F1 and confusion matrices.
- Provide a realtime detection demo for recorded or live microphone input.
- Provide helper utilities to generate therapist-facing summary reports.

Files added (overview):

- `pipeline/prepare_dataset.py` — manifest creation and dataset split helper.
- `pipeline/preprocess.py` — audio preprocessing utilities (resample, trim, normalize).
- `features/extract_features.py` — feature extraction (MFCC, pitch, energy) and batch processing.
- `ml_models/classical_baselines.py` — train and save SVM and RandomForest baselines using scikit-learn.
- `evaluation/evaluate.py` — evaluation metrics and report generation.
- `realtime/realtime_detector.py` — small demo for streaming audio from a microphone and running a saved model.
- `therapist/report.py` — helper to convert analysis results into a concise report for therapists.
- `requirements_extra.txt` — Python packages required for these scripts.

Usage notes and examples are included at the top of each script.
