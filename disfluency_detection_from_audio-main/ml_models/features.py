import numpy as np
import librosa

def extract_acoustic_features(wav_path=None, y=None, sr=None, n_mfcc=13, hop_length=160, n_fft=400):
    """Return MFCCs (mean+std), pitch (median), and energy (mean+std) as a 1D feature vector.
    Provide either wav_path or (y,sr)."""
    if wav_path is not None:
        y, sr = librosa.load(wav_path, sr=None)
    if y is None or sr is None:
        raise ValueError('Provide wav_path or (y,sr)')

    # Pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    # Pitch (using librosa.pyin if available)
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr, hop_length=hop_length)
        pitch_median = np.nanmedian(f0)
        pitch_mean = np.nanmean(f0)
    except Exception:
        # fallback to zero if pyin not available
        pitch_median = 0.0
        pitch_mean = 0.0

    # Energy (RMS)
    rms = librosa.feature.rms(y=y, hop_length=hop_length, frame_length=n_fft)
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    features = np.concatenate([mfcc_mean, mfcc_std, [pitch_mean, pitch_median, rms_mean, rms_std]])
    return features

def extract_acoustic_features_from_array(y, sr):
    return extract_acoustic_features(y=y, sr=sr)
