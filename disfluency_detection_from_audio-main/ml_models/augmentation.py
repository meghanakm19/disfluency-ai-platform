import numpy as np
import librosa
import random

def add_noise(y, noise_factor=0.005):
    noise = np.random.randn(len(y))
    augmented = y + noise_factor * noise
    return augmented.astype(y.dtype)

def time_shift(y, shift_max=0.2, sr=16000):
    shift = int(random.uniform(-shift_max, shift_max) * sr)
    if shift > 0:
        augmented = np.r_[y[shift:], np.zeros(shift)]
    elif shift < 0:
        augmented = np.r_[np.zeros(-shift), y[:shift]]
    else:
        augmented = y
    return augmented

def pitch_shift(y, sr, n_steps=2):
    return librosa.effects.pitch_shift(y, sr, n_steps)

def time_stretch(y, rate=1.1):
    return librosa.effects.time_stretch(y, rate)

def random_augment(y, sr):
    funcs = [add_noise, lambda x: time_shift(x, sr=sr), lambda x: pitch_shift(x, sr, n_steps=random.choice([-2,-1,1,2])), lambda x: time_stretch(x, rate=random.uniform(0.9,1.1))]
    func = random.choice(funcs)
    try:
        return func(y)
    except Exception:
        return y
