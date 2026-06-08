#!/usr/bin/env python3
"""Simple test - just load audio"""

import sys
sys.path.insert(0, '.')

import torch
import librosa

print("Test 1: Loading audio with librosa...")
audio_np, sr = librosa.load('./test_audio.wav', sr=16000, mono=True)
print(f"✓ Loaded audio: shape={audio_np.shape}, sr={sr}")

audio = torch.FloatTensor(audio_np).unsqueeze(0)
print(f"✓ Converted to tensor: shape={audio.shape}")

print("\nTest 2: Loading WavLM model...")
from transformers import WavLMModel
wavlm = WavLMModel.from_pretrained('microsoft/wavlm-base')
print(f"✓ WavLM loaded")

print("\nTest 3: Running feature extractor...")
feats = wavlm.feature_extractor(audio)
print(f"✓ Features extracted: shape={feats.shape}")

print("\nTest 4: Transposing features...")
feats = feats.transpose(1, 2)
print(f"✓ Transposed: shape={feats.shape}")

print("\nTest 5: Feature projection...")
feats, _ = wavlm.feature_projection(feats)
print(f"✓ Projected: shape={feats.shape}")

print("\nTest 6: Encoder...")
emb = wavlm.encoder(feats, return_dict=True)[0]
print(f"✓ Encoded: shape={emb.shape}")

print("\n✓ ALL TESTS PASSED!")
