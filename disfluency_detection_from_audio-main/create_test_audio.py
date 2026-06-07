#!/usr/bin/env python3
"""Create a test audio file for testing the disfluency detection system"""

import numpy as np
import wave

# Generate a simple test audio (2 seconds, 16kHz)
sample_rate = 16000
duration = 2
t = np.linspace(0, duration, sample_rate * duration)

# Create a speech-like signal with multiple frequencies
signal = (np.sin(2*np.pi*200*t) + 0.5*np.sin(2*np.pi*400*t)) * 0.5

# Add some variation to simulate stuttering (amplitude modulation)
signal[int(0.5*sample_rate):int(0.7*sample_rate)] *= 0.3
signal[int(1.0*sample_rate):int(1.2*sample_rate)] *= 0.3

# Convert to 16-bit audio
audio_data = (signal * 32767).astype(np.int16)

# Save as WAV file
with wave.open('test_audio.wav', 'w') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(audio_data.tobytes())

print('✓ Test audio file created: test_audio.wav')
