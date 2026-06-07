#!/usr/bin/env python3
"""Test the model service directly"""

import sys
sys.path.insert(0, '.')

from model_service import ModelInferenceService

# Initialize the service
print("Initializing model service...")
service = ModelInferenceService('./demo_models')

# Test with the test audio file
print("Testing with test_audio.wav...")
result = service.analyze('./test_audio.wav', 'multimodal')

print(f"Result: {result}")
