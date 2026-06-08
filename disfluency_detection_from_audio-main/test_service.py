#!/usr/bin/env python3
"""Test the full model service"""

import sys
sys.path.insert(0, '.')

from model_service import ModelInferenceService

print("Initializing ModelInferenceService...")
service = ModelInferenceService('./demo_models')

print("Testing with test_audio.wav...")
print("Calling service.analyze()...")
result = service.analyze('./test_audio.wav', 'acoustic')

print(f"\nResult: {result}")

if result['success']:
    print("✓ SUCCESS!")
    data = result['data']
    print(f"  Modality: {data['modality']}")
    print(f"  Disfluency count: {data['disfluency_count']}")
    print(f"  Confidence: {data['confidence']}")
else:
    print(f"✗ FAILED: {result['error']}")
