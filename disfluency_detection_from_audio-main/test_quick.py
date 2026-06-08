#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from model_service import ModelInferenceService

service = ModelInferenceService('./demo_models')
result = service.analyze('./test_audio.wav', 'acoustic')
print('SUCCESS' if result['success'] else f"FAILED: {result['error']}")
if result['success']:
    print(f"Disfluency count: {result['data']['disfluency_count']}")
    print(f"Frames: {len(result['data']['predictions'])}")
