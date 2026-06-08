import sys
sys.path.insert(0, '.')

from model_service import ModelInferenceService

service = ModelInferenceService('./demo_models')
result = service.analyze('./test_audio.wav', 'multimodal')
print(result)
