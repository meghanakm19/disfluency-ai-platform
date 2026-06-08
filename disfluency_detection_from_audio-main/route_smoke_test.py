import sys
sys.path.insert(0, '.')

import app as app_module
from app import app

class FakeInferenceService:
    def analyze(self, *args, **kwargs):
        return {
            'success': True,
            'data': {
                'modality': 'acoustic',
                'predictions': [],
                'statistics': {
                    'total_frames': 1,
                    'disfluent_frames': 0,
                    'fluent_frames': 1,
                    'disfluency_rate': 0.0,
                    'FP_count': 0,
                    'FP_avg_confidence': 0.0,
                    'RP_count': 0,
                    'RP_avg_confidence': 0.0,
                    'RV_count': 0,
                    'RV_avg_confidence': 0.0,
                    'RS_count': 0,
                    'RS_avg_confidence': 0.0,
                    'PW_count': 0,
                    'PW_avg_confidence': 0.0,
                },
                'disfluency_count': 0,
                'confidence': 0.0,
            },
        }


app_module.inference_service = FakeInferenceService()

client = app.test_client()
with open('test_audio.wav', 'rb') as audio_file:
    response = client.post(
        '/api/analyze',
        data={
            'audio': (audio_file, 'test_audio.wav'),
            'modality': 'acoustic',
            'user_id': 'client-test',
            'notes': 'client route',
        },
        content_type='multipart/form-data',
    )

print('STATUS', response.status_code)
print(response.data.decode('utf-8')[:2000])
