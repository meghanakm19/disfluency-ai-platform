import app

client = app.app.test_client()
r = client.post('/api/train-baselines', json={
    'manifest': 'examples/feature_manifest_4.csv',
    'out_dir': 'examples/models_live'
})
print('status=', r.status_code)
print('json=', r.get_json())
