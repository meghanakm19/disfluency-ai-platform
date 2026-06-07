import app
print('IMPORT_OK', hasattr(app, 'app'))
client = app.app.test_client()
print('CLIENT_OK')
r = client.post('/api/train-baselines', json={
    'manifest': 'examples/feature_manifest_4.csv',
    'out_dir': 'examples/models_live'
})
print('STATUS', r.status_code)
print('JSON', r.get_json())
