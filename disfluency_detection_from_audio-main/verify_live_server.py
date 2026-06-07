import json
import urllib.request

base = 'http://127.0.0.1:5000'

for path in ('/api/health', '/api/stats'):
    with urllib.request.urlopen(base + path) as r:
        print(path, 'STATUS', r.status)
        print(r.read().decode('utf-8', 'ignore'))

req = urllib.request.Request(
    base + '/api/train-baselines',
    data=json.dumps({
        'manifest': 'examples/feature_manifest_4.csv',
        'out_dir': 'examples/models_live'
    }).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST'
)
with urllib.request.urlopen(req) as r:
    print('/api/train-baselines STATUS', r.status)
    print(r.read().decode('utf-8', 'ignore'))
