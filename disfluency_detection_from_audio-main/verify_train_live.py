import json
import urllib.request

req = urllib.request.Request(
    'http://localhost:5000/api/train-baselines',
    data=json.dumps({
        'manifest': 'examples/feature_manifest_4.csv',
        'out_dir': 'examples/models_live'
    }).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST'
)

with urllib.request.urlopen(req) as r:
    print('STATUS', r.status)
    print(r.read().decode('utf-8', 'ignore'))
