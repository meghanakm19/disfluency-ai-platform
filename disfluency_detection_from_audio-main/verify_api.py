import urllib.request

base = 'http://localhost:5000'
for path in ('/api/health', '/api/stats'):
    try:
        with urllib.request.urlopen(base + path) as r:
            print(path, 'STATUS', r.status)
            print(r.read().decode('utf-8', 'ignore'))
    except Exception as e:
        print(path, 'ERROR', getattr(e, 'code', None), str(e))
