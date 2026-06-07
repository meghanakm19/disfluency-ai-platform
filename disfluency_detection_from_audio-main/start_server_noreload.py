#!/usr/bin/env python3
"""Start the Flask server without watchdog"""

import os
os.environ['FLASK_ENV'] = 'development'

from app import app, db, init_models

# Initialize
db.init_db()
init_models()

print("Starting server...")
app.run(
    host='0.0.0.0',
    port=int(os.getenv('FLASK_PORT', 5000)),
    debug=False,
    use_reloader=False,
    threaded=True
)
