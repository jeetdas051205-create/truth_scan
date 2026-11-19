"""
Root Entrypoint for TruthScan AI Web Application
===============================================
Redirects execution into CS3244-Fake-News-Detection-master/app.py.
Allows users to launch with:
    python app.py
from the workspace root directory.
"""

import sys
import os

SUBDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CS3244-Fake-News-Detection-master")
if SUBDIR not in sys.path:
    sys.path.insert(0, SUBDIR)

from app import app

if __name__ == "__main__":
    print("[*] Starting TruthScan AI Server at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)

