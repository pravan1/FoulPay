import sys
import os

# Add parent directory to path so we can import from backend modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

# Vercel expects a variable named 'app' in the api/index.py file
# This exports the FastAPI app for Vercel's Python runtime