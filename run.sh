#!/bin/bash
# Start FastAPI backend on port 8501
uvicorn main:app --host 0.0.0.0 --port 8501 &

# Serve index.html and static files on port 8000
python -m http.server 8000 --bind 0.0.0.0
