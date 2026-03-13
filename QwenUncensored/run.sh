#!/bin/bash

# Backend: FastAPI app using Qwen model on port 8501
uvicorn qwen_backend:app --host 0.0.0.0 --port 8501 &

# Frontend: serve index.html and static assets from this directory on port 8000
cd "$(dirname "$0")"
python -m http.server 8000 --bind 0.0.0.0

