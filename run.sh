#!/bin/bash
# Start FastAPI backend and serve index.html
uvicorn main:app --host 0.0.0.0 --port 8501
