#!/usr/bin/env bash
# Start script that binds to the port provided by Render (or $PORT)
set -euo pipefail

# Use the PORT env var when available (Render sets this). Fallback to 8000 for local dev.
PORT="${PORT:-8000}"
echo "Starting uvicorn on 0.0.0.0:${PORT}"
exec uvicorn src.app:app --host 0.0.0.0 --port "${PORT}"
