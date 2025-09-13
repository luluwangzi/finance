#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

APP_FILE="/workspace/app.py"
if [ ! -f "$APP_FILE" ]; then
  echo "App file not found: $APP_FILE" >&2
  exit 1
fi

exec ~/.local/bin/streamlit run "$APP_FILE" --server.port 8501 --server.address 0.0.0.0 --server.headless true

