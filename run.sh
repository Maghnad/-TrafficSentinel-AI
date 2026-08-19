#!/usr/bin/env bash
# Launch the dashboard. Assumes ./setup.sh has been run once.
set -e
cd "$(dirname "$0")"
if [ ! -d ".venv" ]; then
    echo "No .venv found. Run ./setup.sh first."
    exit 1
fi
source .venv/bin/activate
exec streamlit run app.py
