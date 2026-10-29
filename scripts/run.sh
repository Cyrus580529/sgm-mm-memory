#!/usr/bin/env bash
# Start the Add/Search service. Reads settings from .env next to this repo.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; [ -f .env ] && . ./.env; set +a
exec .venv/bin/uvicorn app.main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --workers 1
