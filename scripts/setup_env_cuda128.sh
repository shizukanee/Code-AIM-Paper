#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${1:-/data/caches/uv}"
VENV_PATH="${2:-order_kd/.venv311}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "Creating venv at ${VENV_PATH} with Python 3.11"
uv --cache-dir "${CACHE_DIR}" venv --python 3.11 "${VENV_PATH}"

PYTHON_BIN="${VENV_PATH}/bin/python"

echo "Installing base requirements"
uv --cache-dir "${CACHE_DIR}" pip install --python "${PYTHON_BIN}" -r order_kd/requirements/base.txt

echo "Installing CUDA 12.8 torch"
uv --cache-dir "${CACHE_DIR}" pip install --python "${PYTHON_BIN}" -r order_kd/requirements/torch-cu128.txt

echo "Installing training extras"
uv --cache-dir "${CACHE_DIR}" pip install --python "${PYTHON_BIN}" -r order_kd/requirements/train-extra.txt

echo "Sanity checking transformers Trainer import"
"${PYTHON_BIN}" -c "from transformers import Trainer; print('Trainer import ok')"

echo "Environment ready: ${PYTHON_BIN}"
