#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

VENV_DIR="${VENV_DIR:-.venv311}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/uv}"
PYTHON_BIN="${VENV_DIR}/bin/python"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install uv first (https://github.com/astral-sh/uv)." >&2
  exit 1
fi

echo "Creating venv at ${VENV_DIR}"
uv --cache-dir "${CACHE_DIR}" venv --python 3.11 "${VENV_DIR}"

echo "Installing base requirements"
uv --cache-dir "${CACHE_DIR}" pip install --python "${PYTHON_BIN}" -r requirements/base.txt

echo "Installing CUDA 12.8 torch"
uv --cache-dir "${CACHE_DIR}" pip install --python "${PYTHON_BIN}" -r requirements/torch-cu128.txt

echo "Installing training extras"
uv --cache-dir "${CACHE_DIR}" pip install --python "${PYTHON_BIN}" -r requirements/train-extra.txt

echo "Ready. Activate with: source ${VENV_DIR}/bin/activate"
