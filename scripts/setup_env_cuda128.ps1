param(
    [string]$CacheDir = "D:\Hung-Dung\caches\uv",
    [string]$VenvPath = "order_kd\.venv311"
)

$ErrorActionPreference = "Stop"

Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\.."))

Write-Host "Creating venv at $VenvPath with Python 3.11"
uv --cache-dir "$CacheDir" venv --python 3.11 "$VenvPath"

$PythonExe = Join-Path $VenvPath "Scripts\python.exe"

Write-Host "Installing base requirements"
uv --cache-dir "$CacheDir" pip install --python "$PythonExe" -r order_kd/requirements/base.txt

Write-Host "Installing CUDA 12.8 torch"
uv --cache-dir "$CacheDir" pip install --python "$PythonExe" -r order_kd/requirements/torch-cu128.txt

Write-Host "Installing training extras"
uv --cache-dir "$CacheDir" pip install --python "$PythonExe" -r order_kd/requirements/train-extra.txt

Write-Host "Sanity checking transformers Trainer import"
& "$PythonExe" -c "from transformers import Trainer; print('Trainer import ok')"

Write-Host "Environment ready: $PythonExe"
