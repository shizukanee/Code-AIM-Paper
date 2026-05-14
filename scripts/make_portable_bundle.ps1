param(
    [string]$OutputDir = "",
    [string]$HfCacheSource = "D:\Hung-Dung\caches\hf",
    [switch]$IncludeVenv = $true,
    [switch]$IncludeHfCache = $true,
    [string[]]$AdapterPaths = @(),
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Copy-Tree {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Source path not found: $Source"
    }

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    $null = robocopy $Source $Destination /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP
    if ($LASTEXITCODE -ge 8) {
        throw "robocopy failed for $Source -> $Destination (exit code: $LASTEXITCODE)"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $projectRoot "portable_bundle"
}

if ($Clean -and (Test-Path -LiteralPath $OutputDir)) {
    Write-Host "Removing existing output folder: $OutputDir"
    Remove-Item -LiteralPath $OutputDir -Recurse -Force
}

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null

$entriesToCopy = @(
    "README.md",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    "configs",
    "datasets",
    "manual_downloads",
    "melt-upstream",
    "results",
    "scripts"
)

if ($IncludeVenv) {
    $entriesToCopy += ".venv"
}

foreach ($entry in $entriesToCopy) {
    $src = Join-Path $projectRoot $entry
    $dst = Join-Path $OutputDir $entry

    if (-not (Test-Path -LiteralPath $src)) {
        Write-Host "Skipping missing path: $src"
        continue
    }

    $srcItem = Get-Item -LiteralPath $src
    if ($srcItem.PSIsContainer) {
        Write-Host "Copying directory: $entry"
        Copy-Tree -Source $src -Destination $dst
    }
    else {
        Write-Host "Copying file: $entry"
        New-Item -ItemType Directory -Path (Split-Path -Parent $dst) -Force | Out-Null
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }
}

if ($IncludeHfCache) {
    $cacheDestination = Join-Path $OutputDir "portable_cache\hf"
    Write-Host "Copying HF cache from: $HfCacheSource"
    Copy-Tree -Source $HfCacheSource -Destination $cacheDestination
}

$copiedAdapters = @()
if ($AdapterPaths.Count -gt 0) {
    $adaptersRoot = Join-Path $OutputDir "adapters"
    New-Item -ItemType Directory -Path $adaptersRoot -Force | Out-Null

    foreach ($adapterPath in $AdapterPaths) {
        $resolved = (Resolve-Path -LiteralPath $adapterPath).Path
        $adapterName = Split-Path -Leaf $resolved
        $adapterDestination = Join-Path $adaptersRoot $adapterName
        Write-Host "Copying adapter: $resolved"
        Copy-Tree -Source $resolved -Destination $adapterDestination
        $copiedAdapters += "adapters/$adapterName"
    }
}

$runnerScript = @'
param(
    [string]$Plan = "configs/model_plan.qwen25_0_5b_instruct_vs_lora.json",
    [string[]]$ExtraArgs = @("--smoke-test", "--max-models", "1", "--max-datasets", "2")
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$localHf = Join-Path $root "portable_cache\hf"
if (Test-Path -LiteralPath $localHf) {
    $env:HF_HOME = $localHf
    $env:HF_HUB_CACHE = Join-Path $localHf "hub"
    $env:HUGGINGFACE_HUB_CACHE = Join-Path $localHf "hub"
    $env:HF_DATASETS_CACHE = Join-Path $localHf "datasets"
}

$argsList = @("scripts/run_vietnamese_benchmark.py", "--plan", $Plan) + $ExtraArgs
$venvPython = Join-Path $root ".venv\Scripts\python.exe"

if (Test-Path -LiteralPath $venvPython) {
    & $venvPython @argsList
    exit $LASTEXITCODE
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "No local .venv found and uv is not installed on this machine."
}

& uv run --cache-dir (Join-Path $root ".portable_uv_cache") python @argsList
exit $LASTEXITCODE
'@

$runnerPath = Join-Path $OutputDir "run_portable_benchmark.ps1"
Set-Content -LiteralPath $runnerPath -Value $runnerScript -Encoding UTF8

$manifest = @{
    created_at = (Get-Date).ToString("s")
    output_dir = $OutputDir
    include_venv = [bool]$IncludeVenv
    include_hf_cache = [bool]$IncludeHfCache
    hf_cache_source = $HfCacheSource
    copied_adapters = $copiedAdapters
    notes = @(
        "Copy this entire output folder to another Windows machine.",
        "Run run_portable_benchmark.ps1 from inside the copied folder.",
        "If adapter_path in your model plan is still a placeholder, update it before LoRA runs."
    )
}

$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutputDir "portable_bundle_manifest.json") -Encoding UTF8

Write-Host "Portable bundle created at: $OutputDir"
