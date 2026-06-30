<#
.SYNOPSIS
    Launch an XLR-tuned llama.cpp engine for Hermes Agent.
.DESCRIPTION
    Detects your GPU, generates an execution plan, and starts a tuned
    llama-server with optimal settings (KV-cache dtype, CUDA graphs,
    speculative decoding, context size — all plan-driven).

    When the engine is ready, prints instructions for connecting
    Hermes Agent to it as a Custom endpoint provider.
.PARAMETER BinaryPath
    Path to llama-server.exe. Defaults to bin/llama-server.exe.
.PARAMETER ModelPath
    Path to a GGUF model file. If omitted, the script checks
    XLR_MODEL_PATH env var and common locations.
.PARAMETER Port
    Port to serve on (default: 8080).
.EXAMPLE
    .\scripts\start-xlr-engine.ps1
.EXAMPLE
    .\scripts\start-xlr-engine.ps1 -BinaryPath C:\tools\llama-server.exe -ModelPath C:\models\Llama-3.2-3B-Instruct-Q4_K_M.gguf
.NOTES
    Press Ctrl+C to stop the engine.
#>

param(
    [string]$BinaryPath = (Join-Path (Split-Path -Parent $PSScriptRoot) "bin\llama-server.exe"),
    [string]$ModelPath = $env:XLR_MODEL_PATH,
    [int]$Port = 8080
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

# --- Find Hermes venv (same logic as install-xlr.ps1) ------------------------

$hermesHome = Join-Path $env:USERPROFILE ".hermes"
$hermesVenv = Join-Path $hermesHome "hermes-agent\venv"

if (-not (Test-Path $hermesVenv)) {
    $hermesHome = Join-Path $env:LOCALAPPDATA "hermes"
    $hermesVenv = Join-Path $hermesHome "hermes-agent\venv"
}

if (Test-Path $hermesVenv) {
    $hermesPython = Join-Path $hermesVenv "Scripts\python.exe"
} else {
    # Fallback: look for a local .venv in the repo root
    $hermesPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
}

function Write-Step($msg) {
    Write-Host ""
    Write-Host "=== $msg ===" -ForegroundColor Cyan
}

function Write-OK($msg) {
    Write-Host "  $msg" -ForegroundColor Green
}

function Write-Err($msg) {
    Write-Host "  $msg" -ForegroundColor Red
}

function Write-Info($msg) {
    Write-Host "  $msg" -ForegroundColor Gray
}

# --- Check binary -----------------------------------------------------------

if (-not (Test-Path $BinaryPath)) {
    Write-Err "llama-server not found at: $BinaryPath"
    Write-Host ""
    Write-Host "  Options:" -ForegroundColor Yellow
    Write-Host "    1. Run scripts\download-cuda-engine.ps1 to fetch it"
    Write-Host "    2. Pass -BinaryPath <path-to-llama-server.exe>"
    Write-Host "    3. Set XLR_BINARY_PATH env var"
    exit 1
}

# --- Check model ------------------------------------------------------------

if (-not $ModelPath) {
    $candidates = @(
        (Join-Path $repoRoot "models\Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
        (Join-Path $repoRoot "models\Nemotron-Mini-4B-Instruct-Q4_K_M.gguf"),
        (Join-Path $repoRoot "models\nemotron-mini-4b-instruct-q4_k_m.gguf")
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $ModelPath = $c; break }
    }
}

if (-not $ModelPath -or -not (Test-Path $ModelPath)) {
    Write-Err "Model file not found."
    Write-Host ""
    Write-Host "  Options:" -ForegroundColor Yellow
    Write-Host "    1. Pass -ModelPath <path-to-model.gguf>"
    Write-Host "    2. Set XLR_MODEL_PATH env var"
    Write-Host ""
    Write-Host "  Download a model:" -ForegroundColor Yellow
    Write-Host "    curl -L -o models\Llama-3.2-3B-Instruct-Q4_K_M.gguf ``"
    Write-Host "      https://huggingface.co/QuantFactory/Meta-Llama-3.2-3B-Instruct-GGUF/resolve/main/Meta-Llama-3.2-3B-Instruct.Q4_K_M.gguf"
    exit 1
}

Write-Step "XLR Engine Launcher"
Write-OK "Binary:  $BinaryPath"
Write-OK "Model:   $ModelPath"
Write-OK "Port:    $Port"

# --- Detect + Plan ----------------------------------------------------------

Write-Step "DETECT + PLAN"

$planJson = & $hermesPython -c "
import json, sys, os
sys.path.insert(0, r'$repoRoot')
from hermes_nim_xlr.mapper.detect import detect
from hermes_nim_xlr.mapper import plan, catalog

host = detect()
p = plan(host, min_context_tokens=65536)

# Try to match the loaded GGUF to a catalog entry so derived parameters
# reflect the actual engine model, not the plan's VRAM-based selection.
model_path = r'$ModelPath'
loaded_repo = None
if os.path.exists(model_path):
    fname = os.path.basename(model_path).lower()
    name_map = {
        'llama-3.2-3b': 'meta-llama/Llama-3.2-3B-Instruct',
        'llama-3.2-1b': 'Qwen/Qwen2.5-1.5B-Instruct',
        'qwen2.5-1.5b': 'Qwen/Qwen2.5-1.5B-Instruct',
        'qwen2.5-0.5b': 'Qwen/Qwen2.5-0.5B-Instruct',
        'nemotron-mini-4b': 'nvidia/Nemotron-Mini-4B-Instruct',
        'phi-3.5-mini': 'microsoft/Phi-3.5-mini-instruct',
        'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.3',
        'llama-3.1-8b': 'meta-llama/Llama-3.1-8B-Instruct',
        'gemma-2-9b': 'google/Gemma-2-9b-it',
    }
    for key, repo in name_map.items():
        if key in fname:
            loaded_repo = repo
            break

model_override = False
if loaded_repo and loaded_repo != p.model.repo:
    matched = [m for m in catalog.CATALOG if m.repo == loaded_repo and m.weight_quant is p.model.weight_quant]
    if matched:
        loaded = matched[0]
        import dataclasses
        p = dataclasses.replace(p, model=loaded)
        model_override = True

print(json.dumps({
    'model': p.model.repo,
    'model_override': model_override,
    'gpu_layers': p.placement.gpu_layers,
    'total_layers': p.placement.total_layers,
    'ctx_tokens': p.target_ctx_tokens,
    'kv_dtype': p.kv.dtype.value,
    'cache_type_k': p.kv.cache_type_k,
    'cache_type_v': p.kv.cache_type_v,
    'cuda_graphs': p.levers.cuda_graphs,
    'spec_decode': p.levers.spec_decode.value,
    'est_vram_mb': p.est_vram_mb,
    'gpus': [g.name for g in host.gpus],
    'rationale': list(p.rationale),
}, indent=2))
" 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Err "Plan generation failed:"
    Write-Host $planJson
    exit 1
}

$plan = $planJson | ConvertFrom-Json

Write-OK "GPU:        $($plan.gpus -join ', ')"
Write-OK "Model:      $($plan.model)"
Write-OK "VRAM est:   $($plan.est_vram_mb) MiB"
Write-OK "Context:    $($plan.ctx_tokens) tokens"
Write-OK "KV dtype:   $($plan.kv_dtype)"
Write-OK "CUDA graphs: $($plan.cuda_graphs)"
Write-OK "Spec decode: $($plan.spec_decode)"
Write-OK "GPU layers: $($plan.gpu_layers)/$($plan.total_layers)"
foreach ($note in $plan.rationale) {
    Write-Info "  $note"
}

# Override ctx-size when the actual model supports more context than
# the plan's catalog entry suggests (e.g. Llama-3.2-3B has 131K but
# the plan may have selected a different catalog entry).
# Hermes Agent needs at least 8K context.
$minCtx = 65536
if ($plan.ctx_tokens -lt $minCtx) {
    Write-Info "Raising ctx-size from $($plan.ctx_tokens) to $minCtx (Hermes minimum)"
    $plan.ctx_tokens = $minCtx
}

# --- Build llama-server args ------------------------------------------------

$serverArgs = @(
    "--host", "127.0.0.1",
    "--port", $Port,
    "--model", $ModelPath,
    "--n-gpu-layers", $plan.gpu_layers,
    "--ctx-size", $plan.ctx_tokens,
    "--jinja",
    "--cache-prompt"
)

# Note: --cuda-graphs removed — this llama-server build enables CUDA
# graphs automatically when available.

if ($plan.spec_decode -eq "ngram") {
    $serverArgs += @("--speculative-ngram", "32")
}

if ($plan.cache_type_k) {
    $serverArgs += @("--cache-type-k", $plan.cache_type_k)
}
if ($plan.cache_type_v) {
    $serverArgs += @("--cache-type-v", $plan.cache_type_v)
}

# --- Start engine -----------------------------------------------------------

Write-Step "START"

$endpoint = "http://127.0.0.1:$Port/v1"

Write-Info "Launching llama-server..."
Write-Info "  $($BinaryPath) $($serverArgs -join ' ')"
Write-Host ""

$process = Start-Process -FilePath $BinaryPath -ArgumentList $serverArgs -PassThru -NoNewWindow

# --- Wait for health --------------------------------------------------------

Write-Info "Waiting for engine to become healthy..."
$healthy = $false
$deadline = (Get-Date).AddSeconds(30)

while (-not $healthy -and (Get-Date) -lt $deadline) {
    if ($process.HasExited) {
        Write-Err "llama-server exited with code $($process.ExitCode)"
        exit 1
    }
    Start-Sleep -Milliseconds 500
    try {
        $response = Invoke-RestMethod -Uri "$endpoint/models" -Method Get -TimeoutSec 2
        $healthy = $true
    } catch {
        # still warming up
    }
}

if (-not $healthy) {
    Write-Err "Engine did not become healthy within 30 seconds."
    if (-not $process.HasExited) { Stop-Process -Id $process.Id -Force }
    exit 1
}

Write-OK "Engine ready at: $endpoint"

# --- Print Hermes instructions ----------------------------------------------

Write-Step "READY -- connect Hermes Agent"

Write-Host ""
Write-Host "  In another terminal, run:" -ForegroundColor Yellow
Write-Host ""
Write-Host "    hermes model" -ForegroundColor White
Write-Host "    -> Custom endpoint (self-hosted / VLLM / etc.)" -ForegroundColor Gray
Write-Host "    -> $endpoint" -ForegroundColor White
Write-Host "    -> (no API key)" -ForegroundColor Gray
Write-Host "    -> (press Enter to auto-detect model)" -ForegroundColor Gray
Write-Host ""
Write-Host "    hermes" -ForegroundColor White
Write-Host ""

Write-Host "  Press Ctrl+C to stop the engine." -ForegroundColor Gray
Write-Host ""

# --- Wait for Ctrl+C --------------------------------------------------------

try {
    while (-not $process.HasExited) {
        Start-Sleep -Seconds 1
    }
} finally {
    if (-not $process.HasExited) {
        Write-Host ""
        Write-Info "Stopping engine..."
        Stop-Process -Id $process.Id -Force
        Start-Sleep -Seconds 1
    }
    Write-OK "Engine stopped."
}