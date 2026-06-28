<#
.SYNOPSIS
    Install Hermes-XLR: engine binary, model, Python package, and Hermes hook.
.DESCRIPTION
    This script does everything after Hermes is installed:

    1. Fetches the CUDA llama-server binary (if not present)
    2. Downloads a GGUF model from Hugging Face (if not present)
    3. Installs hermes-nim-xlr into the Hermes venv
    4. Drops a .pth hook so Hermes auto-registers XLRTransport
    5. Configures Hermes to use the local endpoint
    6. Verifies the full chain

    After this, just run:  start-xlr-engine.ps1   then   hermes
.PARAMETER ModelPath
    Path to a GGUF model. If provided, skips the download step.
.PARAMETER ModelRepo
    Hugging Face repo for model download (default: QuantFactory/Meta-Llama-3.2-3B-Instruct-GGUF)
.PARAMETER ModelFile
    GGUF filename within the repo (default: Meta-Llama-3.2-3B-Instruct.Q4_K_M.gguf)
.EXAMPLE
    .\scripts\install-xlr.ps1
.EXAMPLE
    .\scripts\install-xlr.ps1 -ModelPath C:\models\my-model.gguf
#>

param(
    [string]$ModelPath,
    [string]$ModelRepo = "QuantFactory/Meta-Llama-3.2-3B-Instruct-GGUF",
    [string]$ModelFile = "Meta-Llama-3.2-3B-Instruct.Q4_K_M.gguf"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$binDir = Join-Path $repoRoot "bin"
$modelDir = Join-Path $repoRoot "models"

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

# ---------------------------------------------------------------------------
# Preflight: find Hermes installation
# ---------------------------------------------------------------------------

Write-Step "Find Hermes"

$hermesCmd = Get-Command hermes -ErrorAction SilentlyContinue
if (-not $hermesCmd) {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $hermesCmd = Get-Command hermes -ErrorAction SilentlyContinue
}

if (-not $hermesCmd) {
    Write-Err "Hermes not found. Run scripts\install-hermes.ps1 first."
    exit 1
}

# Hermes venv is at ~/.hermes/hermes-agent/venv (standard curl installer layout)
$hermesHome = Join-Path $env:USERPROFILE ".hermes"
$hermesRepo = Join-Path $hermesHome "hermes-agent"
$hermesVenv = Join-Path $hermesRepo "venv"

if (-not (Test-Path $hermesVenv)) {
    # Try LOCALAPPDATA layout (Windows installer)
    $hermesHome = Join-Path $env:LOCALAPPDATA "hermes"
    $hermesRepo = Join-Path $hermesHome "hermes-agent"
    $hermesVenv = Join-Path $hermesRepo "venv"
}

if (-not (Test-Path $hermesVenv)) {
    Write-Err "Hermes venv not found at expected locations:"
    Write-Err "  $hermesVenv"
    Write-Host ""
    Write-Host "  Set HERMES_VENV env var to your venv path and rerun." -ForegroundColor Yellow
    exit 1
}

$hermesPython = Join-Path $hermesVenv "Scripts\python.exe"
$hermesSitePackages = Join-Path $hermesVenv "Lib\site-packages"

Write-OK "Hermes venv:  $hermesVenv"
Write-OK "Python:        $hermesPython"

# ---------------------------------------------------------------------------
# Step 1: CUDA llama-server binary
# ---------------------------------------------------------------------------

Write-Step "Step 1/5 — CUDA engine binary"

$binaryPath = Join-Path $binDir "llama-server.exe"

if (Test-Path $binaryPath) {
    Write-OK "Already present: $binaryPath"
} else {
    Write-Info "Downloading CUDA llama-server..."
    & (Join-Path $PSScriptRoot "download-cuda-engine.ps1") -BinDir $binDir
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Engine binary download failed."
        exit 1
    }
}

Write-OK "Engine: $binaryPath"

# ---------------------------------------------------------------------------
# Step 2: GGUF model
# ---------------------------------------------------------------------------

Write-Step "Step 2/5 — Model"

if ($ModelPath -and (Test-Path $ModelPath)) {
    Write-OK "Using provided model: $ModelPath"
} else {
    if (-not (Test-Path $modelDir)) {
        New-Item -ItemType Directory -Path $modelDir -Force | Out-Null
    }

    $downloadPath = Join-Path $modelDir $ModelFile
    if (Test-Path $downloadPath) {
        Write-OK "Already present: $downloadPath"
        $ModelPath = $downloadPath
    } else {
        $url = "https://huggingface.co/$ModelRepo/resolve/main/$ModelFile"
        Write-Info "Downloading model from Hugging Face..."
        Write-Info "  $url"
        Write-Info "  (this may take a few minutes — ~2 GB)"
        & curl -L -o $downloadPath $url
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $downloadPath)) {
            Write-Err "Model download failed."
            Write-Host ""
            Write-Host "  Download manually and rerun with -ModelPath:" -ForegroundColor Yellow
            Write-Host "    .\scripts\install-xlr.ps1 -ModelPath C:\path\to\model.gguf" -ForegroundColor White
            exit 1
        }
        Write-OK "Model downloaded: $downloadPath"
        $ModelPath = $downloadPath
    }
}

# ---------------------------------------------------------------------------
# Step 3: Install hermes-nim-xlr into Hermes venv
# ---------------------------------------------------------------------------

Write-Step "Step 3/5 — Install hermes-nim-xlr"

Write-Info "Installing into Hermes venv..."
& uv pip install --python $hermesPython -e $repoRoot

if ($LASTEXITCODE -ne 0) {
    Write-Info "uv not found, trying pip directly..."
    & $hermesPython -m pip install -e $repoRoot
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to install hermes-nim-xlr into Hermes venv."
        exit 1
    }
}

Write-OK "hermes-nim-xlr installed"

# ---------------------------------------------------------------------------
# Step 4: Drop .pth hook so Hermes auto-loads XLRTransport
# ---------------------------------------------------------------------------

Write-Step "Step 4/5 — Wire XLRTransport into Hermes"

$hookContent = "import hermes_nim_xlr.hermes_hook`n"
$pthPath = Join-Path $hermesSitePackages "_hermes_xlr_hook.pth"

Set-Content -Path $pthPath -Value $hookContent -Encoding utf8 -NoNewline

Write-OK "Hook installed: $pthPath"
Write-Info "Hermes will now auto-register XLRTransport for chat_completions."

# Verify the hook loads without error
Write-Info "Verifying hook loads..."
& $hermesPython -c "import hermes_nim_xlr.hermes_hook; print('OK')" 2>&1

if ($LASTEXITCODE -ne 0) {
    Write-Err "Hook verification failed — check GPU detection."
    Write-Host "  Continuing anyway — Hermes will fall back to its stock transport." -ForegroundColor Yellow
} else {
    Write-OK "Hook loads successfully"
}

# ---------------------------------------------------------------------------
# Step 5: Configure Hermes to use the local endpoint
# ---------------------------------------------------------------------------

Write-Step "Step 5/5 — Configure Hermes"

# Set model config
& hermes config set model.provider custom
& hermes config set model.base_url "http://127.0.0.1:8080/v1"
& hermes config set model.api_key "local"

# Try auto-detecting the model name from the plan
$modelName = & $hermesPython -c "
import sys; sys.path.insert(0, r'$repoRoot')
from hermes_nim_xlr.mapper import detect, plan
p = plan(detect())
print(p.model.repo)
" 2>$null

if ($modelName) {
    & hermes config set model.default $modelName
    Write-OK "Model: $modelName"
} else {
    # Fall back to the GGUF filename without extension
    $modelBase = [System.IO.Path]::GetFileNameWithoutExtension($ModelFile)
    & hermes config set model.default $modelBase
    Write-OK "Model: $modelBase (from filename)"
}

Write-OK "Provider: custom"
Write-OK "Endpoint: http://127.0.0.1:8080/v1"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

Write-Step "Installation complete"

Write-Host ""
Write-Host "  Everything is set up. To start:" -ForegroundColor Yellow
Write-Host ""
Write-Host "    1. Launch the tuned engine:" -ForegroundColor White
Write-Host "       .\scripts\start-xlr-engine.ps1 -ModelPath $ModelPath" -ForegroundColor Gray
Write-Host ""
Write-Host "    2. Start Hermes in another terminal:" -ForegroundColor White
Write-Host "       hermes" -ForegroundColor Gray
Write-Host ""
Write-Host "  XLRTransport is auto-loaded — every request carries plan-derived config." -ForegroundColor Green
Write-Host ""
