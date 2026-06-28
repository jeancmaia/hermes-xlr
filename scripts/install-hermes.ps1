<#
.SYNOPSIS
    Install Hermes Agent on Windows (native, no WSL2).
.DESCRIPTION
    Runs the official Hermes installer, verifies the install, and
    prints next steps.  Run this first, then run install-xlr.ps1.
.EXAMPLE
    .\scripts\install-hermes.ps1
#>

$ErrorActionPreference = "Stop"

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

Write-Step "Hermes Agent Installer (Windows native)"

if (Get-Command hermes -ErrorAction SilentlyContinue) {
    Write-OK "Hermes is already installed:"
    $ver = & hermes --version 2>&1
    Write-Info "  $ver"
    Write-Host ""
    Write-Host "  To reinstall, uninstall first:" -ForegroundColor Yellow
    Write-Host "    hermes uninstall" -ForegroundColor White
    exit 0
}

Write-Info "Running official Hermes installer..."
Write-Host ""

& powershell -NoProfile -Command "iex (irm https://hermes-agent.nousresearch.com/install.ps1)"

if ($LASTEXITCODE -ne 0) {
    Write-Err "Hermes installation failed."
    exit 1
}

Write-Step "Verify"

$hermesPath = Get-Command hermes -ErrorAction SilentlyContinue
if ($hermesPath) {
    Write-OK "Hermes installed: $($hermesPath.Source)"
} else {
    Write-Info "Hermes may need a shell reload to be on PATH."
    Write-Info "Reloading PATH for this session..."
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "User") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $hermesPath = Get-Command hermes -ErrorAction SilentlyContinue
    if ($hermesPath) {
        Write-OK "Hermes installed: $($hermesPath.Source)"
    } else {
        Write-Err "Hermes not found on PATH. Open a new terminal and run 'hermes --version'."
        exit 1
    }
}

Write-Host ""
Write-Host "  Next step:" -ForegroundColor Yellow
Write-Host "    .\scripts\install-xlr.ps1" -ForegroundColor White
Write-Host ""
