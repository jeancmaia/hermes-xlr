<#
.SYNOPSIS
    Download the Nemotron-Mini-4B-Instruct GGUF (Q4_K_M) for the S0.5 bring-up spike.
.DESCRIPTION
    Downloads from bartowski/Nemotron-Mini-4B-Instruct-GGUF on HuggingFace.
    File size: ~2.2 GB. Ensure sufficient disk space and a stable connection.
.NOTES
    HER-3 — Select & acquire a concrete INT4 model
    License: NVIDIA Open Model License
#>

$ModelDir = Split-Path -Parent $PSScriptRoot
$OutputFile = Join-Path $ModelDir "nemotron-mini-4b-instruct-q4_k_m.gguf"
$SourceUrl = "https://huggingface.co/bartowski/Nemotron-Mini-4B-Instruct-GGUF/resolve/main/Nemotron-Mini-4B-Instruct-Q4_K_M.gguf"

if (Test-Path $OutputFile) {
    $existing = (Get-Item $OutputFile).Length
    Write-Host "Model file already exists: $( $OutputFile )" -ForegroundColor Yellow
    Write-Host "Size: $( '{0:N0}' -f $existing ) bytes" -ForegroundColor Yellow
    Write-Host "Remove it first to re-download." -ForegroundColor Yellow
    exit 0
}

Write-Host "Downloading Nemotron-Mini-4B-Instruct Q4_K_M GGUF (~2.2 GB)..." -ForegroundColor Cyan
Write-Host "Source: $SourceUrl" -ForegroundColor Gray
Write-Host "Target: $OutputFile" -ForegroundColor Gray

try {
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $SourceUrl -OutFile $OutputFile -UseBasicParsing
    $downloaded = (Get-Item $OutputFile).Length
    Write-Host "Downloaded $( '{0:N0}' -f $downloaded ) bytes successfully." -ForegroundColor Green
} catch {
    Write-Host "Download failed: $_" -ForegroundColor Red
    if (Test-Path $OutputFile) { Remove-Item $OutputFile -Force }
    exit 1
}