<#
.SYNOPSIS
    Download CUDA-accelerated llama-server for Hermes-NIM-XLR.
.DESCRIPTION
    Fetches the pre-built CUDA llama.cpp release (b9763, CUDA 12.4)
    from GitHub and extracts it to bin/. Also fetches the matching
    CUDA 12.4 runtime DLLs from the cudart companion package.
.NOTES
    HER-15 -- CUDA llama.cpp bring-up + tool-call validation
    Target: bin/llama-server.exe + supporting DLLs
    Runtime requirement: NVIDIA driver >= 525.60 (CUDA 12.x)
#>

param(
    [string]$BinDir = (Join-Path (Split-Path -Parent $PSScriptRoot) "bin"),
    [string]$Tag = "b9763",
    [string]$CudaVer = "12.4"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
}

$baseUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$Tag"

$packages = @(
    @{ Name = "llama-$Tag-bin-win-cuda-$CudaVer-x64.zip"; Desc = "llama-server binaries" },
    @{ Name = "cudart-llama-bin-win-cuda-$CudaVer-x64.zip"; Desc = "CUDA runtime DLLs" }
)

foreach ($pkg in $packages) {
    $url = "$baseUrl/$($pkg.Name)"
    $zipPath = Join-Path $env:TEMP $pkg.Name
    Write-Host "Downloading $($pkg.Desc)..." -ForegroundColor Cyan
    Write-Host "  $url" -ForegroundColor Gray
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $url -OutFile $zipPath -ErrorAction Stop
    } catch {
        Write-Host "Download failed: $_" -ForegroundColor Red
        exit 1
    }
    Write-Host "  Downloaded $((Get-Item $zipPath).Length) bytes" -ForegroundColor Green

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipPath, $BinDir)
    Write-Host "  Extracted to $BinDir" -ForegroundColor Green
    Remove-Item $zipPath -Force
}

Write-Host ""
Write-Host "CUDA engine ready at: $BinDir\llama-server.exe" -ForegroundColor Green
