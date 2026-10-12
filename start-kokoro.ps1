#Requires -Version 5.1

<#
.SYNOPSIS
    Launch Kokoro TTS with Gradio environment configuration.

.DESCRIPTION
    Configures Gradio without modifying application source.
    Uses the existing Python virtual environment.
#>

$ErrorActionPreference = "Stop"

# --------------------------------------------------
# 1. Resolve project and virtual environment
# --------------------------------------------------

$ProjectDir = $PSScriptRoot

$VenvDir = Join-Path $ProjectDir "venv"
$ScriptsDir = Join-Path $VenvDir "Scripts"

$PythonExe = Join-Path $ScriptsDir "python.exe"
$KokoroExe = Join-Path $ScriptsDir "kokoro-tts-web.exe"

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment not found: $PythonExe"
}

if (-not (Test-Path $KokoroExe)) {
    throw "Kokoro executable not found: $KokoroExe"
}

# --------------------------------------------------
# 2a. Configure Kokoro output directory (superceeded by 2b)
# --------------------------------------------------

# $OutputDir = Join-Path $env:LOCALAPPDATA `
#     "kokoro-tts-local\outputs"

# if (-not (Test-Path $OutputDir)) {
#     New-Item -ItemType Directory `
#         -Path $OutputDir -Force | Out-Null
# }

# --------------------------------------------------
# 2b. Configure Kokoro output directory
# --------------------------------------------------

$OutputDir = Join-Path $ProjectDir "output"

if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory `
        -Path $OutputDir -Force | Out-Null
}

# Tell Kokoro to use the project directory for its data.
$env:KOKORO_BASE_DIR = $ProjectDir

# Tell Gradio where generated WAV files can be served.
$env:GRADIO_ALLOWED_PATHS = $OutputDir


# --------------------------------------------------
# 3. Configure Gradio environment
# --------------------------------------------------

# Allow Gradio to serve generated WAV files.
$env:GRADIO_ALLOWED_PATHS = $OutputDir

# Listen only on the local machine.
#$env:GRADIO_SERVER_NAME = "127.0.0.1"
$env:GRADIO_SERVER_NAME = "0.0.0.0"

# Default web interface port.
$env:GRADIO_SERVER_PORT = "7860"

# Disable externally accessible share links.
$env:GRADIO_SHARE = "False"

# Disable Gradio analytics.
$env:GRADIO_ANALYTICS_ENABLED = "False"

# --------------------------------------------------
# 4. Configure Python virtual environment
# --------------------------------------------------

$env:VIRTUAL_ENV = $VenvDir
$env:PATH = "$ScriptsDir;$env:PATH"

# Prevent Python from using user site-packages.
$env:PYTHONNOUSERSITE = "1"

# --------------------------------------------------
# 5. Display configuration
# --------------------------------------------------

Write-Host ""
Write-Host "====================================="
Write-Host "       KOKORO TTS LAUNCHER"
Write-Host "====================================="
Write-Host ""

Write-Host "Project: $ProjectDir"
Write-Host "Python:  $PythonExe"
Write-Host "Output:  $OutputDir"

Write-Host ""
Write-Host "Gradio configuration:"
Write-Host "  Server:        $env:GRADIO_SERVER_NAME"
Write-Host "  Port:          $env:GRADIO_SERVER_PORT"
Write-Host "  Allowed paths: $env:GRADIO_ALLOWED_PATHS"
Write-Host "  Public share:  $env:GRADIO_SHARE"

Write-Host ""

# --------------------------------------------------
# 6. Launch Kokoro
# --------------------------------------------------

Set-Location $ProjectDir

Write-Host "Starting Kokoro..."
Write-Host ""

& $KokoroExe

if ($LASTEXITCODE -ne 0) {
    throw "Kokoro exited with code $LASTEXITCODE"
}