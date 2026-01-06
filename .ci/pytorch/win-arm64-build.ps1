# If you want to rebuild, run this with $env:REBUILD=1
# If you want to build with CUDA, run this with $env:USE_CUDA=1
# If you want to build without CUDA, run this with $env:USE_CUDA=0

# Check for setup.py in the current directory
if (-not (Test-Path "setup.py")) {
    Write-Host "ERROR: Please run this build script from PyTorch root directory."
    exit 1
}

# Get the script's parent directory
$ScriptParentDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Set TMP_DIR and convert to Windows path
$env:TMP_DIR = Join-Path (Get-Location) "build\win_tmp"
$env:TMP_DIR_WIN = $env:TMP_DIR  # Already in Windows format, no cygpath needed

# Set final package directory with default fallback
if (-not $env:PYTORCH_FINAL_PACKAGE_DIR) {
    $env:PYTORCH_FINAL_PACKAGE_DIR = "C:\w\build-results"
}

# Create the final package directory if it doesn't exist
if (-not (Test-Path $env:PYTORCH_FINAL_PACKAGE_DIR)) {
    New-Item -Path $env:PYTORCH_FINAL_PACKAGE_DIR -ItemType Directory -Force | Out-Null
}

# Set script helpers directory
$env:SCRIPT_HELPERS_DIR = Join-Path $ScriptParentDir "win-test-helpers\arm64"

# Run the main build script
& "$env:SCRIPT_HELPERS_DIR\build_pytorch.ps1"

Write-Host "BUILD PASSED"
