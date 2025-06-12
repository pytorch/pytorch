# Import common functions/variables/setup
. "$PSScriptRoot\common.ps1"

# Check for optional components
Write-Host "Disabling CUDA"
$env:USE_CUDA = "0"

Invoke-ExternalCommand "$PSScriptRoot\internal\check_opts.bat"

# Change to parent directory if NIGHTLIES_PYTORCH_ROOT exists
if (Test-Path $env:NIGHTLIES_PYTORCH_ROOT) {
    Set-Location (Split-Path $env:NIGHTLIES_PYTORCH_ROOT -Parent)
}

Invoke-ExternalCommand "$PSScriptRoot\internal\copy_cpu.bat"
Invoke-ExternalCommand "$PSScriptRoot\internal\setup.bat"
