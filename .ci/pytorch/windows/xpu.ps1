# Import common functions
. "$PSScriptRoot\common.ps1"

# The setup.py/MODULE_NAME check, clone/clean, and check_deps are already handled in common.ps1

# Check for optional components
Write-Host "Disabling CUDA"
$env:USE_CUDA = "0"

Invoke-ExternalCommand "$PSScriptRoot\internal\check_opts.bat"

Write-Host "Activate XPU Bundle env"
$env:VS2022INSTALLDIR = $env:VS15INSTALLDIR
$env:XPU_BUNDLE_ROOT = "${env:ProgramFiles(x86)}\Intel\oneAPI"

# Call Intel oneAPI environment setup scripts
Invoke-ExternalCommand "`"$env:XPU_BUNDLE_ROOT\compiler\latest\env\vars.bat`""
Invoke-ExternalCommand "`"$env:XPU_BUNDLE_ROOT\ocloc\latest\env\vars.bat`""

$env:USE_ONEMKL = "1"

# Check if NIGHTLIES_PYTORCH_ROOT exists and change directory if it does
if (Test-Path env:NIGHTLIES_PYTORCH_ROOT) {
    $nightliesParent = Split-Path $env:NIGHTLIES_PYTORCH_ROOT -Parent
    Set-Location $nightliesParent
}

Invoke-ExternalCommand "$PSScriptRoot\internal\copy_cpu.bat"
Invoke-ExternalCommand "$PSScriptRoot\internal\setup.bat"
