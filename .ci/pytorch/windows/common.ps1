# Helper function for automatic exit code checking
function Invoke-ExternalCommand {
    param([string]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command '$Command' failed with exit code $LASTEXITCODE"
    }
}

$MODULE_NAME = "pytorch"

# Check if setup.py or MODULE_NAME directory exists
if (-not (Test-Path "setup.py") -and -not (Test-Path $MODULE_NAME)) {
    Invoke-ExternalCommand "$PSScriptRoot\internal\clone.bat"
    Set-Location $PSScriptRoot
} else {
    Invoke-ExternalCommand "$PSScriptRoot\internal\clean.bat"
}

# Check dependencies
Invoke-ExternalCommand "$PSScriptRoot\internal\check_deps.bat"

