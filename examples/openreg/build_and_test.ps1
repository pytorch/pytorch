# PowerShell script to build and run OpenReg tests on Windows
# Usage: .\build_and_test.ps1 [test_name]
# Example: .\build_and_test.ps1 test_device.py
#          .\build_and_test.ps1 test_device.py::TestDevice::test_device_count

param(
    [string]$TestPath = "test_device.py",
    [switch]$Rebuild = $false,
    [switch]$Verbose = $false
)

Write-Host "OpenReg Build and Test Helper (Windows)" -ForegroundColor Cyan

# Get the repo root
$RepoRoot = Resolve-Path (Split-Path -Parent $PSScriptRoot)
$OpenRegDir = Join-Path $RepoRoot "test/cpp_extensions/open_registration_extension"
$ExtDir = Join-Path $OpenRegDir "torch_openreg"

Write-Host "Repo root: $RepoRoot" -ForegroundColor Gray
Write-Host "OpenReg dir: $OpenRegDir" -ForegroundColor Gray

# Check prerequisites
Write-Host "`n[1/4] Checking prerequisites..." -ForegroundColor Yellow

$checks = @(
    @{ Name = "cmake"; Command = "cmake --version" },
    @{ Name = "ninja"; Command = "ninja --version" },
    @{ Name = "python"; Command = "python --version" }
)

foreach ($check in $checks) {
    try {
        $output = & $check.Command 2>$null
        Write-Host "✓ $($check.Name) found: $output" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ $($check.Name) NOT found" -ForegroundColor Red
        Write-Host "  Install and add to PATH, then retry" -ForegroundColor Red
        exit 1
    }
}

# Build the extension (or skip if already built)
Write-Host "`n[2/4] Building OpenReg extension..." -ForegroundColor Yellow

if ($Rebuild -or -not (Test-Path "$ExtDir/lib")) {
    Write-Host "  Building (this may take a few minutes)..." -ForegroundColor Gray
    
    Push-Location $OpenRegDir
    try {
        python setup.py build_ext --inplace 2>&1 | Tee-Object -FilePath "build.log" | Select-Object -Last 5
        if ($LASTEXITCODE -ne 0) {
            Write-Host "✗ Build failed. See build.log for details." -ForegroundColor Red
            exit 1
        }
        Write-Host "✓ Build successful" -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Host "  (Using cached build, add -Rebuild to force rebuild)" -ForegroundColor Gray
}

# Verify import
Write-Host "`n[3/4] Verifying OpenReg import..." -ForegroundColor Yellow

$importTest = @"
import torch
import sys
sys.path.insert(0, '$OpenRegDir')
try:
    import torch_openreg
    print(f'✓ torch_openreg imported from: {torch_openreg.__file__}')
    count = torch.accelerator.device_count()
    print(f'✓ Device count: {count}')
except ImportError as e:
    print(f'✗ Import failed: {e}')
    sys.exit(1)
"@

python -c $importTest
if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Import failed" -ForegroundColor Red
    exit 1
}

# Run tests
Write-Host "`n[4/4] Running tests..." -ForegroundColor Yellow
Write-Host "  Test path: $TestPath" -ForegroundColor Gray

$verboseFlag = if ($Verbose) { "-vv" } else { "-v" }
$testDir = Join-Path $OpenRegDir "torch_openreg/tests"

# Convert friendly test path to full path if needed
if (-not $TestPath.Contains("/") -and -not $TestPath.Contains("\")) {
    $TestPath = Join-Path $testDir $TestPath
}

Write-Host "  Running: pytest $TestPath $verboseFlag" -ForegroundColor Gray
Write-Host ""

python -m pytest $TestPath $verboseFlag --tb=short

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Tests passed!" -ForegroundColor Green
}
else {
    Write-Host "`n✗ Some tests failed" -ForegroundColor Red
    Write-Host "  For debugging, see failure_interpretation.md in docs/openreg/" -ForegroundColor Yellow
}

exit $LASTEXITCODE
