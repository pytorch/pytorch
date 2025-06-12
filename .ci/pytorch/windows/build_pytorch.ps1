# This script parses args, installs required libraries (MKL, Magma, libuv)
# and then delegates to cpu.bat, cuda126.bat, etc.

param(
    [string]$CudaVersion,
    [string]$PyTorchBuildVersion,
    [string]$PyTorchBuildNumber
)

# Abort on first error (similar to set -e)
$ErrorActionPreference = "Stop"

# Set environment variables from parameters if not already set
if ([string]::IsNullOrEmpty($env:CUDA_VERSION) -or
    [string]::IsNullOrEmpty($env:PYTORCH_BUILD_VERSION) -or
    [string]::IsNullOrEmpty($env:PYTORCH_BUILD_NUMBER)) {

    if ([string]::IsNullOrEmpty($CudaVersion) -or
        [string]::IsNullOrEmpty($PyTorchBuildVersion) -or
        [string]::IsNullOrEmpty($PyTorchBuildNumber)) {
        Write-Host "Illegal number of parameters. Pass cuda version, pytorch version, build number"
        Write-Host "CUDA version should be Mm with no dot, e.g. '80'"
        Write-Host "DESIRED_PYTHON should be M.m, e.g. '2.7'"
        exit 1
    }

    $env:CUDA_VERSION = $CudaVersion
    $env:PYTORCH_BUILD_VERSION = $PyTorchBuildVersion
    $env:PYTORCH_BUILD_NUMBER = $PyTorchBuildNumber
}

$CUDA_PREFIX = "cuda$($env:CUDA_VERSION)"
if ($env:CUDA_VERSION -eq "cpu") { $CUDA_PREFIX = "cpu" }
if ($env:CUDA_VERSION -eq "xpu") { $CUDA_PREFIX = "xpu" }

if ([string]::IsNullOrEmpty($env:DESIRED_PYTHON)) {
    $env:DESIRED_PYTHON = "3.5;3.6;3.7"
}

$DESIRED_PYTHON_PREFIX = $env:DESIRED_PYTHON -replace '\.', ''
$DESIRED_PYTHON_PREFIX = "py" + ($DESIRED_PYTHON_PREFIX -replace ';', ';py')

$SRC_DIR = $PSScriptRoot
Push-Location $SRC_DIR

$ORIG_PATH = $env:PATH

# Setup build environment with retry logic
$setupSuccess = $false

foreach ($attempt in 1..3) {
    try {
        & .\setup_build.bat
        if ($LASTEXITCODE -eq 0) {
            $setupSuccess = $true
            break
        }
    } catch {
        # Continue to next attempt
    }
}

if (-not $setupSuccess) {
    Write-Host "Failed to setup build environment"
    exit 1
}

# Download MAGMA Files on CUDA builds
$MAGMA_VERSION = "2.5.4"

if ($env:DEBUG -eq "1") {
    $BUILD_TYPE = "debug"
} else {
    $BUILD_TYPE = "release"
}

if ($env:CUDA_VERSION -ne "cpu" -and $env:CUDA_VERSION -ne "xpu") {
    $magmaDir = "magma_$($CUDA_PREFIX)_$($BUILD_TYPE)"
    $magmaArchive = "$magmaDir.7z"

    if (Test-Path $magmaDir) {
        Remove-Item -Recurse -Force $magmaDir
    }
    if (Test-Path $magmaArchive) {
        Remove-Item -Force $magmaArchive
    }

    $magmaUrl = "https://s3.amazonaws.com/ossci-windows/magma_$($MAGMA_VERSION)_$($CUDA_PREFIX)_$($BUILD_TYPE).7z"
    curl -k $magmaUrl -o $magmaArchive
    7z x -aoa $magmaArchive -o$magmaDir
}

# Install sccache
if ($env:USE_SCCACHE -eq "1") {
    $tmpBinDir = Join-Path $PWD "tmp_bin"
    if (-not (Test-Path $tmpBinDir)) {
        New-Item -ItemType Directory -Path $tmpBinDir | Out-Null
    }

    curl -k https://s3.amazonaws.com/ossci-windows/sccache.exe --output (Join-Path $tmpBinDir "sccache.exe")
    curl -k https://s3.amazonaws.com/ossci-windows/sccache-cl.exe --output (Join-Path $tmpBinDir "sccache-cl.exe")

    if (-not [string]::IsNullOrEmpty($env:CUDA_VERSION)) {
        $env:ADDITIONAL_PATH = $tmpBinDir
        $env:SCCACHE_IDLE_TIMEOUT = "1500"

        # randomtemp is used to resolve the intermittent build error related to CUDA.
        # code: https://github.com/peterjc123/randomtemp-rust
        # issue: https://github.com/pytorch/pytorch/issues/25393
        #
        # CMake requires a single command as CUDA_NVCC_EXECUTABLE, so we push the wrappers
        # randomtemp.exe and sccache.exe into a batch file which CMake invokes.

        $randomTempPath = Join-Path $tmpBinDir "randomtemp.exe"
        curl -kL https://github.com/peterjc123/randomtemp-rust/releases/download/v0.4/randomtemp.exe --output $randomTempPath

        $nvccBatPath = Join-Path $tmpBinDir "nvcc.bat"
        $nvccBatContent = "@`"$randomTempPath`" `"$(Join-Path $tmpBinDir 'sccache.exe')`" `"$($env:CUDA_PATH)\bin\nvcc.exe`" %*"
        Set-Content -Path $nvccBatPath -Value $nvccBatContent

        Write-Host "Contents of nvcc.bat:"
        Get-Content $nvccBatPath

        $env:CUDA_NVCC_EXECUTABLE = $nvccBatPath

        # CMake doesn't accept back-slashes in the path
        $cygpathResult = & cygpath -m "$($env:CUDA_PATH)\bin\nvcc.exe"
        $env:CMAKE_CUDA_COMPILER = $cygpathResult
        $env:CMAKE_CUDA_COMPILER_LAUNCHER = "$randomTempPath;$(Join-Path $tmpBinDir 'sccache.exe')"
    }
}

$env:PYTORCH_BINARY_BUILD = "1"
$env:TH_BINARY_BUILD = "1"
$env:INSTALL_TEST = "0"

# Split and process each Python version
$pythonVersions = $DESIRED_PYTHON_PREFIX -split ';'

foreach ($pythonVersion in $pythonVersions) {
    # Set Environment vars for the build
    $env:CMAKE_PREFIX_PATH = "$PWD\Python\Library\;$($env:PATH)"
    $env:PYTHON_LIB_PATH = "$PWD\Python\Library\bin"

    if (-not [string]::IsNullOrEmpty($env:ADDITIONAL_PATH)) {
        $env:PATH = "$($env:ADDITIONAL_PATH);$($env:PATH)"
    }

    pip install ninja

    # Set Flags
    if ($env:CUDA_VERSION -ne "cpu" -and $env:CUDA_VERSION -ne "xpu") {
        $env:MAGMA_HOME = Join-Path $PWD "magma_$($CUDA_PREFIX)_$($BUILD_TYPE)"
    }

    Write-Host "Calling arch build script"
    $buildScript = "$CUDA_PREFIX.bat"
    & .\$buildScript

    if ($LASTEXITCODE -ne 0) {
        exit 1
    }
}

$env:PATH = $ORIG_PATH
Pop-Location

if ($LASTEXITCODE -ne 0) {
    exit 1
}
