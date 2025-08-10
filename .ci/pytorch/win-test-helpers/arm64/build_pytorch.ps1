# TODO: we may can use existing build_pytorch.bat for arm64

if ($env:DEBUG -eq "1") {
    $env:BUILD_TYPE = "debug"
} else {
    $env:BUILD_TYPE = "release"
}

# This inflates our log size slightly, but it is REALLY useful to be
# able to see what our cl.exe commands are. (since you can actually
# just copy-paste them into a local Windows setup to just rebuild a
# single file.)
# log sizes are too long, but leaving this here in case someone wants to use it locally
# $env:CMAKE_VERBOSE_MAKEFILE = "1"

$env:INSTALLER_DIR = Join-Path $env:SCRIPT_HELPERS_DIR "installation-helpers"

cd ..

# Environment variables
$env:SCCACHE_IDLE_TIMEOUT = "0"
$env:SCCACHE_IGNORE_SERVER_IO_ERROR = "1"
$env:CMAKE_BUILD_TYPE = $env:BUILD_TYPE
$env:CMAKE_C_COMPILER_LAUNCHER = "sccache"
$env:CMAKE_CXX_COMPILER_LAUNCHER = "sccache"
$env:libuv_ROOT = Join-Path $env:DEPENDENCIES_DIR "libuv\install"
$env:MSSdk = "1"

if ($env:PYTORCH_BUILD_VERSION) {
    $env:PYTORCH_BUILD_VERSION = $env:PYTORCH_BUILD_VERSION
    $env:PYTORCH_BUILD_NUMBER = "1"
}

$env:CMAKE_POLICY_VERSION_MINIMUM = "3.5"

# Set BLAS type
if ($env:ENABLE_APL -eq "1") {
    $env:BLAS = "APL"
    $env:USE_LAPACK = "1"
} elseif ($env:ENABLE_OPENBLAS -eq "1") {
    $env:BLAS = "OpenBLAS"
    $env:OpenBLAS_HOME = Join-Path $env:DEPENDENCIES_DIR "OpenBLAS\install"
}

# Change to source directory
Set-Location $env:PYTORCH_ROOT

# Copy libuv.dll
Copy-Item -Path (Join-Path $env:libuv_ROOT "lib\Release\uv.dll") -Destination "torch\lib\uv.dll" -Force

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
where.exe python

# Python install dependencies
python -m pip install --upgrade pip
pip install setuptools pyyaml
pip install -r requirements.txt

# Set after installing psutil
$env:DISTUTILS_USE_SDK = "1"

# Print all environment variables
Get-ChildItem Env:

# Start and inspect sccache
sccache --start-server
sccache --zero-stats
sccache --show-stats

# Build the wheel
python setup.py bdist_wheel
if ($LASTEXITCODE -ne 0) { exit 1 }

# Install the wheel locally
$whl = Get-ChildItem -Path "dist\*.whl" | Select-Object -First 1
if ($whl) {
    python -mpip install --no-index --no-deps $whl.FullName
}

# Copy final wheel
robocopy "dist" "$env:PYTORCH_FINAL_PACKAGE_DIR" *.whl

# Export test times
python tools/stats/export_test_times.py

# Copy additional CI files
robocopy ".additional_ci_files" "$env:PYTORCH_FINAL_PACKAGE_DIR\.additional_ci_files" /E

# Save ninja log
Copy-Item -Path "build\.ninja_log" -Destination $env:PYTORCH_FINAL_PACKAGE_DIR -Force

# Final sccache stats and stop
sccache --show-stats
sccache --stop-server

exit 0
