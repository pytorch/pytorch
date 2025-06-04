#!/usr/bin/env bash

set -e  # Exit on error
set -o pipefail
set -x  # Print commands for debugging

# Determine build type
if [[ "$DEBUG" == "1" ]]; then
  BUILD_TYPE="debug"
else
  BUILD_TYPE="release"
fi

# Set installer directory
INSTALLER_DIR="$SCRIPT_HELPERS_DIR/installation-helpers"

# Environment variables
export SCCACHE_IDLE_TIMEOUT=0
export SCCACHE_IGNORE_SERVER_IO_ERROR=1
export CMAKE_BUILD_TYPE="$BUILD_TYPE"
export CMAKE_C_COMPILER_LAUNCHER="sccache"
export CMAKE_CXX_COMPILER_LAUNCHER="sccache"
export libuv_ROOT="$DEPENDENCIES_DIR/libuv/install"
export MSSdk=1
export CMAKE_POLICY_VERSION_MINIMUM=3.5

if [[ -n "$PYTORCH_BUILD_VERSION" ]]; then
  export PYTORCH_BUILD_VERSION="$PYTORCH_BUILD_VERSION"
  export PYTORCH_BUILD_NUMBER=1
fi

# Set BLAS type
if [[ "$ENABLE_APL" == "1" ]]; then
  export BLAS="APL"
  export USE_LAPACK=1
elif [[ "$ENABLE_OPENBLAS" == "1" ]]; then
  export BLAS="OpenBLAS"
  export OpenBLAS_HOME="$DEPENDENCIES_DIR/OpenBLAS/install"
fi

# Activate ARM64 cross-compilation toolchain or environment if needed
# (Placeholder — modify as per your cross-compilation setup)
# source /opt/toolchains/arm64/env.sh

# Change to PyTorch root directory
cd "$PYTORCH_ROOT"

# Copy uv.dll equivalent (if building for Windows target)
cp "$libuv_ROOT/lib/Release/uv.dll" torch/lib/uv.dll || true

# Create Python virtual environment
python -m venv .venv
echo "*" > .venv/.gitignore
source ./.venv/Scripts/activate
which python

# Install Python dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Set DISTUTILS_USE_SDK after psutil install
export DISTUTILS_USE_SDK=1

# Add link.exe to PATH
set PATH=C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Tools\MSVC\14.43.34808\bin\Hostarm64\arm64;%PATH%

# Print environment variables (debugging)
env

# Start sccache server and reset stats
sccache --start-server
sccache --zero-stats
sccache --show-stats

# Build wheel
python setup.py bdist_wheel

# Check build status
if [[ $? -ne 0 ]]; then
  echo "Build failed"
  exit 1
fi

# Install built wheel
WHEEL_FILE=$(ls dist/*.whl | head -n 1)
python -m pip install --no-index --no-deps "$WHEEL_FILE"

# Copy wheel to final package dir
cp -v dist/*.whl "$PYTORCH_FINAL_PACKAGE_DIR/"

# Export test times
python tools/stats/export_test_times.py

# Copy additional CI files
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR/.additional_ci_files"
cp -r .additional_ci_files/* "$PYTORCH_FINAL_PACKAGE_DIR/.additional_ci_files/"

# Copy .ninja_log
cp -v build/.ninja_log "$PYTORCH_FINAL_PACKAGE_DIR/"

# Show and stop sccache
sccache --show-stats
sccache --stop-server

exit 0
