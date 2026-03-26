#!/usr/bin/env bash
# Install build dependencies inside the build container.
#
# Usage: build_install_deps.sh <package_dir>
#
# Environment variables:
#   DESIRED_CUDA - CUDA variant
#   GPU_ARCH_TYPE - GPU architecture type

set -ex

PACKAGE_DIR="$1"

retry () {
    "$@" || (sleep 1 && "$@") || (sleep 2 && "$@") || (sleep 4 && "$@") || (sleep 8 && "$@")
}

cd "$PACKAGE_DIR"

retry pip install -qUr requirements-build.txt
python setup.py clean
retry pip install -qr requirements.txt

# NumPy version selection based on Python version
# (keep in sync with .ci/manywheel/build_common.sh)
PYTHON_VERSION=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
case "$PYTHON_VERSION" in
    cp314*)
        retry pip install -q --pre numpy==2.3.4
        ;;
    cp31*)
        retry pip install -q --pre numpy==2.1.0
        ;;
    *)
        retry pip install -q --pre numpy==2.0.2
        ;;
esac

# ROCm: run AMD build script
if [[ "${DESIRED_CUDA:-}" == *"rocm"* ]]; then
    echo "Running build_amd.py at $(date)"
    python tools/amd_build/build_amd.py
fi
