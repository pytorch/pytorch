#!/bin/bash
set -e

# Source common script to detect ROCM_PATH
source /detect_rocm_path.sh

# Determine if we're using TheRock nightly or traditional ROCm and configure paths
if [[ "${ROCM_VERSION}" == "nightly" ]]; then
    echo "Detected TheRock nightly installation, configuring paths dynamically..."
    echo "ROCm root path: $ROCM_PATH"
    
    # Export environment variables (ROCM_PATH, ROCM_HOME already set by detect_rocm_path.sh)
    export PATH="${ROCM_BIN}:${ROCM_PATH}/bin:${ROCM_PATH}/llvm/bin:${PATH}"
    export CMAKE_PREFIX_PATH="${ROCM_CMAKE}:${CMAKE_PREFIX_PATH:-}"
    
    # Note: MAGMA_HOME not set - theRock installations don't include MAGMA
    
    # Set for PyTorch build
    export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}"
    
    # Disable FBGEMM_GENAI for theRock nightly (not yet supported)
    export USE_FBGEMM_GENAI=0
    
    # Set ROCM_SOURCE_DIR for Kineto's CMake to find roctracer headers correctly
    # Kineto uses: ROCTRACER_INCLUDE_DIR = "${ROCM_SOURCE_DIR}/include/roctracer"
    export ROCM_SOURCE_DIR="${ROCM_PATH}"
    
    echo "TheRock ROCm environment configured:"
    echo "  ROCM_PATH=$ROCM_PATH"
    echo "  ROCM_HOME=$ROCM_HOME"
else
    echo "Using traditional ROCm installation from /opt/rocm"
    # Paths already set by detect_rocm_path.sh, just add to PATH
    export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:${PATH}"
    export MAGMA_HOME="${MAGMA_HOME:-/opt/rocm/magma}"
fi

# Execute the command passed to the container
exec "$@"

