#!/bin/bash
set -e

# Function to get ROCm path dynamically using rocm_sdk
get_rocm_path() {
    local path_name="$1"
    python3 -c "
import sys
import subprocess
from pathlib import Path

def capture(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()

try:
    path = Path(capture([sys.executable, '-m', 'rocm_sdk', 'path', f'--{path_name}']))
    print(str(path))
except Exception as e:
    print('', file=sys.stderr)
    sys.exit(1)
"
}

# Determine if we're using TheRock nightly or traditional ROCm
if [[ "${ROCM_VERSION}" == "nightly" ]]; then
    echo "Detected TheRock nightly installation, configuring paths dynamically..."
    
    # Check if rocm_sdk is available
    if python3 -m rocm_sdk path --root &>/dev/null; then
        # Get ROCm paths dynamically
        ROCM_PATH=$(get_rocm_path "root")
        ROCM_BIN=$(get_rocm_path "bin")
        ROCM_CMAKE=$(get_rocm_path "cmake")
        
        if [[ -n "$ROCM_PATH" ]]; then
            echo "ROCm root path: $ROCM_PATH"
            
            # Export environment variables
            export ROCM_PATH="$ROCM_PATH"
            export ROCM_HOME="$ROCM_PATH"
            export PATH="${ROCM_BIN}:${ROCM_PATH}/bin:${ROCM_PATH}/hcc/bin:${ROCM_PATH}/hip/bin:${ROCM_PATH}/opencl/bin:${ROCM_PATH}/llvm/bin:${PATH}"
            export MAGMA_HOME="${ROCM_PATH}/magma"
            export CMAKE_PREFIX_PATH="${ROCM_CMAKE}:${CMAKE_PREFIX_PATH:-}"
            
            # Set for PyTorch build
            export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}"
            
            echo "TheRock ROCm environment configured:"
            echo "  ROCM_PATH=$ROCM_PATH"
            echo "  ROCM_HOME=$ROCM_HOME"
            echo "  MAGMA_HOME=$MAGMA_HOME"
        fi
    fi
else
    echo "Using traditional ROCm installation from /opt/rocm"
    # Use traditional ROCm paths
    export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
    export PATH="/opt/rocm/bin:/opt/rocm/hcc/bin:/opt/rocm/hip/bin:/opt/rocm/opencl/bin:/opt/rocm/llvm/bin:${PATH}"
    export MAGMA_HOME="${MAGMA_HOME:-/opt/rocm/magma}"
fi

# Execute the command passed to the container
exec "$@"

