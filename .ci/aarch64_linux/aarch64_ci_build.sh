#!/bin/bash
set -eux -o pipefail

GPU_ARCH_VERSION=${GPU_ARCH_VERSION:-}

# Set CUDA architecture lists to match x86 build_cuda.sh
if [[ "$GPU_ARCH_VERSION" == *"12.6"* ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;9.0"
elif [[ "$GPU_ARCH_VERSION" == *"12.8"* ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0;12.0"
elif [[ "$GPU_ARCH_VERSION" == *"13.0"* ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0;11.0;12.0+PTX"
fi

# Compress the fatbin with -compress-mode=size for CUDA 13
if [[ "$DESIRED_CUDA" == *"13"* ]]; then
    export TORCH_NVCC_FLAGS="-compress-mode=size"
    # Bundle ptxas into the cu13 wheel, see https://github.com/pytorch/pytorch/issues/163801
    export BUILD_BUNDLE_PTXAS=1
fi

SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
source $SCRIPTPATH/aarch64_ci_setup.sh

###############################################################################
# Run aarch64 builder python
###############################################################################
cd /
# adding safe directory for git as the permissions will be
# on the mounted pytorch repo
git config --global --add safe.directory /pytorch
pip install -r /pytorch/requirements.txt
pip install auditwheel==6.2.0 wheel
if [ "$DESIRED_CUDA" = "cpu" ]; then
    echo "BASE_CUDA_VERSION is not set. Building cpu wheel."
    python /pytorch/.ci/aarch64_linux/aarch64_wheel_ci_build.py --enable-mkldnn
else
    echo "BASE_CUDA_VERSION is set to: $DESIRED_CUDA"
    export USE_SYSTEM_NCCL=1

    # Check if we should use NVIDIA libs from PyPI (similar to x86 build_cuda.sh logic)
    if [[ -z "$PYTORCH_EXTRA_INSTALL_REQUIREMENTS" ]]; then
        echo "Bundling CUDA libraries with wheel for aarch64."
    else
        echo "Using nvidia libs from pypi for aarch64."
        echo "Updated PYTORCH_EXTRA_INSTALL_REQUIREMENTS for aarch64: $PYTORCH_EXTRA_INSTALL_REQUIREMENTS"
        export USE_NVIDIA_PYPI_LIBS=1
    fi

    python /pytorch/.ci/aarch64_linux/aarch64_wheel_ci_build.py --enable-mkldnn --enable-cuda
fi
