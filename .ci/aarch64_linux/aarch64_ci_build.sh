#!/bin/bash
set -eux -o pipefail

GPU_ARCH_VERSION=${GPU_ARCH_VERSION:-}

if [[ "$GPU_ARCH_VERSION" == *"12.9"* ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0;12.0"
fi

if [[ "$GPU_ARCH_VERSION" == *"13.0"* ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0;11.0;12.0"
fi

# Compress the fatbin with -compress-mode=size for CUDA 13
if [[ "$DESIRED_CUDA" == *"13"* ]]; then
    export TORCH_NVCC_FLAGS="-compress-mode=size"
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
pip install auditwheel==6.2.0
if [ "$DESIRED_CUDA" = "cpu" ]; then
    echo "BASE_CUDA_VERSION is not set. Building cpu wheel."
    #USE_PRIORITIZED_TEXT_FOR_LD for enable linker script optimization https://github.com/pytorch/pytorch/pull/121975/files
    USE_PRIORITIZED_TEXT_FOR_LD=1 python /pytorch/.ci/aarch64_linux/aarch64_wheel_ci_build.py --enable-mkldnn
else
    echo "BASE_CUDA_VERSION is set to: $DESIRED_CUDA"
    export USE_SYSTEM_NCCL=1
    
    # Check if we should use NVIDIA libs from PyPI (similar to x86 build_cuda.sh logic)
    if [[ -z "$PYTORCH_EXTRA_INSTALL_REQUIREMENTS" ]]; then
        echo "Bundling CUDA libraries with wheel for aarch64."
    else
        echo "Using nvidia libs from pypi for aarch64."
        
        # Fix platform constraints in PYTORCH_EXTRA_INSTALL_REQUIREMENTS for aarch64
        # Replace 'platform_machine == "x86_64"' with 'platform_machine == "aarch64"'
        export PYTORCH_EXTRA_INSTALL_REQUIREMENTS="${PYTORCH_EXTRA_INSTALL_REQUIREMENTS//platform_machine == \'x86_64\'/platform_machine == \'aarch64\'}"
        fi
        
        echo "Updated PYTORCH_EXTRA_INSTALL_REQUIREMENTS for aarch64: $PYTORCH_EXTRA_INSTALL_REQUIREMENTS"
        CUDA_RPATHS=(
            '$ORIGIN/../../nvidia/cublas/lib'
            '$ORIGIN/../../nvidia/cuda_cupti/lib'
            '$ORIGIN/../../nvidia/cuda_nvrtc/lib'
            '$ORIGIN/../../nvidia/cuda_runtime/lib'
            '$ORIGIN/../../nvidia/cudnn/lib'
            '$ORIGIN/../../nvidia/cufft/lib'
            '$ORIGIN/../../nvidia/curand/lib'
            '$ORIGIN/../../nvidia/cusolver/lib'
            '$ORIGIN/../../nvidia/cusparse/lib'
            '$ORIGIN/../../nvidia/cusparselt/lib'
            '$ORIGIN/../../cusparselt/lib'
            '$ORIGIN/../../nvidia/nccl/lib'
            '$ORIGIN/../../nvidia/nvshmem/lib'
            '$ORIGIN/../../nvidia/nvtx/lib'
            '$ORIGIN/../../nvidia/cufile/lib'
        )
        CUDA_RPATHS=$(IFS=: ; echo "${CUDA_RPATHS[*]}")
        export C_SO_RPATH=$CUDA_RPATHS':$ORIGIN:$ORIGIN/lib'
        export LIB_SO_RPATH=$CUDA_RPATHS':$ORIGIN'
        export FORCE_RPATH="--force-rpath"
        export USE_STATIC_NCCL=0
        export ATEN_STATIC_CUDA=0
        export USE_CUDA_STATIC_LINK=0
        export USE_CUPTI_SO=1
        export USE_NVIDIA_PYPI_LIBS=1
    fi
    
    #USE_PRIORITIZED_TEXT_FOR_LD for enable linker script optimization https://github.com/pytorch/pytorch/pull/121975/files
    USE_PRIORITIZED_TEXT_FOR_LD=1 python /pytorch/.ci/aarch64_linux/aarch64_wheel_ci_build.py --enable-mkldnn --enable-cuda
fi
