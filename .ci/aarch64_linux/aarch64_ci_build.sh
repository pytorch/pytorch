#!/bin/bash
set -eux -o pipefail

GPU_ARCH_VERSION=${GPU_ARCH_VERSION:-}

if [[ "$GPU_ARCH_VERSION" == *"12.9"* ]]; then
    export TORCH_CUDA_ARCH_LIST="8.0;9.0;10.0;12.0"
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
    #USE_PRIORITIZED_TEXT_FOR_LD for enable linker script optimization https://github.com/pytorch/pytorch/pull/121975/files
    USE_PRIORITIZED_TEXT_FOR_LD=1 python /pytorch/.ci/aarch64_linux/aarch64_wheel_ci_build.py --enable-mkldnn --enable-cuda
fi
